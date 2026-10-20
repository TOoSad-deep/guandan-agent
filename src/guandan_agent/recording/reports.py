"""Read already-generated reports; loading never invokes a model or changes logs."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
from pathlib import Path

from guandan_agent.contracts import (
    ErrorInfo, EvaluationReport, ReportPage, ReportSummary, ReviewReport,
)
from guandan_agent.recording.query import QueryError, QueryService


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="reports", safe_message=message, related_event_id=None)


class ReportStore:
    """A small in-memory shelf populated explicitly at startup or by future P19."""

    def __init__(self, queries: QueryService, *, max_reports_per_run: int = 100):
        self.queries = queries
        self.max_reports = max_reports_per_run
        self._reports: dict[str, dict[str, ReviewReport | EvaluationReport]] = {}
        self._errors: dict[str, list[ErrorInfo]] = {}

    async def load_directory(self, run_id: str, directory: Path) -> None:
        self.queries.closed_record(run_id)

        def read():
            values, failures = [], 0
            if directory.is_symlink():
                return values, 1
            if not directory.is_dir():
                return values, 0
            paths = sorted(directory.glob("*.json"))
            for path in paths[:self.max_reports]:
                try:
                    if path.is_symlink() or path.stat().st_size > 512 * 1024:
                        failures += 1
                        continue
                    value = json.loads(path.read_bytes())
                    # Batch audit/probe files are evidence, not generated reports.
                    if isinstance(value, dict) and value.get("kind") in {"review", "evaluation"}:
                        values.append(value)
                except (OSError, ValueError):
                    failures += 1
            return values, failures + int(len(paths) > self.max_reports)

        values, failures = await asyncio.to_thread(read)
        for value in values:
            try:
                report = (ReviewReport if value["kind"] == "review" else EvaluationReport).model_validate(value)
                self.register(run_id, report)
            except (ValueError, QueryError):
                failures += 1
        if failures:
            self._errors.setdefault(run_id, []).append(_error("REPORT_UNAVAILABLE", "部分报告文件无法安全读取或超出加载上限。"))

    def register(self, run_id: str, report: ReviewReport | EvaluationReport) -> None:
        """P19 can publish its validated result here; this grants no model budget."""
        source = self.queries.closed_record(run_id)
        if isinstance(report, ReviewReport):
            if report.run_id != run_id or report.config_id != source.config_id:
                raise QueryError("REPORT_OWNERSHIP", 422, "Report does not belong to this source run.")
            # Copy + strict validation also rejects unsafe model_copy updates.
            report = ReviewReport.model_validate_json(report.model_dump_json())
            index = {e.event_id: e for e in source.cached_events}
            checked, missing = [], False
            errors = list(report.errors)
            for item in report.items:
                available = bool(item.evidence_event_ids) and source.summary.log_complete
                try:
                    detail = self.queries.decision(run_id, item.session_id, item.request_id)
                    binding = detail.state.binding
                    opened_seq = next((e.seq for e in source.cached_events
                                       if e.event_type == "request_opened" and e.session_id == item.session_id
                                       and e.request_id == item.request_id), None)
                    direct_reference = False
                    for identifier in item.evidence_event_ids:
                        event = index.get(identifier)
                        if event is None or event.session_id != item.session_id or event.seat != binding.seat:
                            available = False
                            continue
                        if event.request_id == item.request_id:
                            direct_reference = True
                            continue
                        # Later facts may cite a later public notification in the
                        # same seat's log; another request's private context is not allowed.
                        body = event.payload.get("event", {})
                        if not isinstance(body, Mapping):
                            body = {}
                        allowed_public = (body.get("kind") in {"play_observed", "round_result", "match_result"}
                                          and event.event_type == "platform_received")
                        if not (item.later_facts and allowed_public and opened_seq is not None
                                and event.seq > opened_seq):
                            available = False
                    available = available and direct_reference
                except QueryError:
                    available = False
                # Source ownership authorizes an evidence link, not factual
                # prose. Recompute program facts rather than trusting flags
                # serialized by a report writer (including this application).
                verified = False
                if (available and item.facts_source == "program"
                        and report.generation.get("prompt_version") in {"b09-offline-v2", "b09-offline-v3"}):
                    from guandan_agent.offline.review import verify_program_facts
                    verified = verify_program_facts(source, item)
                invalid_facts = not verified and (item.facts_source == "program" or item.facts_verified)
                if invalid_facts:
                    error = _error("PROGRAM_FACTS_UNVERIFIED", "报告声称的程序事实无法与源记录核对；保留原文，不能视为已核验事实。")
                    if error not in errors:
                        errors.append(error)
                missing = missing or not available or invalid_facts
                checked.append(item.model_copy(update={
                    "facts_verified": verified,
                    "evidence_available": available,
                    "evidence_error": None if available else _error("EVIDENCE_UNAVAILABLE", "证据缺失、归属不匹配或源记录不完整，不能打开原决策。"),
                }))
            report = report.model_copy(update={
                "items": tuple(checked),
                "errors": tuple(errors),
                "status": "partial" if missing and report.status == "completed" else report.status,
            })
        else:
            report = EvaluationReport.model_validate_json(report.model_dump_json())
            if run_id not in report.run_ids:
                raise QueryError("REPORT_OWNERSHIP", 422, "Evaluation does not include this source run.")
            for source_id, config_id in zip(report.run_ids, report.config_ids):
                if self.queries.closed_record(source_id).config_id != config_id:
                    raise QueryError("REPORT_OWNERSHIP", 422, "Evaluation source configuration does not match.")
        shelf = self._reports.setdefault(run_id, {})
        if report.report_id not in shelf and len(shelf) >= self.max_reports:
            raise QueryError("REPORT_LIMIT", 409, "The report shelf is full.")
        shelf[report.report_id] = report

    def list(self, run_id: str) -> ReportPage:
        source = self.queries.closed_record(run_id)
        items = sorted(self._reports.get(run_id, {}).values(), key=lambda r: (r.created_at, r.report_id))
        return ReportPage(items=tuple(ReportSummary(report_id=r.report_id, kind=r.kind,
                              status=r.status, created_at=r.created_at) for r in items),
                          log_complete=source.summary.log_complete,
                          errors=tuple(self._errors.get(run_id, ())))

    def get(self, run_id: str, report_id: str) -> ReviewReport | EvaluationReport:
        self.queries.closed_record(run_id)
        try:
            return self._reports[run_id][report_id]
        except KeyError:
            raise QueryError("REPORT_NOT_FOUND", 404, "Report was not found in this run.") from None
