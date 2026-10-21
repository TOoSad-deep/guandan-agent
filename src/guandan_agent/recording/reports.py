"""Explicit report refresh and deterministic evaluation over closed recordings."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
from pathlib import Path
from uuid import uuid4

from guandan_agent.clock import SystemClock
from guandan_agent.contracts import (
    ErrorInfo, EvaluationReport, ReportPage, ReportSummary, ReviewReport,
)
from guandan_agent.recording.query import QueryError, QueryService
from guandan_agent.recording.store import HistoryUnavailable, RecordingClosedError


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="reports", safe_message=message, related_event_id=None)


class ReportStore:
    """A bounded shelf; reads stay in memory and explicit writes never call models."""

    def __init__(self, queries: QueryService, *, max_reports_per_run: int = 100):
        self.queries = queries
        self.max_reports = max_reports_per_run
        self._reports: dict[str, dict[str, ReviewReport | EvaluationReport]] = {}
        self._errors: dict[str, list[ErrorInfo]] = {}
        self._mutation_lock = asyncio.Lock()

    def _directory(self, run_id: str) -> Path:
        # HTTP can select only a registered run, never a filesystem destination.
        archive = self.queries.archive_path(run_id)
        directory = archive / "reports"
        if not archive.is_dir() or archive.resolve() != archive or directory.is_symlink():
            raise QueryError("REPORT_DIRECTORY_UNAVAILABLE", 422, "The registered report directory is unavailable.")
        return directory

    def _check_idle(self) -> None:
        if self._mutation_lock.locked():
            raise QueryError("REPORT_BUSY", 409, "A report operation is already in progress. Please retry shortly.")

    async def refresh(self, run_id: str) -> ReportPage:
        directory = self._directory(run_id)
        self._check_idle()
        async with self._mutation_lock:
            await self.load_directory(run_id, directory)
            return self.list(run_id)

    async def evaluate(self, run_id: str) -> EvaluationReport:
        """Generate/reuse one single-run match report, without a provider or task."""
        from guandan_agent.offline.evaluate import evaluate
        from guandan_agent.offline.output import save_report

        source = self.queries.closed_record(run_id)
        directory = self._directory(run_id)
        self._check_idle()
        async with self._mutation_lock:
            await self.load_directory(run_id, directory)
            existing = [report for report in self._reports.get(run_id, {}).values()
                        if isinstance(report, EvaluationReport) and report.run_ids == (run_id,)
                        and report.unit == "match" and report.status in {"completed", "partial"}]
            if existing:
                return max(existing, key=lambda report: (report.created_at, report.report_id))
            if len(self._reports.get(run_id, {})) >= self.max_reports:
                raise QueryError("REPORT_LIMIT", 409, "The report shelf is full.")
            try:
                report = await asyncio.to_thread(
                    evaluate, (source,), report_id=str(uuid4()),
                    created_at=SystemClock().utc_now(), unit="match",
                )
            except HistoryUnavailable as exc:
                raise QueryError("HISTORY_UNAVAILABLE", 410, "The complete recording is unavailable for evaluation.",
                                 retained_from_seq=exc.retained_from_seq) from None
            except (ValueError, RecordingClosedError):
                raise QueryError("EVALUATION_UNAVAILABLE", 422, "The recording cannot be evaluated.") from None
            try:
                await save_report(directory / f"evaluation-{report.report_id}.json", report)
            except (OSError, ValueError):
                raise QueryError("REPORT_WRITE_FAILED", 503, "The statistics report could not be saved.") from None
            self.register(run_id, report)
            return report

    async def load_directory(self, run_id: str, directory: Path) -> None:
        self.queries.closed_record(run_id)

        def read():
            values, failures = [], 0
            if directory.is_symlink():
                return values, 1
            if not directory.is_dir():
                return values, 0
            # One execution sidecar accompanies each generated model review.
            # It is not a report and must not consume the report shelf quota.
            paths = sorted(path for path in directory.glob("*.json")
                           if not path.name.endswith(".execution.json"))
            scan_limit = self.max_reports * 2
            for path in paths[:scan_limit]:
                try:
                    if path.is_symlink() or path.stat().st_size > 512 * 1024:
                        failures += 1
                        continue
                    value = json.loads(path.read_bytes())
                    # Batch audit/probe files are evidence, not generated reports.
                    if isinstance(value, dict) and value.get("kind") in {"review", "evaluation"}:
                        if len(values) >= self.max_reports:
                            failures += 1
                            continue
                        values.append(value)
                except (OSError, ValueError):
                    failures += 1
            return values, failures + int(len(paths) > scan_limit)

        values, failures = await asyncio.to_thread(read)
        for value in values:
            try:
                report = (ReviewReport if value["kind"] == "review" else EvaluationReport).model_validate(value)
                self.register(run_id, report)
            except (ValueError, QueryError):
                failures += 1
        # A refresh describes its current scan, not every historical failure.
        self._errors[run_id] = ([_error("REPORT_UNAVAILABLE", "部分报告文件无法安全读取或超出加载上限。")]
                               if failures else [])

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
