"""Seat-scoped, read-only projections of live caches and explicitly loaded logs.

Archive registration is the sole I/O entry point. HTTP GET handlers only read
bounded memory, and no query can construct a strategy, provider or platform.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from guandan_agent.contracts import (
    CandidateSnapshot, DecisionDetail, DecisionPage, DecisionProposal, DecisionSummary,
    ErrorInfo, ExecutionRecord, LogEvent, ModelCallResult, PlatformEvent, ReviewResult,
    RunListItem, RunSessionView, RunView, SessionView, VisibleState, RequestContext,
)
from guandan_agent.recording.store import ClosedRunLog
from guandan_agent.runtime.manager import RunManager, RunManagerError
from guandan_agent.runtime.state import StateConflict, TableState


class QueryError(LookupError):
    def __init__(self, code: str, status_code: int, safe_message: str,
                 *, retained_from_seq: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        self.status = status_code
        self.safe_message = safe_message
        self.retained_from_seq = retained_from_seq
        super().__init__(safe_message)


def _history(retained_from_seq: int | None = None) -> QueryError:
    return QueryError("HISTORY_UNAVAILABLE", 410, "The requested decision evidence is unavailable.",
                      retained_from_seq=retained_from_seq)


def _manager_error(error: RunManagerError) -> QueryError:
    status = 410 if error.error.code == "HISTORY_UNAVAILABLE" else 404
    return QueryError(error.error.code, status, error.error.safe_message)


T = TypeVar("T", bound=BaseModel)


def _typed(kind: type[T], value: Any) -> T:
    # Persisted arrays are immutable tuples/maps; JSON hydration preserves the
    # same strict validation rules as the original serialized contract.
    def plain(item: Any) -> Any:
        from collections.abc import Mapping
        if isinstance(item, Mapping):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        return item
    return kind.model_validate_json(json.dumps(plain(value)))


@dataclass(frozen=True)
class _Archive:
    record: ClosedRunLog
    view: RunView
    sessions: dict[str, SessionView]
    events: dict[str, tuple[LogEvent, ...]]
    label: str


class QueryService:
    def __init__(self, manager: RunManager, *, cache_events: int = 20000) -> None:
        if type(cache_events) is not int or cache_events < 1:
            raise ValueError("cache_events must be positive")
        self.manager = manager
        self.cache_events = cache_events
        self._archives: dict[str, _Archive] = {}
        self._paths: dict[Path, str] = {}

    async def register_archive(self, path: str | Path) -> RunView:
        """Explicit one-time load. Re-registering a path never refreshes files."""
        canonical = Path(path).resolve()
        if canonical in self._paths:
            return self._archives[self._paths[canonical]].view
        try:
            record = await ClosedRunLog.load(canonical, cache_events=self.cache_events)
        except (OSError, ValueError):
            raise QueryError("ARCHIVE_INVALID", 422, "The archive could not be loaded.") from None
        if record.run_id in self._archives:
            raise QueryError("ARCHIVE_CONFLICT", 409, "This run already has a registered archive.")
        archive = _archive(record)
        self._archives[record.run_id] = archive
        self._paths[canonical] = record.run_id
        return archive.view

    def list_runs(self) -> tuple[RunListItem, ...]:
        values: dict[str, RunListItem] = {}
        for item in self.manager.list_views():
            values[item.run_id] = RunListItem(**item.model_dump(),
                read_only=item.status in {"completed", "interrupted"},
                label=self.manager.config_label(item.run_id))
        for run_id, archive in self._archives.items():
            # A loaded old snapshot must never replace a currently live run.
            if run_id not in values:
                values[run_id] = RunListItem(**archive.view.model_dump(), read_only=True,
                                            label=archive.label)
        return tuple(values.values())

    def view(self, run_id: str) -> RunView:
        try:
            return self.manager.view(run_id)
        except RunManagerError as exc:
            if run_id in self._archives:
                return self._archives[run_id].view
            raise _manager_error(exc) from None

    def session_view(self, run_id: str, session_id: str) -> SessionView:
        self._session(run_id, session_id)
        try:
            return self.manager.session_view(run_id, session_id)
        except RunManagerError:
            archive = self._archives.get(run_id)
            if archive is None or session_id not in archive.sessions:
                raise QueryError("SESSION_NOT_FOUND", 404, "Session does not belong to this run.") from None
            return archive.sessions[session_id]

    def closed_record(self, run_id: str) -> ClosedRunLog:
        self.view(run_id)
        try:
            active = not self.manager.is_closed(run_id)
        except RunManagerError:
            active = False
        if active:
            raise QueryError("RUN_NOT_CLOSED", 409, "Reports require a closed recording.")
        archive = self._archives.get(run_id)
        if archive is None:
            raise QueryError("RECORD_NOT_REGISTERED", 409, "The closed record is not registered yet.")
        if not archive.record.health.closed:
            raise QueryError("RUN_NOT_CLOSED", 409, "Reports require a closed recording.")
        return archive.record

    def _session(self, run_id: str, session_id: str) -> RunSessionView:
        view = self.view(run_id)
        for session in view.sessions:
            if session.session_id == session_id:
                return session
        raise QueryError("SESSION_NOT_FOUND", 404, "Session does not belong to this run.")

    def _events(self, run_id: str, session_id: str) -> tuple[tuple[LogEvent, ...], int, bool]:
        owner = self._session(run_id, session_id)
        try:
            return self.manager.query_snapshot(run_id, session_id)
        except RunManagerError as exc:
            archive = self._archives.get(run_id)
            if archive is None:
                raise _manager_error(exc) from None
            return (tuple(event for event in archive.events.get(session_id, ())
                          if event.seat == owner.seat),
                    archive.record.retained_from_seq, archive.view.log_complete)

    def decisions(self, run_id: str, session_id: str, after_seq: int = 0,
                  limit: int = 50) -> DecisionPage:
        if type(after_seq) is not int or after_seq < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise QueryError("INVALID_QUERY", 422, "Pagination parameters are outside their bounds.")
        events, retained, complete = self._events(run_id, session_id)
        if after_seq < retained - 1:
            raise _history(retained)
        anchors = {event.request_id for event in events if event.event_type == "request_opened"}
        orphaned = [event for event in events if event.request_id is not None
                    and event.seq > after_seq and event.request_id not in anchors]
        if orphaned:
            # An absent request_opened is not an empty decision history. A
            # later execution/call must never be offered without its facts.
            raise _history(max(retained, max(e.seq for e in orphaned) + 1))
        opened = [event for event in events if event.event_type == "request_opened" and event.seq > after_seq]
        items = []
        for event in opened[:limit]:
            related = [row for row in events if row.request_id == event.request_id]
            try:
                detail = _detail(events, event.request_id, complete)
            except QueryError:
                raise _history(max(row.seq for row in related) + 1) from None
            items.append(DecisionSummary(
                request_id=event.request_id, opened_seq=event.seq,
                round_id=detail.state.binding.round_id, stage=detail.state.stage,
                source=detail.proposal.source if detail.proposal else None,
                execution_status=detail.execution.status if detail.execution else None,
                updated_at=related[-1].recorded_at,
                log_complete=detail.log_complete,
            ))
        return DecisionPage(items=tuple(items),
            next_after_seq=items[-1].opened_seq if items else None,
            has_more=len(opened) > limit, retained_from_seq=retained, log_complete=complete)

    def decision(self, run_id: str, session_id: str, request_id: str) -> DecisionDetail:
        events, retained, complete = self._events(run_id, session_id)
        if not re.fullmatch(re.escape(session_id) + r":q[1-9][0-9]*", request_id):
            raise QueryError("DECISION_NOT_FOUND", 404, "Decision does not belong to this session.")
        if not any(event.request_id == request_id for event in events):
            if retained > 1:
                raise _history(retained)
            raise QueryError("DECISION_NOT_FOUND", 404, "Decision was not found.")
        return _detail(events, request_id, complete)


def _detail(events: tuple[LogEvent, ...], request_id: str | None, complete: bool) -> DecisionDetail:
    rows = [event for event in events if event.request_id == request_id]
    opened = [row for row in rows if row.event_type == "request_opened"]
    if request_id is None or len(opened) != 1:
        raise _history()
    anchor = opened[0]
    try:
        context = _typed(RequestContext, anchor.payload["context"])
        state = context.state
        binding = state.binding
        if (binding.run_id, binding.session_id, binding.seat, binding.match_id, binding.request_id) != (
                anchor.run_id, anchor.session_id, anchor.seat, anchor.match_id, request_id):
            raise ValueError("foreign binding")
        candidates = proposal = review = execution = None
        calls: dict[str, ModelCallResult] = {}
        for row in rows:
            if row.seq < anchor.seq:
                raise ValueError("request order")
            if row.event_type == "candidates_prepared":
                candidates = _typed(CandidateSnapshot, row.payload["snapshot"])
                if candidates.binding != binding:
                    raise ValueError("foreign candidates")
            if row.event_type == "proposal_ready":
                proposal = _typed(DecisionProposal, row.payload["proposal"])
                if proposal.binding != binding:
                    raise ValueError("foreign proposal")
                review = proposal.review or review
            if row.event_type == "review_finished":
                if _typed(type(binding), row.payload["binding"]) != binding:
                    raise ValueError("foreign review")
                review = _typed(ReviewResult, row.payload["review"])
            if "execution" in row.payload:
                execution = _typed(ExecutionRecord, row.payload["execution"])
                if execution.binding != binding:
                    raise ValueError("foreign execution")
            if row.event_type == "model_call_finished":
                result = _typed(ModelCallResult, row.payload["result"])
                if result.binding != binding:
                    raise ValueError("foreign call")
                calls[result.call_id or f"uncalled:{row.seq}"] = result
            if row.event_type == "model_call_usage_updated":
                call_id = row.payload["call_id"]
                if call_id not in calls:
                    raise ValueError("missing original call")
                original = calls[call_id]
                calls[call_id] = _typed(ModelCallResult, {
                    **original.model_dump(mode="json"), "usage": row.payload["usage"],
                    "transport_requests": row.payload["transport_requests"],
                })
        candidate_ids = set() if candidates is None else {row.action_id for row in candidates.candidates}
        candidate_actions = tuple(action for action in context.legal.actions if action.action_id in candidate_ids)
        if len(candidate_actions) != len(candidate_ids):
            raise ValueError("candidate absent from original request")
        explanation = ()
        chosen_id = execution.action_id if execution else proposal.action_id if proposal else None
        chosen = next((action for action in context.legal.actions if action.action_id == chosen_id), None)
        if chosen is not None and (execution is None or execution.wire_action == chosen.wire_action):
            from guandan_agent.semantics import describe_candidate_effect
            explanation = describe_candidate_effect(state, chosen.wire_action)
        return DecisionDetail(state=state, candidates=candidates, proposal=proposal,
            review=review, execution=execution, calls=tuple(calls.values()), log_complete=complete,
            candidate_actions=candidate_actions, program_explanation=explanation)
    except (KeyError, TypeError, ValueError, ValidationError):
        raise _history() from None


def _archive(record: ClosedRunLog) -> _Archive:
    events = record.cached_events
    groups: dict[str, list[LogEvent]] = {}
    complete = record.health.log_complete
    status = "interrupted"
    error = record.summary.error
    for event in events:
        if event.event_type == "run_finished":
            candidate = event.payload.get("status")
            if candidate in {"completed", "interrupted"}:
                status = candidate
            if event.payload.get("error") is not None:
                try:
                    error = _typed(ErrorInfo, event.payload["error"])
                except (TypeError, ValueError, ValidationError):
                    complete = False
        if event.session_id and event.seat is not None and event.match_id:
            groups.setdefault(event.session_id, []).append(event)
    sessions = {}
    session_events = {}
    for session_id, rows in groups.items():
        owner = rows[0]
        if any((row.seat, row.match_id) != (owner.seat, owner.match_id) for row in rows):
            complete = False
            continue
        table = TableState(owner.seat)
        state_available = True
        last_rx = 0
        state_version = 0
        latest_execution = None
        decision_stage = None
        session_status = "interrupted"
        session_error = record.summary.error
        for row in rows:
            try:
                if row.event_type == "platform_received":
                    source = row.payload["event"]
                    if not isinstance(source, Mapping):
                        raise ValueError("invalid platform evidence shape")
                    relative = source.get("received_relative_ms", 0.0)
                    data = {key: source[key] for key in (
                        "session_id", "connection_epoch", "rx_seq", "received_at", "kind", "payload")}
                    data["received_mono"] = float(relative) / 1000.0
                    event = _typed(PlatformEvent, data)
                    if event.session_id != session_id or event.rx_seq != last_rx + 1:
                        state_available = False
                    last_rx = event.rx_seq
                    if state_available:
                        table.apply(event, row.event_id)
                if row.event_type == "request_opened":
                    state = _typed(VisibleState, row.payload["context"]["state"])
                    if (state.binding.run_id, state.binding.session_id, state.binding.seat) != (
                            record.run_id, session_id, owner.seat):
                        raise ValueError("foreign state")
                    state_version = state.binding.state_version
                    decision_stage = "deciding"
                if "execution" in row.payload:
                    value = _typed(ExecutionRecord, row.payload["execution"])
                    if (value.binding.run_id, value.binding.session_id, value.binding.seat) != (
                            record.run_id, session_id, owner.seat):
                        raise ValueError("foreign execution")
                    latest_execution = value
                    decision_stage = value.status
                if row.event_type == "session_completed":
                    session_status = "completed"
                if row.event_type == "session_interrupted":
                    session_status = "interrupted"
                    session_error = _typed(ErrorInfo, row.payload["error"])
            except (KeyError, TypeError, ValueError, ValidationError, StateConflict):
                state_available = False
        # Closed records never retain live decision authority.
        table_view = table.table_view() if state_available else None
        if table_view is not None and not complete:
            table.mark_degraded()
            table_view = table.table_view()
        sessions[session_id] = SessionView(
            run_id=record.run_id, match_id=owner.match_id, session_id=session_id,
            seat=owner.seat, round_id=max(table.round_id,
                                        max((row.round_id or 0 for row in rows), default=0)),
            state_version=state_version, view_version=rows[-1].seq,
            updated_at=rows[-1].recorded_at, status=session_status,
            state=table_view, active_request_id=None, decision_stage=decision_stage,
            latest_execution=latest_execution, log_complete=complete and state_available,
            error=session_error,
        )
        session_events[session_id] = tuple(rows)
    view = RunView(run_id=record.run_id, status=status, config_id=record.config_id,
        active_match_id=None, sessions=tuple(RunSessionView(
            session_id=session.session_id, match_id=session.match_id,
            seat=session.seat, status=session.status) for session in sessions.values()),
        log_complete=complete, error=error)
    label = record.config.run.experiment_id if record.config is not None else None
    return _Archive(record, view, sessions, session_events, label or "Archived run")
