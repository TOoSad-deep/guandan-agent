"""One local run owner; seats retain their own state and submission authority."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from guandan_agent.clock import Clock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    ErrorInfo, EventDraft, EventPage, EventQueryScope, LogEvent, RunSessionView,
    RunView, SeatConnection, SessionView, OfflineTaskView, ReviewReport,
)
from guandan_agent.hand import CHEAP_RULE_VERSION, FINAL_RULE_VERSION, CANDIDATE_VERSION, HandTools
from guandan_agent.platform.adapter import PlatformPort
from guandan_agent.recording import ClosedRunLog, EventStore
from guandan_agent.runtime.budget import RunCallBudget
from guandan_agent.runtime.offline import OfflineCallPort, OfflineReviews, OfflineReviewError


RULES_VERSION = "e92db7fa9d915fb318b3be78703d5ccd4a09c887"
ADAPTER_VERSION = "openguandan-e92db7f-v1"


def _source_metadata() -> dict[str, Any]:
    """Hash this installed package only; never inspect the workspace or env."""
    package = Path(__file__).resolve().parents[1]
    try:
        hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(package.rglob("*.py"))}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() if hashes else None
    except OSError:
        hashes, digest = {}, None
    return {"python_version": sys.version.split()[0], "source_sha256": digest,
            "source_files_sha256": hashes, "seed": None}


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="runtime", safe_message=message,
                     related_event_id=None)


class RunManagerError(RuntimeError):
    """A stable error safe for CLI and the future HTTP boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.error = _error(code, message)
        super().__init__(message)


@dataclass
class _OwnedSession:
    match_id: str
    connection: SeatConnection
    session: Any
    task: asyncio.Task[None]


@dataclass
class _Run:
    run_id: str
    config: RunConfig
    status: str = "starting"
    active_match_id: str | None = None
    sessions: list[_OwnedSession] = field(default_factory=list)
    recorder: Any = None
    task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    stop_requested: bool = False
    started: bool = False
    log_complete: bool = True
    error: ErrorInfo | None = None
    tools: Any = None
    provider: Any = None
    gateway: Any = None
    policy: Any = None
    call_budget: Any = None


def _positive_seconds(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _consume(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


class RunManager:
    """Synchronous start/stop claims, asynchronous bounded resource cleanup.

    All methods belong to one running asyncio loop. A policy never receives this
    object. Read APIs select a run and (for private state) exactly one session.
    """

    def __init__(
        self, *, platform_factory: Callable[[SeatConnection], PlatformPort],
        clock: Clock, scheduler: Any, recording_root: str | Path = "runs",
        session_factory: Callable[..., Any] | None = None,
        recorder_factory: Callable[..., Awaitable[Any]] | None = None,
        tools_factory: Callable[[RunConfig], Any] | None = None,
        provider_factory: Callable[[RunConfig], Any] | None = None,
        startup_timeout_s: float = 15.0, shutdown_timeout_s: float = 2.0,
    ) -> None:
        self._platform_factory = platform_factory
        self._clock = clock
        self._scheduler = scheduler
        self._recording_root = Path(recording_root)
        if session_factory is None:
            from guandan_agent.runtime.session import Session
            session_factory = Session
        self._session_factory = session_factory
        self._recorder_factory = recorder_factory or EventStore.create
        self._tools_factory = tools_factory or (lambda _: HandTools(clock=self._clock))
        self._provider_factory = provider_factory
        self._startup_timeout_s = _positive_seconds(startup_timeout_s, "startup_timeout_s")
        self._shutdown_timeout_s = _positive_seconds(shutdown_timeout_s, "shutdown_timeout_s")
        self._runs: dict[str, _Run] = {}
        self._active_run_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reviews = OfflineReviews(
            clock=clock, provider_factory=provider_factory,
            online_active=lambda: self._active_run_id is not None,
            shutdown_timeout_s=self._shutdown_timeout_s,
        )

    def _check_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RunManagerError("RUN_LOOP_CONFLICT", "RunManager belongs to a different event loop.")
        self._loop = loop
        return loop

    def _get(self, run_id: str) -> _Run:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunManagerError("RUN_NOT_FOUND", "Run was not found.") from None

    def start(self, config: RunConfig) -> RunView:
        loop = self._check_loop()
        if self._active_run_id is not None:
            raise RunManagerError("RUN_CONFLICT", "Another run is active.")
        if not isinstance(config, RunConfig):
            raise RunManagerError("INVALID_CONFIG", "A validated RunConfig is required.")
        # Revalidate even an unsafe model_copy update, then retain a fresh frozen
        # tree. No environment, credentials, filesystem or transport is read here.
        frozen = RunConfig.model_validate_json(config.model_dump_json())
        wants_model = any(seat.policy in {"V1", "V2"} for seat in frozen.run.seats)
        wants_review = any(seat.policy == "V2" for seat in frozen.run.seats)
        if wants_model and self._provider_factory is None:
            raise RunManagerError("POLICY_UNAVAILABLE", "Model policies require an explicit provider.")
        legacy = frozen.versions.policy == CHEAP_RULE_VERSION
        if (frozen.platform.rules_version != RULES_VERSION
                or frozen.platform.adapter_version != ADAPTER_VERSION
                or frozen.versions.policy not in (CHEAP_RULE_VERSION, FINAL_RULE_VERSION)
                or frozen.versions.fallback != CHEAP_RULE_VERSION
                or (legacy and wants_model)
                or (not legacy and (frozen.versions.candidate != CANDIDATE_VERSION
                                    or frozen.versions.prompt not in {"b05-main-v1", "b07-review-v1", "b09-facts-v2", "b09-facts-v3"}))
                or (wants_review and frozen.versions.prompt not in {"b07-review-v1", "b09-facts-v2", "b09-facts-v3"})):
            raise RunManagerError("VERSION_UNSUPPORTED", "Configured rules or policy version is unsupported.")
        # Online work has priority. Revoke offline qualification synchronously;
        # neither report generation nor a remote cancellation delays this start.
        self._reviews.stop_active()
        run_id = str(uuid4())
        run = _Run(run_id, frozen)
        self._runs[run_id] = run
        self._active_run_id = run_id
        run.task = loop.create_task(self._drive(run), name=f"run-{run_id}")
        return self.view(run_id)

    def start_review(
        self, config: RunConfig, source: ClosedRunLog,
        job: Callable[[OfflineCallPort], Awaitable[ReviewReport]],
    ) -> OfflineTaskView:
        self._check_loop()
        try:
            return self._reviews.start(config, source, job)
        except OfflineReviewError as exc:
            raise RunManagerError(exc.error.code, exc.error.safe_message) from None

    def review_view(self, task_id: str) -> OfflineTaskView:
        try:
            return self._reviews.view(task_id)
        except OfflineReviewError as exc:
            raise RunManagerError(exc.error.code, exc.error.safe_message) from None

    async def wait_review(self, task_id: str) -> ReviewReport:
        self._check_loop()
        try:
            return await self._reviews.wait(task_id)
        except OfflineReviewError as exc:
            raise RunManagerError(exc.error.code, exc.error.safe_message) from None

    def stop_review(self, task_id: str, reason: str = "Offline review was stopped.") -> OfflineTaskView:
        self._check_loop()
        try:
            return self._reviews.stop(task_id, reason)
        except OfflineReviewError as exc:
            raise RunManagerError(exc.error.code, exc.error.safe_message) from None

    def offline_usage(self) -> dict[str, int]:
        return self._reviews.usage()

    async def close_reviews(self) -> None:
        self._check_loop()
        await self._reviews.close()

    def stop(self, run_id: str) -> RunView:
        self._check_loop()
        run = self._get(run_id)
        if run.done.is_set() or run.stop_requested or run.status == "completed":
            return self.view(run_id)
        run.stop_requested = True
        run.status = "stopping"
        run.error = run.error or _error("RUN_STOPPED", "Run was stopped locally.")
        self._stop_sessions(run, "run_stopped")
        # Do not cancel an unstarted driver: its finally block owns release and
        # completion. The flag is checked before its first resource operation.
        if run.task is not None and run.started:
            run.task.cancel()
        return self.view(run_id)

    async def wait(self, run_id: str) -> RunView:
        self._check_loop()
        run = self._get(run_id)
        await run.done.wait()
        return self.view(run_id)

    def view(self, run_id: str) -> RunView:
        run = self._get(run_id)
        complete = run.log_complete
        if run.recorder is not None:
            complete = complete and run.recorder.health.log_complete
        return RunView(
            run_id=run.run_id, status=run.status, config_id=run.config.config_id,
            active_match_id=run.active_match_id,
            sessions=tuple(RunSessionView(
                session_id=owned.connection.session_id, match_id=owned.match_id,
                seat=owned.connection.seat, status=owned.session.status,
            ) for owned in run.sessions), log_complete=complete, error=run.error,
        )

    def session_view(self, run_id: str, session_id: str) -> SessionView:
        run = self._get(run_id)
        for owned in run.sessions:
            if owned.connection.session_id == session_id:
                return owned.session.view()
        raise RunManagerError("SESSION_NOT_FOUND", "Session does not belong to this run.")

    def list_views(self) -> tuple[RunView, ...]:
        """Run summaries contain no seat-private card data."""
        return tuple(self.view(run_id) for run_id in self._runs)

    def config_label(self, run_id: str) -> str:
        return self._get(run_id).config.run.experiment_id or "Local run"

    def is_closed(self, run_id: str) -> bool:
        run = self._get(run_id)
        return (run.done.is_set() and run.recorder is not None
                and bool(getattr(run.recorder.health, "closed", False)))

    def archive_path(self, run_id: str) -> Path:
        self._get(run_id)
        return self._recording_root / run_id

    def query_events(self, run_id: str, session_id: str, after_seq: int = 0,
                     limit: int = 50) -> EventPage:
        run, scope = self._query_scope(run_id, session_id)
        return run.recorder.query(scope, after_seq=after_seq, limit=limit)

    def query_snapshot(self, run_id: str, session_id: str) -> tuple[tuple[LogEvent, ...], int, bool]:
        """Single-seat retained evidence; seat is resolved from owned connection."""
        run, scope = self._query_scope(run_id, session_id)
        retained = run.recorder.retained_from_seq
        events = tuple(event for event in run.recorder.cached_events
                       if event.session_id == scope.session_id and event.seat == scope.seat)
        return events, retained, self.view(run_id).log_complete

    def _query_scope(self, run_id: str, session_id: str) -> tuple[_Run, EventQueryScope]:
        run = self._get(run_id)
        for owned in run.sessions:
            if owned.connection.session_id == session_id:
                if run.recorder is None:
                    raise RunManagerError("HISTORY_UNAVAILABLE", "Recording is not available yet.")
                return run, EventQueryScope(run_id=run_id, session_id=session_id,
                                           seat=owned.connection.seat)
        raise RunManagerError("SESSION_NOT_FOUND", "Session does not belong to this run.")

    def _stop_sessions(self, run: _Run, reason: str) -> None:
        for owned in run.sessions:
            owned.session.stop(reason)

    def _record(self, run: _Run, event_type: str, payload: dict[str, Any]) -> None:
        if run.recorder is None:
            return
        try:
            result = run.recorder.append(EventDraft(
                schema_version="1.0", run_id=run.run_id,
                match_id=run.active_match_id, session_id=None, seat=None,
                round_id=None, request_id=None, event_type=event_type, payload=payload,
            ))
            if not result.accepted:
                run.log_complete = False
        except Exception:
            # A log failure does not fabricate or erase the game result.
            run.log_complete = False

    async def _bounded(self, awaitable: Awaitable[Any], timeout: float) -> Any:
        task = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait((task,), timeout=timeout)
            if not done:
                raise TimeoutError
            return task.result()
        finally:
            if not task.done():
                task.cancel()
                task.add_done_callback(_consume)

    def _session_done(self, run: _Run, owned: _OwnedSession,
                      task: asyncio.Task[None]) -> None:
        failed = task.cancelled() or task.exception() is not None
        failed = failed or owned.session.status != "completed" or owned.session.match_result is None
        if failed and not run.stop_requested and not run.done.is_set():
            run.error = _error("SESSION_INTERRUPTED", "A seat ended without a trusted match completion.")
            run.stop_requested = True
            run.status = "stopping"
            self._stop_sessions(run, "peer_interrupted")
            if run.task is not None:
                run.task.cancel()

    async def _new_session(self, run: _Run, match_id: str, seat: Any,
                           mode: str, room_id: int | None) -> _OwnedSession:
        if run.stop_requested:
            raise asyncio.CancelledError
        connection = SeatConnection(
            url=run.config.platform.url, mode=mode, room_id=room_id,
            user_name=seat.user_name, seat=seat.seat, session_id=str(uuid4()),
            connection_epoch=1, server_rounds=run.config.platform.server_rounds,
        )
        platform = self._platform_factory(connection)
        session = self._session_factory(
            run_id=run.run_id, match_id=match_id, connection=connection,
            config=run.config, platform=platform, recorder=run.recorder,
            clock=self._clock, scheduler=self._scheduler,
            tools=run.tools, policy=run.policy, call_budget=run.call_budget,
        )
        task = asyncio.create_task(session.run(), name=f"session-{connection.session_id}")
        owned = _OwnedSession(match_id, connection, session, task)
        run.sessions.append(owned)
        task.add_done_callback(lambda completed: self._session_done(run, owned, completed))
        await self._bounded(session.wait_joined(), self._startup_timeout_s)
        if run.stop_requested:
            raise asyncio.CancelledError
        return owned

    async def _drive(self, run: _Run) -> None:
        run.started = True
        try:
            if run.stop_requested:
                return
            await self._bounded(self._create_recorder(run), self._startup_timeout_s)
            if run.stop_requested:
                return
            metadata = await asyncio.to_thread(_source_metadata)
            if run.stop_requested:
                return
            if metadata["source_sha256"] is None:
                run.log_complete = False
            self._record(run, "run_started", {"config_id": run.config.config_id,
                         "config": run.config.model_dump(mode="json"), "environment": metadata})
            self._prepare_resources(run)
            for _ in range(run.config.run.matches):
                if run.stop_requested:
                    return
                match_id = str(uuid4())
                run.active_match_id = match_id
                run.status = "starting"
                self._record(run, "match_started", {})
                seats = run.config.run.seats
                first = await self._new_session(run, match_id, seats[0],
                                                run.config.platform.mode,
                                                run.config.platform.room_id)
                room_id = first.session.room_id
                if type(room_id) is not int or room_id < 0:
                    raise RunManagerError("JOIN_INVALID", "Admission did not return a valid room ID.")
                owned = [first]
                for seat in seats[1:]:
                    owned.append(await self._new_session(run, match_id, seat, "join", room_id))
                run.status = "running"
                await asyncio.gather(*(asyncio.shield(item.task) for item in owned))
                if run.stop_requested:
                    return
                if any(item.session.status != "completed" or item.session.match_result is None
                       for item in owned):
                    raise RunManagerError("MATCH_UNCONFIRMED", "Match completion is not confirmed.")
                results = [item.session.match_result for item in owned]
                if any(result != results[0] for result in results[1:]):
                    raise RunManagerError("MATCH_RESULT_CONFLICT", "Seats received inconsistent match results.")
                self._record(run, "match_completed", {"result": dict(results[0])})
                await self._close_sessions(owned)
            run.status = "completed"
        except asyncio.CancelledError:
            run.error = run.error or _error("RUN_STOPPED", "Run was stopped locally.")
        except TimeoutError:
            run.error = _error("STARTUP_TIMEOUT", "Run resource or seat admission timed out.")
        except RunManagerError as exc:
            run.error = exc.error
        except Exception:
            run.error = _error("RUN_START_OR_EXECUTION_FAILED", "Run resource or seat execution failed.")
        finally:
            if run.status != "completed":
                run.status = "interrupted"
                run.stop_requested = True
                self._stop_sessions(run, "run_interrupted")
            await self._close_sessions(run.sessions)
            await self._close_resources(run)
            self._record(run, "model_budget_summary", {
                         "slots_used": run.call_budget.slots_used if run.call_budget else 0,
                         "calls_started": run.call_budget.calls_started if run.call_budget else 0})
            self._record(run, "run_finished", {"status": run.status,
                         "error": None if run.error is None else run.error.model_dump(mode="json")})
            await self._close_recorder(run)
            run.active_match_id = None
            if self._active_run_id == run.run_id:
                self._active_run_id = None
            run.done.set()

    def _prepare_resources(self, run: _Run) -> None:
        from guandan_agent.strategy import DecisionPolicy
        run.call_budget = RunCallBudget(run.config.batch.max_model_calls)
        if run.config.versions.policy != CHEAP_RULE_VERSION:
            run.tools = self._tools_factory(run.config)
        gateway = None
        if any(seat.policy in {"V1", "V2"} for seat in run.config.run.seats):
            from guandan_agent.model.gateway import ModelGateway
            assert self._provider_factory is not None
            run.provider = self._provider_factory(run.config)
            gateway = ModelGateway(run.provider, clock=self._clock,
                                   concurrency=run.config.model.concurrency,
                                   timeout_s=run.config.model.timeout_s,
                                   max_output_tokens=run.config.model.max_output_tokens,
                                   max_input_bytes=run.config.model.max_input_bytes,
                                   prompt_version=run.config.versions.prompt)
            run.gateway = gateway
        run.policy = DecisionPolicy(gateway=gateway, review_config=run.config.review)

    async def _close_resources(self, run: _Run) -> None:
        for resource in (run.tools, run.gateway if run.gateway is not None else run.provider):
            close = getattr(resource, "aclose", None)
            if close is not None:
                try:
                    await self._bounded(close(), self._shutdown_timeout_s)
                except (Exception, asyncio.CancelledError):
                    run.log_complete = False

    async def _create_recorder(self, run: _Run) -> None:
        recorder = await self._recorder_factory(
            self._recording_root / run.run_id, run.run_id, run.config,
            clock=self._clock, startup_timeout_s=self._startup_timeout_s,
        )
        # Publish acquisition before the awaiter's next turn, so cancellation
        # at the creation boundary cannot strand an already acquired writer.
        run.recorder = recorder
        if run.stop_requested and run.done.is_set():
            await self._close_recorder(run)

    async def _close_sessions(self, sessions: list[_OwnedSession]) -> None:
        async def close(owned: _OwnedSession) -> None:
            if not owned.task.done():
                owned.task.cancel()
            try:
                await self._bounded(owned.session.shutdown(timeout_s=self._shutdown_timeout_s),
                                    self._shutdown_timeout_s)
            except (Exception, asyncio.CancelledError):
                pass
        if sessions:
            await asyncio.gather(*(close(owned) for owned in sessions))

    async def _close_recorder(self, run: _Run) -> None:
        if run.recorder is None:
            run.log_complete = False
            return
        unconfirmed = tuple(record.submission_id for owned in run.sessions
                            for record in owned.session.execution_records
                            if record.status in {"writing", "sent", "unknown"})
        try:
            summary = await self._bounded(run.recorder.close(
                run.run_id, timeout_s=self._shutdown_timeout_s,
                pending_call_ids=tuple(call_id for owned in run.sessions
                                       for call_id in getattr(owned.session, "pending_call_ids", ())),
                unconfirmed_submission_ids=unconfirmed,
            ), self._shutdown_timeout_s + 0.1)
            run.log_complete = run.log_complete and summary.closed and summary.log_complete
        except (Exception, asyncio.CancelledError):
            run.log_complete = False
