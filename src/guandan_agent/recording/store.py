"""One bounded writer per run, with explicit incomplete-record boundaries.

The writer thread owns disk I/O. Cancellation cannot interrupt an OS write: a
close timeout therefore returns closed=False, stops further work, and never
turns a late write into a completed log. Queries only read the bounded cache.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import threading
from typing import Callable, Protocol

from pydantic import TypeAdapter, ValidationError

from guandan_agent.clock import Clock, SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    AppendResult,
    ErrorInfo,
    EventDraft,
    EventPage,
    EventQueryScope,
    LogEvent,
    RunSummary,
    UUIDString,
)


def _error(code: str) -> ErrorInfo:
    messages = {
        "LOG_QUEUE_FULL": "The recording queue rejected an event.",
        "LOG_WRITE_FAILED": "The recording writer could not persist all records.",
        "LOG_CLOSE_TIMEOUT": "The recording writer did not finish before the close deadline.",
        "LOG_START_FAILED": "The run recording could not be created.",
        "LOG_START_TIMEOUT": "The recording writer did not start before the deadline.",
        "LOG_CLOSED": "The run recording is no longer accepting events.",
        "LOG_NOT_CLOSED": "This record has no verified completed close.",
        "LOG_INCOMPLETE": "The recorded files contain missing or invalid evidence.",
        "HISTORY_UNAVAILABLE": "The requested cursor precedes the retained event window.",
    }
    return ErrorInfo(
        code=code, component="recording", safe_message=messages[code], related_event_id=None,
    )


class RecordingClosedError(RuntimeError):
    def __init__(self, *, incomplete_source: bool = False) -> None:
        self.error = _error("LOG_NOT_CLOSED" if incomplete_source else "LOG_CLOSED")
        super().__init__(self.error.safe_message)


class HistoryUnavailable(LookupError):
    """A later HTTP adapter maps this explicit cache miss to status 410."""

    def __init__(self, retained_from_seq: int) -> None:
        self.error = _error("HISTORY_UNAVAILABLE")
        self.retained_from_seq = retained_from_seq
        super().__init__(self.error.safe_message)


@dataclass(frozen=True)
class RecordingHealth:
    closed: bool
    log_complete: bool
    last_persisted_seq: int
    error: ErrorInfo | None


class EventSink(Protocol):
    """Blocking sink used only by the single writer, injectable for fault tests."""

    def write_config(self, content: str) -> None: ...
    def write_event(self, content: str) -> None: ...
    def close_events(self) -> None: ...
    def write_summary(self, content: str) -> None: ...
    def publish_summary(self) -> None: ...


class FileEventSink:
    """Exclusive files prevent replacing a prior run, including incomplete runs."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        # No buffered bytes are implicitly retried by close after a failed
        # event write. A partial OS write is an evidence gap, never retried.
        self._events = (run_dir / "events.jsonl").open("xb", buffering=0)

    def write_config(self, content: str) -> None:
        with (self._run_dir / "config.json").open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")

    def write_event(self, content: str) -> None:
        data = (content + "\n").encode("utf-8")
        if self._events.write(data) != len(data):
            raise OSError("incomplete event write")

    def close_events(self) -> None:
        self._events.close()

    def write_summary(self, content: str) -> None:
        # A blocked/stopped write never exposes a completed summary. Only the
        # writer's final publication step creates summary.json.
        with (self._run_dir / "summary.pending").open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")

    def publish_summary(self) -> None:
        # Linking alone is not a trusted close: summary.pending remains the
        # unconfirmed-close marker until the waiting owner commits completion.
        os.link(self._run_dir / "summary.pending", self._run_dir / "summary.json")


def _seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be a finite positive number")
    return float(value)


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _page(
    *, events: tuple[LogEvent, ...], run_id: str, scope: EventQueryScope,
    after_seq: int, limit: int, max_query_limit: int, evicted_through: int,
    retained_from_seq: int, log_complete: bool,
) -> EventPage:
    if not isinstance(scope, EventQueryScope):
        raise TypeError("query requires an EventQueryScope")
    if scope.run_id != run_id:
        raise ValueError("query scope does not belong to this run")
    if type(after_seq) is not int or after_seq < 0:
        raise ValueError("after_seq must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= max_query_limit:
        raise ValueError("query limit exceeds its configured bounds")
    if after_seq < evicted_through:
        raise HistoryUnavailable(retained_from_seq)
    matches = [
        event for event in events
        if event.seq > after_seq and event.session_id == scope.session_id and event.seat == scope.seat
    ]
    items = tuple(matches[:limit])
    return EventPage(
        items=items, next_after_seq=items[-1].seq if items else None,
        has_more=len(matches) > limit, retained_from_seq=retained_from_seq,
        log_complete=log_complete,
    )


class EventStore:
    """Create asynchronously, append synchronously, then close exactly once.

    Public online reads require run/session/seat together. The merged JSONL is
    a local evidence file, not a public all-seat query endpoint.
    """

    def __init__(
        self, run_dir: Path, run_id: str, config: RunConfig, clock: Clock,
        sink_factory: Callable[[Path], EventSink],
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.config = config
        self._clock = clock
        self._sink_factory = sink_factory
        self._queue: queue.Queue[LogEvent] = queue.Queue(config.logs.queue_capacity)
        self._lock = threading.Lock()
        self._events: deque[LogEvent] = deque(maxlen=config.logs.cache_events)
        self._seq = 0
        self._evicted_through = 0
        self._last_persisted_seq = 0
        self._log_complete = True
        self._error: ErrorInfo | None = None
        self._accepting = False
        self._write_failed = False
        self._closed = False
        self._startup_error = False
        self._ready = threading.Event()
        self._done = threading.Event()
        self._closing = threading.Event()
        self._abort = threading.Event()
        self._gaps: list[tuple[int, int]] = []
        self._gap_ranges_omitted = 0
        self._pending_call_ids: tuple[str, ...] = ()
        self._unconfirmed_submission_ids: tuple[str, ...] = ()
        self._summary: RunSummary | None = None
        self._close_candidate: RunSummary | None = None
        self._close_task: asyncio.Task[RunSummary] | None = None
        self._thread = threading.Thread(target=self._writer, name=f"run-log-{run_id}", daemon=True)

    @classmethod
    async def create(
        cls, run_dir: str | Path, run_id: str, config: RunConfig, *,
        clock: Clock | None = None, sink_factory: Callable[[Path], EventSink] | None = None,
        startup_timeout_s: float = 5.0,
    ) -> EventStore:
        timeout = _seconds(startup_timeout_s)
        TypeAdapter(UUIDString).validate_python(run_id)
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a validated RunConfig")
        store = cls(Path(run_dir), run_id, config, clock or SystemClock(), sink_factory or FileEventSink)
        store._thread.start()
        try:
            ready = await asyncio.to_thread(store._ready.wait, timeout)
        except asyncio.CancelledError:
            store._abort.set()
            store._closing.set()
            raise
        if not ready:
            store._abort.set()
            store._closing.set()
            raise RuntimeError(_error("LOG_START_TIMEOUT").safe_message)
        if store._startup_error:
            raise RuntimeError(_error("LOG_START_FAILED").safe_message)
        with store._lock:
            store._accepting = True
        return store

    @property
    def health(self) -> RecordingHealth:
        with self._lock:
            return RecordingHealth(self._closed, self._log_complete, self._last_persisted_seq, self._error)

    @property
    def cached_events(self) -> tuple[LogEvent, ...]:
        """Local diagnostic snapshot; public adapters must enforce seat ownership."""
        with self._lock:
            return tuple(self._events)

    @property
    def retained_from_seq(self) -> int:
        with self._lock:
            return self._events[0].seq if self._events else 1

    def _new_event(self, draft: EventDraft) -> LogEvent:
        # The caller holds the short metadata lock; no sink I/O occurs here.
        self._seq += 1
        return LogEvent(
            **draft.model_dump(), seq=self._seq, event_id=f"{self.run_id}:e{self._seq}",
            recorded_at=self._clock.utc_now(),
        )

    def _cache(self, event: LogEvent) -> None:
        if len(self._events) == self._events.maxlen:
            self._evicted_through = self._events[0].seq
        self._events.append(event)

    def _fail(self, code: str) -> None:
        self._log_complete = False
        if self._error is None or code in {"LOG_CLOSE_TIMEOUT", "LOG_WRITE_FAILED"}:
            self._error = _error(code)

    def _gap(self, seq: int) -> None:
        if self._gaps and self._gaps[-1][1] + 1 == seq:
            self._gaps[-1] = self._gaps[-1][0], seq
        elif len(self._gaps) < 64:
            self._gaps.append((seq, seq))
        else:
            self._gap_ranges_omitted += 1

    def append(self, draft: EventDraft) -> AppendResult:
        if not isinstance(draft, EventDraft):
            raise TypeError("append requires a validated EventDraft")
        if draft.run_id != self.run_id:
            raise ValueError("draft does not belong to this run")
        with self._lock:
            if not self._accepting:
                raise RecordingClosedError()
            event = self._new_event(draft)
            if self._write_failed:
                self._gap(event.seq)
                return AppendResult(accepted=False, seq=event.seq, event_id=event.event_id, error=_error("LOG_WRITE_FAILED"))
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._fail("LOG_QUEUE_FULL")
                self._gap(event.seq)
                return AppendResult(accepted=False, seq=event.seq, event_id=event.event_id, error=_error("LOG_QUEUE_FULL"))
            self._cache(event)
            return AppendResult(accepted=True, seq=event.seq, event_id=event.event_id, error=None)

    def query(self, scope: EventQueryScope, after_seq: int = 0, limit: int = 50) -> EventPage:
        with self._lock:
            events = tuple(self._events)
            return _page(
                events=events, run_id=self.run_id, scope=scope, after_seq=after_seq, limit=limit,
                max_query_limit=self.config.logs.max_query_limit, evicted_through=self._evicted_through,
                retained_from_seq=events[0].seq if events else self._seq + 1,
                log_complete=self._log_complete,
            )

    def _internal_event(self, event_type: str, payload: dict) -> LogEvent:
        event = self._new_event(EventDraft(
            schema_version="1.0", run_id=self.run_id, match_id=None, session_id=None,
            seat=None, round_id=None, request_id=None, event_type=event_type, payload=payload,
        ))
        self._cache(event)
        return event

    def _persist(self, sink: EventSink, event: LogEvent) -> bool:
        if self._abort.is_set():
            return False
        try:
            sink.write_event(event.model_dump_json())
        except Exception:
            with self._lock:
                self._write_failed = True
                self._fail("LOG_WRITE_FAILED")
                self._gap(event.seq)
            return False
        if self._abort.is_set():
            return False
        with self._lock:
            self._last_persisted_seq = event.seq
        return True

    def _make_summary(self, *, closed: bool) -> RunSummary:
        return RunSummary(
            schema_version="1.0", run_id=self.run_id, config_id=self.config.config_id,
            closed=closed, log_complete=self._log_complete and closed,
            last_persisted_seq=self._last_persisted_seq,
            pending_call_ids=self._pending_call_ids,
            unconfirmed_submission_ids=self._unconfirmed_submission_ids, error=self._error,
        )

    def _writer(self) -> None:
        sink: EventSink | None = None
        events_closed = False
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            if self._abort.is_set():
                return
            sink = self._sink_factory(self.run_dir)
            if self._abort.is_set():
                return
            sink.write_config(json.dumps({
                "schema_version": "1.0", "run_id": self.run_id, "config_id": self.config.config_id,
                "config": self.config.model_dump(mode="json"),
            }, ensure_ascii=False, sort_keys=True, allow_nan=False))
            if self._abort.is_set():
                return
            (self.run_dir / "reports").mkdir()
            self._ready.set()
            while not self._abort.is_set():
                if self._closing.is_set() and self._queue.empty():
                    break
                try:
                    event = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                if self._write_failed:
                    with self._lock:
                        self._gap(event.seq)
                else:
                    self._persist(sink, event)
                self._queue.task_done()
            if self._abort.is_set():
                return
            # No accepted events remain. Missing ranges are recorded only if the
            # existing writer is usable; failed files are never reopened/retried.
            with self._lock:
                gap = self._internal_event("log_gap", {
                    "missing_seq_ranges": self._gaps,
                    "omitted_range_count": self._gap_ranges_omitted,
                    "reason": "queue_rejected_events",
                    "last_persisted_seq": self._last_persisted_seq,
                }) if self._gaps and not self._write_failed else None
            if gap is not None:
                self._persist(sink, gap)
            if self._abort.is_set():
                return
            with self._lock:
                closing_event = self._internal_event("run_closed", {
                    "log_complete": self._log_complete,
                    "pending_call_ids": self._pending_call_ids,
                    "unconfirmed_submission_ids": self._unconfirmed_submission_ids,
                }) if not self._write_failed else None
            if closing_event is not None:
                self._persist(sink, closing_event)
            if self._abort.is_set():
                return
            try:
                events_closed = True  # A failed close is not retried implicitly.
                sink.close_events()
            except Exception:
                with self._lock:
                    self._fail("LOG_WRITE_FAILED")
                return
            if self._abort.is_set():
                return
            with self._lock:
                summary = self._make_summary(closed=True)
            # Summary I/O is also part of closing: a timeout while writing it
            # must not leave a truthful-looking closed marker. The default sink
            # publishes summary only after completing a separate temporary file.
            sink.write_summary(summary.model_dump_json())
            if self._abort.is_set():
                return
            sink.publish_summary()
            if self._abort.is_set():
                return
            with self._lock:
                self._close_candidate = summary
        except Exception:
            with self._lock:
                self._write_failed = True
                self._fail("LOG_WRITE_FAILED")
                if not self._ready.is_set():
                    self._startup_error = True
        finally:
            if sink is not None and not events_closed:
                try:
                    sink.close_events()
                except Exception:
                    with self._lock:
                        self._fail("LOG_WRITE_FAILED")
            self._ready.set()
            self._done.set()

    async def close(
        self, run_id: str, *, timeout_s: float = 2.0,
        pending_call_ids: tuple[str, ...] = (), unconfirmed_submission_ids: tuple[str, ...] = (),
    ) -> RunSummary:
        if run_id != self.run_id:
            raise ValueError("close run_id does not belong to this run")
        timeout = _seconds(timeout_s)
        if self._close_task is None:
            # Validate caller metadata before stopping admission. Unknown calls
            # and submissions remain identifiers, never invented usage/results.
            validated = RunSummary(
                schema_version="1.0", run_id=self.run_id, config_id=self.config.config_id,
                closed=False, log_complete=False, last_persisted_seq=0,
                pending_call_ids=pending_call_ids,
                unconfirmed_submission_ids=unconfirmed_submission_ids, error=None,
            )
            with self._lock:
                self._accepting = False
                self._pending_call_ids = validated.pending_call_ids
                self._unconfirmed_submission_ids = validated.unconfirmed_submission_ids
            self._closing.set()
            self._close_task = asyncio.create_task(self._finish_close(timeout))
        return await asyncio.shield(self._close_task)

    async def _finish_close(self, timeout: float) -> RunSummary:
        done = await asyncio.to_thread(self._done.wait, timeout)
        if not done:
            self._abort.set()
            with self._lock:
                self._fail("LOG_CLOSE_TIMEOUT")
                while True:
                    try:
                        event = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._gap(event.seq)
                    self._queue.task_done()
        with self._lock:
            if done and not self._abort.is_set() and self._close_candidate is not None:
                # No await, injected sink operation, or later writer work can
                # occur during this short local close commit. A timed-out owner
                # never removes the marker, even if publication finishes later.
                try:
                    (self.run_dir / "summary.pending").unlink()
                except OSError:
                    self._fail("LOG_WRITE_FAILED")
                else:
                    self._closed = True
                    self._summary = self._close_candidate
            if self._summary is None:
                self._summary = self._make_summary(closed=False)
            return self._summary


class ClosedRunLog:
    """One-time historical load; diagnostic queries also support broken runs.

    offline_events() requires a verified close and the complete retained window.
    Consumers of incomplete, but closed, files must retain health.log_complete.
    No read opens models, writes files, or changes a recording.
    """

    def __init__(
        self, *, run_id: str, config_id: str, events: tuple[LogEvent, ...],
        summary: RunSummary, evicted_through: int, max_query_limit: int,
        config: RunConfig | None = None,
    ) -> None:
        self.run_id = run_id
        self.config_id = config_id
        self.summary = summary
        self._events = events
        self._evicted_through = evicted_through
        self._max_query_limit = max_query_limit
        self.config = config

    @property
    def cached_events(self) -> tuple[LogEvent, ...]:
        """Bounded diagnostic evidence, including incomplete/unclosed records."""
        return self._events

    @property
    def retained_from_seq(self) -> int:
        return self._events[0].seq if self._events else 1

    @property
    def health(self) -> RecordingHealth:
        return RecordingHealth(
            self.summary.closed, self.summary.log_complete,
            self.summary.last_persisted_seq, self.summary.error,
        )

    @classmethod
    async def load(
        cls, run_dir: str | Path, *, cache_events: int = 2000, max_query_limit: int = 200,
    ) -> ClosedRunLog:
        _positive_int(cache_events, "cache_events")
        if not 1 <= _positive_int(max_query_limit, "max_query_limit") <= 200:
            raise ValueError("max_query_limit cannot exceed 200")
        return await asyncio.to_thread(cls._load, Path(run_dir), cache_events, max_query_limit)

    @classmethod
    def _load(cls, run_dir: Path, cache_events: int, max_query_limit: int) -> ClosedRunLog:
        try:
            wrapper = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            run_id = TypeAdapter(UUIDString).validate_python(wrapper["run_id"])
            config = RunConfig.model_validate(wrapper["config"])
            if wrapper["schema_version"] != "1.0" or wrapper["config_id"] != config.config_id:
                raise ValueError("invalid config identity")
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("The recorded configuration is missing or invalid.") from None
        events: deque[LogEvent] = deque(maxlen=cache_events)
        evicted_through = 0
        last_seq = 0
        complete = True
        run_closed_seen = False
        try:
            with (run_dir / "events.jsonl").open("rb") as stream:
                for line in stream:
                    try:
                        event = LogEvent.model_validate_json(line)
                        if not line.endswith(b"\n") or event.run_id != run_id or event.seq <= last_seq:
                            raise ValueError("invalid event position")
                    except (ValueError, ValidationError):
                        complete = False
                        continue
                    if event.seq != last_seq + 1:
                        complete = False
                    last_seq = event.seq
                    if event.event_type == "log_gap":
                        complete = False
                    run_closed_seen = event.event_type == "run_closed"
                    if len(events) == events.maxlen:
                        evicted_through = events[0].seq
                    events.append(event)
        except OSError:
            complete = False
        try:
            summary = RunSummary.model_validate_json((run_dir / "summary.json").read_bytes())
            if summary.run_id != run_id or summary.config_id != config.config_id:
                raise ValueError("invalid summary identity")
            if summary.last_persisted_seq != last_seq:
                complete = False
        except (OSError, ValueError, ValidationError):
            summary = RunSummary(
                schema_version="1.0", run_id=run_id, config_id=config.config_id,
                closed=False, log_complete=False, last_persisted_seq=last_seq,
                pending_call_ids=(), unconfirmed_submission_ids=(), error=_error("LOG_INCOMPLETE"),
            )
            complete = False
        # A writer may finish its uninterruptible publication after its owner
        # timed out. The owner alone removes this marker after observing done;
        # merely finding summary.json cannot establish a completed close.
        verified_closed = summary.closed and not (run_dir / "summary.pending").exists()
        complete = complete and summary.log_complete and verified_closed and run_closed_seen
        # Reconstruct an immutable diagnostic projection, never rewrite source files.
        summary = RunSummary(
            **{**summary.model_dump(), "closed": verified_closed, "log_complete": complete,
               "error": summary.error if complete else summary.error or _error("LOG_INCOMPLETE")},
        )
        return cls(
            run_id=run_id, config_id=config.config_id, events=tuple(events), summary=summary,
            evicted_through=evicted_through, max_query_limit=max_query_limit,
            config=config,
        )

    def query(self, scope: EventQueryScope, after_seq: int = 0, limit: int = 50) -> EventPage:
        return _page(
            events=self._events, run_id=self.run_id, scope=scope, after_seq=after_seq, limit=limit,
            max_query_limit=self._max_query_limit, evicted_through=self._evicted_through,
            retained_from_seq=self._events[0].seq if self._events else 1,
            log_complete=self.summary.log_complete,
        )

    def offline_events(self) -> tuple[LogEvent, ...]:
        if not self.summary.closed:
            raise RecordingClosedError(incomplete_source=True)
        if self._evicted_through:
            raise HistoryUnavailable(self._events[0].seq)
        return self._events
