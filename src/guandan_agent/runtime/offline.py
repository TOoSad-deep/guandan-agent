"""Independent, bounded authority for explicitly requested offline reviews.

This owner never receives a Session, an online ticket or a send capability. Its
per-manager budget is permanent across source logs and tasks. A closed task's
report and usage remain frozen even if a cancelled provider later responds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
from typing import Protocol
from uuid import uuid4

from guandan_agent.clock import Clock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    ErrorInfo, OfflineCallRecord, OfflineTaskView, ProviderInput, ProviderMessage,
    ProviderReply, ReviewReport, Usage,
)
from guandan_agent.model.provider import ProviderPort, unknown_usage
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.budget import RunCallBudget


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="offline", safe_message=message,
                     related_event_id=None)


class OfflineReviewError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.error = _error(code, message)
        super().__init__(message)


class OfflineCallPort(Protocol):
    """The review generator can only make bounded offline provider attempts."""

    @property
    def task_id(self) -> str: ...

    async def call(self, messages: tuple[ProviderMessage, ...]) -> ProviderReply: ...


@dataclass
class _Task:
    task_id: str
    config: RunConfig
    source: ClosedRunLog
    deadline: float
    job: Callable[[OfflineCallPort], Awaitable[ReviewReport]]
    driver: asyncio.Task | None = None
    work: asyncio.Task | None = None
    provider: ProviderPort | None = None
    attempts: set[asyncio.Task] = field(default_factory=set)
    records: dict[str, OfflineCallRecord] = field(default_factory=dict)
    reserved: set[str] = field(default_factory=set)
    started_at: dict[str, float] = field(default_factory=dict)
    report: ReviewReport | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: ErrorInfo | None = None
    sequence: int = 0
    cleaning: bool = False


class _Port:
    def __init__(self, owner: OfflineReviews, task: _Task) -> None:
        self._owner, self._task = owner, task

    @property
    def task_id(self) -> str:
        return self._task.task_id

    async def call(self, messages: tuple[ProviderMessage, ...]) -> ProviderReply:
        return await self._owner._call(self._task, messages)


def _consume(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()


def _usage(records: tuple[OfflineCallRecord, ...]) -> Usage:
    """Unknown usage is not free; do not infer token totals from components."""
    started = [record.usage for record in records if record.calls_started]
    tokens = {
        name: (sum(values) if all(value is not None for value in values) else None)
        for name in ("input_tokens", "output_tokens", "total_tokens")
        for values in [[getattr(usage, name) for usage in started]]
    }
    # Currency and pricing bases must agree before costs can be summed.
    costs_known = bool(started) and all(usage.cost_amount is not None for usage in started)
    currencies = {usage.currency for usage in started}
    bases = {usage.cost_basis for usage in started}
    costs_known = costs_known and len(currencies) == len(bases) == 1 and None not in currencies and None not in bases
    return Usage(**tokens,
                 cost_amount=sum(usage.cost_amount for usage in started) if costs_known else None,
                 currency=next(iter(currencies)) if costs_known else None,
                 cost_basis=next(iter(bases)) if costs_known else None)


class OfflineReviews:
    """One offline batch within one manager; all mutations use its event loop."""

    def __init__(self, *, clock: Clock,
                 provider_factory: Callable[[RunConfig], ProviderPort] | None,
                 online_active: Callable[[], bool], shutdown_timeout_s: float) -> None:
        self._clock = clock
        self._provider_factory = provider_factory
        self._online_active = online_active
        self._shutdown_timeout_s = shutdown_timeout_s
        self._tasks: dict[str, _Task] = {}
        self._active: str | None = None
        self._signature: str | None = None
        self._budget: RunCallBudget | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._cleanup: set[asyncio.Task] = set()

    def start(self, config: RunConfig, source: ClosedRunLog,
              job: Callable[[OfflineCallPort], Awaitable[ReviewReport]]) -> OfflineTaskView:
        if self._online_active():
            raise OfflineReviewError("OFFLINE_ONLINE_ACTIVE", "Offline review cannot start during an online run.")
        if self._active is not None:
            raise OfflineReviewError("OFFLINE_CONFLICT", "Another offline review is active.")
        if not isinstance(config, RunConfig):
            raise OfflineReviewError("INVALID_CONFIG", "A validated RunConfig is required.")
        frozen = RunConfig.model_validate_json(config.model_dump_json())
        if not isinstance(source, ClosedRunLog) or not source.summary.closed:
            raise OfflineReviewError("OFFLINE_SOURCE_OPEN", "Offline review requires a verified closed source log.")
        if source.summary.run_id != source.run_id or source.summary.config_id != source.config_id:
            raise OfflineReviewError("OFFLINE_SOURCE_INVALID", "Source identity is inconsistent.")
        # A tail cache is diagnostic evidence, not a complete offline source.
        try:
            source.offline_events()
        except Exception:
            raise OfflineReviewError("OFFLINE_SOURCE_TRUNCATED", "Offline source evidence is not fully retained.") from None
        if not callable(job):
            raise OfflineReviewError("OFFLINE_JOB_INVALID", "An offline review generator is required.")
        signature = json.dumps({"model": frozen.model.model_dump(mode="json"),
                                "offline": frozen.offline.model_dump(mode="json"),
                                "prompt": frozen.versions.prompt}, sort_keys=True)
        if self._signature is not None and signature != self._signature:
            raise OfflineReviewError("OFFLINE_CONFIG_CONFLICT", "Offline model and budget configuration is already frozen.")
        if self._signature is None:
            self._signature = signature
            self._budget = RunCallBudget(frozen.offline.max_model_calls)
            self._semaphore = asyncio.Semaphore(frozen.model.concurrency)
        task_id = str(uuid4())
        task = _Task(task_id, frozen, source,
                     self._clock.monotonic() + frozen.model.timeout_s * max(1, frozen.offline.max_model_calls), job)
        self._tasks[task_id] = task
        self._active = task_id
        task.driver = asyncio.create_task(self._drive(task), name=f"offline-{task_id}")
        return self.view(task_id)

    def _get(self, task_id: str) -> _Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise OfflineReviewError("OFFLINE_NOT_FOUND", "Offline review was not found.") from None

    def view(self, task_id: str) -> OfflineTaskView:
        task = self._get(task_id)
        records = tuple(task.records.values())
        return OfflineTaskView(
            task_id=task_id, source_run_id=task.source.run_id,
            status="running" if task.report is None else task.report.status,
            slots_used=len(task.reserved), calls_started=sum(record.calls_started for record in records),
            calls=records, error=task.error,
            report_ready=task.report is not None,
        )

    async def wait(self, task_id: str) -> ReviewReport:
        task = self._get(task_id)
        await task.done.wait()
        assert task.report is not None
        return task.report

    def usage(self) -> dict[str, int]:
        return {"max_slots": 0 if self._budget is None else self._budget.max_slots,
                "slots_used": 0 if self._budget is None else self._budget.slots_used,
                "calls_started": 0 if self._budget is None else self._budget.calls_started}

    def stop(self, task_id: str, reason: str = "Offline review was stopped.") -> OfflineTaskView:
        task = self._get(task_id)
        if task.report is None:
            # Caller text is not a safe log payload; use one fixed diagnostic.
            self._finish_error(task, "OFFLINE_CANCELLED", "Offline review was stopped.", "partial")
            if task.work is not None:
                task.work.cancel()
            for attempt in task.attempts:
                attempt.cancel()
            if task.driver is not None:
                task.driver.cancel()
            self._schedule_cleanup(task)
        return self.view(task_id)

    def stop_active(self) -> None:
        if self._active is not None:
            self.stop(self._active)

    async def close(self) -> None:
        self.stop_active()
        # Drivers relinquish ownership immediately after cancellation; remote
        # cleanup is best-effort and cannot extend the bounded local close.
        drivers = {task.driver for task in self._tasks.values()
                   if task.driver is not None and not task.driver.done()}
        pending = drivers | self._cleanup
        if pending:
            await asyncio.wait(pending, timeout=self._shutdown_timeout_s)

    def _active_task(self, task: _Task) -> bool:
        return (task.report is None and self._active == task.task_id and not self._online_active()
                and self._clock.monotonic() < task.deadline)

    def _finish_error(self, task: _Task, code: str, message: str, status: str) -> None:
        if task.report is not None:
            return
        error = _error(code, message)
        for call_id, record in tuple(task.records.items()):
            if record.error is None and call_id in task.started_at:
                task.records[call_id] = record.model_copy(update={
                    "error": error,
                    "elapsed_ms": max(0.0, self._clock.monotonic() - task.started_at[call_id]) * 1000,
                })
        task.error = error
        self._finish(task, ReviewReport(
            report_id=task.task_id, run_id=task.source.run_id, config_id=task.source.config_id,
            status=status, created_at=self._clock.utc_now(), items=(),
            usage=_usage(tuple(task.records.values())), errors=(error,),
        ))

    def _finish(self, task: _Task, report: ReviewReport) -> None:
        if task.report is not None:
            return
        task.report = report.model_copy(update={"usage": _usage(tuple(task.records.values()))})
        if self._active == task.task_id:
            self._active = None
        task.done.set()

    async def _drive(self, task: _Task) -> None:
        try:
            if task.report is not None:
                return
            task.work = asyncio.create_task(task.job(_Port(self, task)), name=f"review-{task.task_id}")
            task.work.add_done_callback(_consume)
            done, _ = await asyncio.wait((task.work,), timeout=max(0.0, task.deadline - self._clock.monotonic()))
            if task.report is not None:
                return
            if not done:
                self._finish_error(task, "OFFLINE_TIMEOUT", "Offline review exceeded its time limit.", "partial")
                task.work.cancel()
                return
            report = task.work.result()
            if (not isinstance(report, ReviewReport) or report.report_id != task.task_id
                    or report.run_id != task.source.run_id or report.config_id != task.source.config_id):
                self._finish_error(task, "OFFLINE_REPORT_IDENTITY", "Generated report identity does not match its source.", "failed")
                return
            # Revalidate unsafe model_copy updates before allowing report export.
            report = ReviewReport.model_validate_json(report.model_dump_json())
            errors = list(report.errors)
            if not task.source.summary.log_complete:
                errors.append(_error("OFFLINE_SOURCE_INCOMPLETE", "Source log is closed but incomplete."))
            # A generator cannot turn a denied/failed call into a complete
            # report, or complete while a provider operation is still pending.
            known_codes = {error.code for error in errors}
            for record in task.records.values():
                if record.error is not None and record.error.code not in known_codes:
                    errors.append(record.error)
                    known_codes.add(record.error.code)
            if task.started_at:
                errors.append(_error("OFFLINE_CALL_PENDING", "Offline review ended with an unfinished call."))
            if errors:
                report = report.model_copy(update={
                    "status": "partial" if report.status == "completed" else report.status,
                    "errors": tuple(errors),
                })
            task.error = report.errors[0] if report.errors else None
            self._finish(task, report)
        except asyncio.CancelledError:
            self._finish_error(task, "OFFLINE_CANCELLED", "Offline review was cancelled.", "partial")
        except Exception:
            self._finish_error(task, "OFFLINE_FAILED", "Offline review could not be generated.", "failed")
        finally:
            if task.work is not None and not task.work.done():
                task.work.cancel()
            for attempt in task.attempts:
                attempt.cancel()
            self._schedule_cleanup(task)

    def _schedule_cleanup(self, task: _Task) -> None:
        if task.cleaning:
            return
        task.cleaning = True
        if task.provider is None:
            return
        async def cleanup() -> None:
            close_task = asyncio.create_task(task.provider.aclose())
            close_task.add_done_callback(_consume)
            done, _ = await asyncio.wait((close_task,), timeout=self._shutdown_timeout_s)
            if not done:
                close_task.cancel()
        close = asyncio.create_task(cleanup(), name=f"offline-close-{task.task_id}")
        self._cleanup.add(close)
        close.add_done_callback(self._cleanup.discard)
        close.add_done_callback(_consume)

    async def _call(self, task: _Task, messages: tuple[ProviderMessage, ...]) -> ProviderReply:
        task.sequence += 1
        call_id = f"{task.task_id}:o{task.sequence}"
        start = self._clock.monotonic()
        deadline = min(task.deadline, start + task.config.model.timeout_s)
        started = 0
        transport = 0
        usage = unknown_usage()
        acquired = False
        attempt: asyncio.Task | None = None

        def reply(error: ErrorInfo | None, content: str | None = None) -> ProviderReply:
            result = ProviderReply(call_id=call_id, content=content, usage=usage,
                                   transport_requests=transport,
                                   elapsed_ms=max(0.0, self._clock.monotonic() - start) * 1000,
                                   error=error)
            if task.report is None:
                task.records[call_id] = OfflineCallRecord(
                    call_id=call_id, calls_started=started,
                    transport_requests=transport, elapsed_ms=result.elapsed_ms,
                    usage=usage, error=error,
                )
                task.started_at.pop(call_id, None)
            return result

        try:
            if not self._active_task(task):
                return reply(_error("OFFLINE_CANCELLED", "Offline review no longer accepts calls."))
            task.started_at[call_id] = start
            task.records[call_id] = OfflineCallRecord(
                call_id=call_id, calls_started=0, transport_requests=0,
                elapsed_ms=0.0, usage=unknown_usage(), error=None)
            if (not isinstance(messages, tuple) or not messages
                    or not all(isinstance(message, ProviderMessage) for message in messages)):
                return reply(_error("OFFLINE_INPUT_INVALID", "Offline model messages are invalid."))
            size = len(json.dumps([message.model_dump(mode="json") for message in messages],
                                  ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if size > task.config.model.max_input_bytes:
                return reply(_error("INPUT_LIMIT", "Offline model input exceeds its byte limit."))
            assert self._budget is not None and self._semaphore is not None
            # Waiters do not reserve a slot. A transport which ignores cancel
            # retains its semaphore permit until it actually exits.
            try:
                async with asyncio.timeout(max(0.0, deadline - self._clock.monotonic())):
                    await self._semaphore.acquire()
                acquired = True
            except TimeoutError:
                return reply(_error("OFFLINE_TIMEOUT", "No offline model slot before the deadline."))
            if not self._active_task(task) or self._clock.monotonic() >= deadline:
                return reply(_error("OFFLINE_CANCELLED", "Offline review no longer accepts calls."))
            if not self._budget.reserve(call_id):
                return reply(_error("OFFLINE_BUDGET", "Offline model budget is exhausted."))
            task.reserved.add(call_id)
            task.records[call_id] = OfflineCallRecord(
                call_id=call_id, calls_started=0, transport_requests=0,
                elapsed_ms=0.0, usage=unknown_usage(), error=None)
            if task.provider is None:
                if self._provider_factory is None:
                    return reply(_error("POLICY_UNAVAILABLE", "Offline review requires an explicit provider."))
                try:
                    task.provider = self._provider_factory(task.config)
                except Exception:
                    return reply(_error("OFFLINE_PROVIDER_UNAVAILABLE", "Offline model provider could not be initialized."))
            provider_input = ProviderInput(call_id=call_id, messages=messages,
                                           deadline_mono=deadline,
                                           max_output_tokens=task.config.model.max_output_tokens)

            async def authorized_attempt() -> ProviderReply:
                nonlocal started, transport
                if not self._active_task(task) or self._clock.monotonic() >= deadline:
                    return reply(_error("OFFLINE_CANCELLED", "Offline model authorization was revoked."))
                if not self._budget.begin(call_id):
                    return reply(_error("OFFLINE_BUDGET", "Offline model authorization was already used."))
                started, transport = 1, None
                task.started_at[call_id] = start
                task.records[call_id] = OfflineCallRecord(
                    call_id=call_id, calls_started=1, transport_requests=None,
                    elapsed_ms=0.0, usage=unknown_usage(), error=None)
                return await task.provider.complete(provider_input)

            attempt = asyncio.create_task(authorized_attempt(), name=f"offline-call-{call_id}")
            task.attempts.add(attempt)
            def completed(result: asyncio.Task) -> None:
                task.attempts.discard(result)
                self._semaphore.release()
                # The local timeout result remains a failure. Only unknown
                # accounting fields can be filled while the task is still open.
                try:
                    late = result.result()
                    previous = task.records.get(call_id)
                    if (task.report is None and previous is not None and previous.error is not None
                            and isinstance(late, ProviderReply) and late.call_id == call_id):
                        merged = previous.usage.model_dump()
                        for name, value in late.usage.model_dump().items():
                            if merged[name] is None:
                                merged[name] = value
                        task.records[call_id] = previous.model_copy(update={
                            "usage": Usage(**merged),
                            "transport_requests": (late.transport_requests if previous.transport_requests is None
                                                   else previous.transport_requests),
                        })
                except (asyncio.CancelledError, Exception):
                    pass
            attempt.add_done_callback(completed)
            acquired = False
            done, _ = await asyncio.wait((attempt,), timeout=max(0.0, deadline - self._clock.monotonic()))
            if not done:
                attempt.cancel()
                return reply(_error("MODEL_TIMEOUT", "Offline model request exceeded its time limit."))
            response = attempt.result()
            if not isinstance(response, ProviderReply) or response.call_id != call_id:
                return reply(_error("OFFLINE_REPLY_IDENTITY", "Offline provider reply has an invalid identity."))
            usage, transport = response.usage, response.transport_requests
            if not self._active_task(task):
                return reply(_error("OFFLINE_CANCELLED", "Offline review no longer accepts calls."))
            if self._clock.monotonic() >= deadline:
                return reply(_error("MODEL_TIMEOUT", "Offline model reply arrived after its deadline."))
            return reply(response.error, response.content)
        except asyncio.CancelledError:
            if attempt is not None:
                attempt.cancel()
            reply(_error("OFFLINE_CANCELLED", "Offline model request was cancelled."))
            raise
        except Exception:
            return reply(_error("OFFLINE_PROVIDER_FAILED", "Offline model request failed."))
        finally:
            if acquired:
                self._semaphore.release()
