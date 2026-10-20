"""One seat's synchronous authority around asynchronous policy and transport work.

There is deliberately no business inbox: ``post`` applies an event before it
returns. Only immutable request snapshots leave this object. A proposal never
owns permission to write; the platform rechecks that permission immediately
before its single socket write.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Coroutine, Mapping
from typing import Any

from guandan_agent.clock import Clock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    BudgetView, CallFinished, CallTicket, CandidateSnapshot, MainReady, ModelCallResult, ModelDecision,
    ConnectionLost, ConfirmationDeadline, DecisionInput, DecisionOutcome,
    DecisionProposal, ErrorInfo, EventDraft, ExecutionRecord, GateDecision,
    LegalActionSet, PlatformEvent, ProposalReady, RequestBinding, RequestContext,
    ReviewResult, SeatConnection, SendFinished, SendResult, SessionView, Stop,
    SubmissionDeadline, SubmissionIntent, SuggestionDeadline, TributeContext,
    ToolLimits, ToolResult, ToolsReady,
)
from guandan_agent.hand import cheap_choice
from guandan_agent.platform.adapter import PlatformPort
from guandan_agent.runtime.scheduling import Scheduler
from guandan_agent.runtime.budget import RunCallBudget
from guandan_agent.runtime.state import StateConflict, TableState
from guandan_agent.strategy import DecisionPolicy


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="runtime", safe_message=message, related_event_id=None)


class _CallPort:
    """Capability wrapper: policies receive no state or transport interface."""

    __slots__ = ("__owner",)

    def __init__(self, owner: Session) -> None:
        self.__owner = owner

    def status(self, binding: RequestBinding) -> BudgetView:
        return self.__owner._call_status(binding)

    def admit_call(self, binding: RequestBinding, role: str) -> CallTicket | ErrorInfo:
        return self.__owner._admit_call(binding, role)

    def begin_call(self, ticket: CallTicket) -> GateDecision:
        return self.__owner._begin_call(ticket)

    def record_result(self, result: ModelCallResult) -> None:
        self.__owner._record_result(result)

    def register_main(self, main: MainReady) -> bool:
        return self.__owner._register_main(main)


class Session:
    def __init__(self, *, run_id: str, match_id: str, connection: SeatConnection,
                 config: RunConfig, platform: PlatformPort, recorder: Any,
                 clock: Clock, scheduler: Scheduler, policy: Any = None,
                 tools: Any = None, call_budget: RunCallBudget | None = None) -> None:
        self.run_id = run_id
        self.match_id = match_id
        self.connection = connection
        self.config = config
        self.platform = platform
        self.recorder = recorder
        self.clock = clock
        self._start_mono = clock.monotonic()
        self.scheduler = scheduler
        self.policy = policy if policy is not None else DecisionPolicy()
        self.tools = tools
        self._call_budget = call_budget if call_budget is not None else RunCallBudget(config.batch.max_model_calls)
        self._call_port = _CallPort(self)
        self._tickets: dict[str, CallTicket] = {}
        self._request_calls: dict[str, list[str]] = {}
        self._request_limits: dict[str, int] = {}
        self._started_calls: set[str] = set()
        self._call_results: dict[str, ModelCallResult] = {}
        self._candidate_snapshot: CandidateSnapshot | None = None
        self._registered_main: MainReady | None = None
        self._second_call_started = False
        self.joined = asyncio.Event()
        self.finished = asyncio.Event()
        self._room_id: int | None = None
        self._status = "starting"
        self._error: ErrorInfo | None = None
        self._table = TableState(connection.seat, history_limit=config.history.max_events)
        self._state_version = 0
        self._view_version = 0
        self._updated_at = clock.utc_now()
        self._rx_seq = 0
        self._request_seq = 0
        self._fact_generation = 0
        self._request_generation: int | None = None
        self._active: RequestContext | None = None
        self._active_submission: str | None = None
        self._decision_stage: str | None = None
        self._final_deadline: float | None = None
        self._suggest_deadline: float | None = None
        self._records: dict[str, ExecutionRecord] = {}
        self._intents: dict[str, SubmissionIntent] = {}
        self._write_rx: dict[str, int] = {}
        self._write_mono: dict[str, float] = {}
        self._send_finished_ids: set[str] = set()
        self._handles: list[Any] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._policy_task: asyncio.Task[Any] | None = None
        self._run_task: asyncio.Task[Any] | None = None
        self._shutdown_task: asyncio.Task[Any] | None = None
        self._started = False
        self._closed = False
        self._log_complete = True
        self._projected_log_complete = True
        self._missing_event_seq = 0
        self._match_result: Mapping[str, Any] | None = None

    @property
    def room_id(self) -> int | None:
        return self._room_id

    @property
    def status(self) -> str:
        return self._status

    @property
    def active_context(self) -> RequestContext | None:
        return self._active

    @property
    def call_port(self) -> _CallPort:
        return self._call_port

    @property
    def candidate_snapshot(self) -> CandidateSnapshot | None:
        return self._candidate_snapshot

    @property
    def registered_main(self) -> MainReady | None:
        return self._registered_main

    @property
    def pending_call_ids(self) -> tuple[str, ...]:
        return tuple(call_id for call_id in self._tickets
                     if call_id in self._started_calls and call_id not in self._call_results)

    @property
    def call_results(self) -> tuple[ModelCallResult, ...]:
        return tuple(self._call_results.values())

    @property
    def execution_records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._records.values())

    @property
    def match_result(self) -> Mapping[str, Any] | None:
        return self._match_result

    def _touch(self) -> None:
        self._view_version += 1
        self._updated_at = self.clock.utc_now()

    def _log(self, event_type: str, payload: Mapping[str, Any],
             binding: RequestBinding | None = None) -> str:
        try:
            result = self.recorder.append(EventDraft(
                schema_version="1.0", run_id=self.run_id, match_id=self.match_id,
                session_id=self.connection.session_id, seat=self.connection.seat,
                round_id=binding.round_id if binding else (self._table.round_id or None),
                request_id=binding.request_id if binding else None,
                event_type=event_type, payload=payload,
            ))
            if not result.accepted:
                self._log_complete = False
            return result.event_id
        except Exception:
            # Recording is evidence, never authority to block legal play.
            self._log_complete = False
            self._missing_event_seq += 1
            return f"{self.connection.session_id}:unrecorded:{self._missing_event_seq}"

    def view(self) -> SessionView:
        health = getattr(self.recorder, "health", None)
        complete = self._log_complete and (health is None or health.log_complete)
        if complete != self._projected_log_complete:
            self._projected_log_complete = complete
            self._touch()  # Evidence health changes the UI, never decision authority.
        return SessionView(
            run_id=self.run_id, match_id=self.match_id, round_id=self._table.round_id,
            session_id=self.connection.session_id, seat=self.connection.seat,
            state_version=self._state_version, view_version=self._view_version,
            updated_at=self._updated_at, status=self._status, state=self._table.table_view(),
            active_request_id=self._active.state.binding.request_id if self._active else None,
            decision_stage=self._decision_stage,
            latest_execution=next(reversed(self._records.values()), None),
            log_complete=complete, error=self._error,
        )

    async def wait_joined(self) -> int:
        if not self.joined.is_set() and not self.finished.is_set():
            waits = {asyncio.create_task(self.joined.wait()), asyncio.create_task(self.finished.wait())}
            try:
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in waits:
                    task.cancel()
                await asyncio.gather(*waits, return_exceptions=True)
        if self._room_id is None or self._status not in ("running", "completed"):
            raise RuntimeError("Session did not join a running room")
        return self._room_id

    def _spawn(self, operation: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(operation)
        self._tasks.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            self._tasks.discard(done)
            if not done.cancelled():
                # Retrieve exceptions even for cancellation-resistant late work.
                done.exception()

        task.add_done_callback(completed)
        return task

    async def run(self) -> None:
        if self._started:
            raise RuntimeError("Session cannot be run twice")
        self._started = True
        self._run_task = asyncio.current_task()
        try:
            if self.finished.is_set():
                return
            await self.platform.connect(self.connection)
            if self.finished.is_set():
                return
            async for event in self.platform.events():
                self.post(event)
                if self.finished.is_set():
                    break
            if not self.finished.is_set():
                self.post(ConnectionLost(run_id=self.run_id, session_id=self.connection.session_id,
                                         reason="Platform stream ended before match completion"))
        except asyncio.CancelledError:
            if not self.finished.is_set():
                self.stop("Session run cancelled")
        except Exception:
            if not self.finished.is_set():
                self.post(ConnectionLost(run_id=self.run_id, session_id=self.connection.session_id,
                                         reason="Platform connection failed"))
        finally:
            await self.shutdown()

    def stop(self, reason: str = "Run stopped") -> None:
        self.post(Stop(run_id=self.run_id, session_id=self.connection.session_id, reason=reason))

    def post(self, event: Any) -> None:
        """Apply one event now; this method never awaits policy, network or disk."""
        if self._closed:
            return
        if isinstance(event, PlatformEvent):
            self._platform_event(event)
        elif isinstance(event, (Stop, ConnectionLost)):
            if (event.run_id, event.session_id) == (self.run_id, self.connection.session_id):
                code = "STOPPED" if isinstance(event, Stop) else "CONNECTION_LOST"
                self._interrupt(_error(code, event.reason))
        elif isinstance(event, ProposalReady):
            if self._current(event.binding):
                self._proposal(event.outcome)
        elif isinstance(event, ToolsReady):
            if self._current(event.binding):
                self._tools_ready(event)
        elif isinstance(event, MainReady):
            self._register_main(event)
        elif isinstance(event, CallFinished):
            if event.binding == event.result.binding:
                self._record_result(event.result)
        elif isinstance(event, SuggestionDeadline):
            if self._current(event.binding) and self._active_submission is None:
                self._suggestion_expired()
        elif isinstance(event, SubmissionDeadline):
            if self._current(event.binding):
                self._invalidate("FINAL_DEADLINE")
        elif isinstance(event, SendFinished):
            self._send_finished(event)
        elif isinstance(event, ConfirmationDeadline):
            record = self._records.get(event.submission_id)
            if record is not None and record.binding == event.binding and record.status in ("writing", "sent"):
                self._update(record, status="unknown", error=_error("CONFIRMATION_TIMEOUT", "No reliable acceptance feedback before the confirmation deadline"))
        else:
            raise TypeError("Unsupported Session event")

    def _current(self, binding: RequestBinding) -> bool:
        return (not self._closed and self._status == "running" and self._active is not None
                and self._active.state.binding == binding)

    def _invalidate(self, cause: str) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        self._candidate_snapshot = None
        self._registered_main = None
        self._second_call_started = False
        self._decision_stage = "invalidated"
        if self._policy_task is not None and not self._policy_task.done():
            self._policy_task.cancel()
        if self._active_submission is not None:
            record = self._records[self._active_submission]
            if record.status == "reserved":
                self._update(record, status="cancelled_before_write", error=_error(cause, "Request lost permission before writing"))
        self._log("request_invalidated", {"cause_code": cause}, active.state.binding)
        self._active_submission = None
        self._touch()

    def _interrupt(self, error: ErrorInfo) -> None:
        if self._status == "completed":
            return
        if self._status != "interrupted":
            self._error = error
            self._status = "interrupted"
            self._invalidate(error.code)
            for record in tuple(self._records.values()):
                if record.status in ("writing", "sent"):
                    self._update(record, status="unknown", error=error)
            self._log("session_interrupted", {"error": error.model_dump(mode="json")})
            self._touch()
        self.finished.set()
        for task in tuple(self._tasks):
            task.cancel()
        if self._run_task is not None and self._run_task is not asyncio.current_task():
            self._run_task.cancel()

    def _platform_event(self, event: PlatformEvent) -> None:
        if (event.session_id, event.connection_epoch) != (self.connection.session_id, self.connection.connection_epoch):
            return
        if event.rx_seq <= self._rx_seq:
            return  # A locally redelivered envelope, never content deduplication.
        if event.rx_seq != self._rx_seq + 1:
            self._table.mark_degraded()
            self._interrupt(_error("PLATFORM_SEQUENCE_GAP", "Platform receive sequence is not continuous"))
            return
        self._rx_seq = event.rx_seq
        persisted = event.model_dump(mode="json")
        persisted.pop("received_mono")
        persisted["received_relative_ms"] = (event.received_mono - self._start_mono) * 1000.0
        event_id = self._log("platform_received", {"event": persisted})
        if event.kind == "joined":
            if self._status != "starting" or self._room_id is not None:
                self._interrupt(_error("INVALID_PLATFORM_ORDER", "Repeated admission notification"))
                return
            self._room_id = event.payload["data"]["roomId"]
            self._status = "running"
            self._log("session_started", {
                "config_id": self.config.config_id,
                "room_id": self._room_id, "seat": self.connection.seat,
                "connection_epoch": self.connection.connection_epoch,
                "rules_version": self.config.platform.rules_version,
                "adapter_version": self.config.platform.adapter_version,
                "seed": None,
            })
            self.joined.set()
            self._touch()
            return
        if event.kind in ("unknown", "join_failed"):
            self._table.mark_degraded()
            self._interrupt(_error("INVALID_PLATFORM_EVENT", "Platform event is invalid or admission was rejected"))
            return
        if event.kind == "auxiliary_notice":
            return
        if self._room_id is None:
            self._interrupt(_error("INVALID_PLATFORM_ORDER", "Gameplay arrived before admission"))
            return
        if event.kind == "match_result":
            if self._status != "running" or self._table.stage != "settled" or self._match_result is not None:
                self._interrupt(_error("STATE_CONFLICT", "Match settlement does not follow a settled round"))
                return
            self._invalidate("MATCH_SETTLED")
            self._match_result = event.payload
            self._log("match_result", {"result": event.payload, "evidence_event_id": event_id})
            self._touch()
            return
        if event.kind == "match_closed":
            self._invalidate("MATCH_CLOSED")
            if self._status == "running" and self._match_result is not None:
                self._status = "completed"
                self._log("session_completed", {"match_result": self._match_result})
                self._touch()
                self.finished.set()
            else:
                self._interrupt(_error("MISSING_MATCH_RESULT", "Match closed without trusted settlement"))
            return
        if event.kind == "action_requested" and self._status != "running":
            return
        if event.kind == "action_requested" and self._request_generation == self._fact_generation:
            self._table.mark_degraded()
            self._interrupt(_error("STATE_CONFLICT", "Repeated action request without a public fact transition"))
            return
        try:
            self._table.apply(event, event_id)
        except (StateConflict, ValueError):
            self._table.mark_degraded()
            self._interrupt(_error("STATE_CONFLICT", "Platform facts conflict with the visible table state"))
            return
        if event.kind == "round_result":
            self._log("round_result", {"result": event.payload, "evidence_event_id": event_id})
        self._confirm(event, event_id)
        self._invalidate("PLATFORM_ADVANCED")
        self._state_version += 1
        self._touch()
        if event.kind == "action_requested":
            self._request_generation = self._fact_generation
            self._request(event)
        else:
            self._fact_generation += 1

    def _request(self, event: PlatformEvent) -> None:
        if self._status != "running" or self._table.facts_status != "trusted":
            return
        # A new turn cannot retroactively confirm an older write. In
        # particular, PASS does not change the hand and could otherwise look
        # superficially consistent despite a missing acceptance broadcast.
        if any(record.binding.round_id == self._table.round_id
               and record.status in ("writing", "sent", "unknown")
               for record in self._records.values()):
            self._interrupt(_error("UNCONFIRMED_PREVIOUS_SUBMISSION",
                                   "A new request arrived before the prior write was confirmed"))
            return
        self._request_seq += 1
        binding = RequestBinding(
            run_id=self.run_id, match_id=self.match_id, round_id=self._table.round_id,
            session_id=self.connection.session_id, seat=self.connection.seat,
            connection_epoch=self.connection.connection_epoch,
            request_id=f"{self.connection.session_id}:q{self._request_seq}",
            state_version=self._state_version,
        )
        try:
            state = self._table.visible(binding)
            legal = LegalActionSet(binding=binding, **event.model_dump(mode="json")["payload"]["parsed_actions"])
            fallback_id = cheap_choice(state, legal)
            platform_context = event.payload["platform_context"]
            policy = next((seat.policy for seat in self.config.run.seats if seat.seat == self.connection.seat), self.config.run.policy)
            context = RequestContext(
                state=state, legal=legal, received_at=event.received_at,
                fallback_action_id=fallback_id, policy=policy,
                platform_context=TributeContext(**platform_context) if platform_context else None,
            )
        except Exception:
            self._table.mark_degraded()
            self._interrupt(_error("INVALID_REQUEST", "Cannot build a trusted request and fallback"))
            return
        self._active = context
        self._request_limits[binding.request_id] = {"V0": 0, "V1": 1, "V2": 2}[policy]
        self._active_submission = None
        self._candidate_snapshot = None
        self._registered_main = None
        self._second_call_started = False
        self._final_deadline = event.received_mono + self.config.decision.total_s
        self._suggest_deadline = self._final_deadline - self.config.decision.submit_reserve_s
        self._decision_stage = "deciding"
        # Log configured durations directly; subtracting absolute float clock
        # values can introduce rounding noise into otherwise identical budgets.
        self._log("request_opened", {"context": context.model_dump(mode="json"),
                                    "suggestion_budget_ms": (self.config.decision.total_s - self.config.decision.submit_reserve_s) * 1000.0,
                                    "final_budget_ms": self.config.decision.total_s * 1000.0,
                                    "budget_source": "local_experiment"}, binding)
        self._handles.append(self.scheduler.call_at(self._suggest_deadline,
            lambda: self.post(SuggestionDeadline(binding=binding))))
        self._handles.append(self.scheduler.call_at(self._final_deadline,
            lambda: self.post(SubmissionDeadline(binding=binding))))
        if self.clock.monotonic() >= self._final_deadline:
            self._invalidate("FINAL_DEADLINE")
        elif self.clock.monotonic() >= self._suggest_deadline:
            self._suggestion_expired()
        else:
            if self.tools is not None and len(context.legal.actions) > 1:
                self._decision_stage = "preparing"
                self._policy_task = self._spawn(self._prepare(context))
            else:
                self._policy_task = self._spawn(self._decide(context))
        self._touch()

    async def _prepare(self, context: RequestContext) -> None:
        assert self._suggest_deadline is not None
        started = self.clock.monotonic()
        limits = ToolLimits(
            deadline_mono=min(self._suggest_deadline, started + self.config.tools.total_s),
            deep_eval_limit=self.config.tools.deep_eval_limit,
            candidate_target=self.config.tools.candidate_target,
            candidate_hard_limit=self.config.tools.candidate_hard_limit,
        )
        try:
            result = await self.tools.prepare(context, limits)
            if not isinstance(result, ToolResult):
                raise TypeError("Invalid tools result")
        except asyncio.CancelledError:
            return
        except Exception:
            self.post(ProposalReady(binding=context.state.binding, outcome=DecisionOutcome(
                status="unavailable", proposal=None,
                error=_error("TOOL_INVALID_RESULT", "Candidate preparation failed"),
            )))
            return
        self.post(ToolsReady(binding=context.state.binding, result=result))

    def _tools_ready(self, event: ToolsReady) -> None:
        assert self._active is not None
        if self._active_submission is not None or self._candidate_snapshot is not None:
            return
        if not self._call_status(event.binding).suggestions_open:
            self._suggestion_expired()
            return
        result = event.result
        snapshot = result.snapshot
        if snapshot is None:
            self._log("tools_failed", {"error": result.error.model_dump(mode="json") if result.error else None}, event.binding)
            self._fallback(result.error.code if result.error else "TOOL_INVALID_RESULT")
            return
        legal = {action.action_id: action for action in self._active.legal.actions}
        ids = {candidate.action_id for candidate in snapshot.candidates}
        valid = (snapshot.binding == event.binding
                 and snapshot.candidate_snapshot_id == f"{event.binding.request_id}:c1"
                 and bool(ids) and len(ids) == len(snapshot.candidates)
                 and len(ids) <= self.config.tools.candidate_hard_limit
                 and ids <= legal.keys() and snapshot.rule_action_id in legal
                 and ((snapshot.tool_status == "ok" and not snapshot.degradation_codes)
                      or (snapshot.tool_status == "degraded" and bool(snapshot.degradation_codes))))
        if valid:
            # Mandatory legal choices are inexpensive to verify independently
            # of the ranking tool. Reordering or shortening candidates cannot
            # remove PASS or a way to finish the hand.
            hand = Counter(self._active.state.self_hand)
            required = {action.action_id for action in legal.values()
                        if action.wire_action[0] == "PASS" or (
                            self._active.state.stage == "play"
                            and Counter(action.wire_action[2]) == hand)}
            valid = required <= ids
            state = self._active.state
            leader = state.leader_seat
            if (state.stage == "play" and leader is not None
                    and leader % 2 != self.connection.seat % 2
                    and state.remaining_counts[leader] in (1, 2)):
                pressure = {key for key, action in legal.items() if action.wire_action[0] != "PASS"}
                valid = valid and (not pressure or bool(ids & pressure))
            for candidate in snapshot.candidates:
                features = candidate.features
                wire = legal[candidate.action_id].wire_action
                if state.stage != "play":
                    valid = valid and all(value is None for value in (
                        features.played_count, features.remaining_count, features.estimated_turns,
                        features.single_count, features.split_cost)) and not any((
                            features.is_bomb, features.finishes_hand, features.beats_leader))
                else:
                    played = 0 if wire[0] == "PASS" else len(wire[2])
                    remaining = len(state.self_hand) - played
                    valid = valid and (features.played_count == played
                        and features.remaining_count == remaining
                        and features.finishes_hand == (remaining == 0)
                        and features.is_bomb == (wire[0] in {"Bomb", "StraightFlush", "FourKings"})
                        and features.beats_leader == (wire[0] != "PASS" and leader is not None))
        if not valid:
            self._fallback("TOOL_INVALID_RESULT")
            return
        self._candidate_snapshot = snapshot
        self._decision_stage = "deciding"
        self._log("candidates_prepared", {
            "snapshot": snapshot.model_dump(mode="json"),
            "error": result.error.model_dump(mode="json") if result.error else None,
            "elapsed_ms": result.elapsed_ms,
        }, event.binding)
        self._policy_task = self._spawn(self._decide(self._active, snapshot))
        self._touch()

    def _call_status(self, binding: RequestBinding) -> BudgetView:
        active = self._current(binding)
        remaining = max(0.0, (self._suggest_deadline or 0.0) - self.clock.monotonic()) if active else 0.0
        return BudgetView(active=active,
                          suggestions_open=active and remaining > 0 and self._active_submission is None,
                          remaining_s=remaining,
                          slots_used=len(self._request_calls.get(binding.request_id, ())),
                          max_slots=self._request_limits.get(binding.request_id, 0))

    def _role_allowed(self, binding: RequestBinding, role: str, ordinal: int) -> bool:
        if ordinal == 1:
            return role == "main"
        if ordinal != 2 or role not in ("review", "repair"):
            return False
        first_id = self._request_calls.get(binding.request_id, [None])[0]
        first = self._call_results.get(first_id)
        if first is None or first.role != "main" or first.calls_started != 1:
            return False
        if role == "review":
            return self._registered_main is not None and self._registered_main.call_id == first_id
        return (first.decision is None and first.error is not None
                and first.error.code in {"MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN"})

    def _admit_call(self, binding: RequestBinding, role: str) -> CallTicket | ErrorInfo:
        view = self._call_status(binding)
        if not view.active:
            return _error("REQUEST_STALE", "The model request is no longer current")
        if not view.suggestions_open:
            return _error("DEADLINE", "The request no longer accepts suggestions")
        if self._candidate_snapshot is None or self._candidate_snapshot.binding != binding:
            return _error("TOOL_INVALID_RESULT", "No registered candidates for this request")
        if view.slots_used >= view.max_slots:
            return _error("CALL_BUDGET", "The request's model-call budget is exhausted")
        ordinal = view.slots_used + 1
        if not self._role_allowed(binding, role, ordinal):
            return _error("CALL_ORDER", "The requested model role does not follow the current call state")
        if role in ("review", "repair") and view.remaining_s < self.config.review.min_remaining_s:
            return _error("NO_REVIEW_TIME", "Insufficient remaining time for a second call")
        call_id = f"{binding.request_id}:m{ordinal}"
        if not self._call_budget.reserve(call_id):
            return _error("BATCH_BUDGET", "The run's model-call budget is exhausted")
        assert self._suggest_deadline is not None
        ticket = CallTicket(call_id=call_id, binding=binding, role=role, ordinal=ordinal,
                            deadline_mono=min(self._suggest_deadline, self.clock.monotonic() + self.config.model.timeout_s))
        self._tickets[call_id] = ticket
        self._request_calls.setdefault(binding.request_id, []).append(call_id)
        self._log("model_call_admitted", {
            "call_id": call_id, "role": role, "ordinal": ordinal,
            "deadline_remaining_ms": max(0.0, ticket.deadline_mono - self.clock.monotonic()) * 1000.0,
            "slots_used": ordinal, "run_slots_used": self._call_budget.slots_used,
        }, binding)
        return ticket

    def _begin_call(self, ticket: CallTicket) -> GateDecision:
        view = self._call_status(ticket.binding)
        valid = (self._tickets.get(ticket.call_id) == ticket
                 and ticket.call_id not in self._started_calls
                 and ticket.call_id not in self._call_results
                 and view.suggestions_open
                 and self.clock.monotonic() < ticket.deadline_mono
                 and self._role_allowed(ticket.binding, ticket.role, ticket.ordinal))
        if ticket.role in ("review", "repair"):
            valid = valid and view.remaining_s >= self.config.review.min_remaining_s
        if not valid or not self._call_budget.begin(ticket.call_id):
            return GateDecision(status="deny", error=_error("CALL_DENIED", "The model call lost its start permission"))
        self._started_calls.add(ticket.call_id)
        if ticket.role in ("review", "repair"):
            self._second_call_started = True
            self._decision_stage = ticket.role
            self._touch()
        self._log("model_call_started", {
            "call_id": ticket.call_id, "role": ticket.role, "ordinal": ticket.ordinal,
            "calls_started": 1, "run_calls_started": self._call_budget.calls_started,
        }, ticket.binding)
        return GateDecision(status="allow", error=None)

    def _record_result(self, result: ModelCallResult) -> None:
        if self._closed or result.call_id is None:
            return
        ticket = self._tickets.get(result.call_id)
        if (ticket is None or ticket.binding != result.binding or ticket.role != result.role
                or result.candidate_snapshot_id != f"{ticket.binding.request_id}:c1"
                or result.calls_started != int(result.call_id in self._started_calls)
                or (result.calls_started == 0 and result.decision is not None)):
            return
        previous = self._call_results.get(result.call_id)
        if previous is not None:
            # A cancellation-resistant provider may reveal usage after a
            # timeout was already recorded. Only fill previously unknown
            # telemetry; never turn that timed-out decision into success.
            updates = {key: value for key, value in result.usage.model_dump().items()
                       if value is not None and getattr(previous.usage, key) is None}
            transport = previous.transport_requests
            if transport is None:
                transport = result.transport_requests
            if not updates and transport == previous.transport_requests:
                return
            usage = previous.usage.model_copy(update=updates)
            updated = previous.model_copy(update={"usage": usage, "transport_requests": transport})
            self._call_results[result.call_id] = updated
            self._log("model_call_usage_updated", {
                "call_id": result.call_id, "usage": usage.model_dump(mode="json"),
                "transport_requests": transport,
            }, result.binding)
            return
        self._call_results[result.call_id] = result
        self._log("model_call_finished", {"result": result.model_dump(mode="json"),
                                           "late": not self._call_status(result.binding).suggestions_open}, result.binding)

    def _register_main(self, main: MainReady) -> bool:
        view = self._call_status(main.binding)
        snapshot = self._candidate_snapshot
        result = self._call_results.get(main.call_id)
        if (not view.suggestions_open or self._second_call_started
                or snapshot is None or main.binding != snapshot.binding
                or main.candidate_snapshot_id != snapshot.candidate_snapshot_id
                or len(main.trigger_codes) != len(set(main.trigger_codes))
                or main.decision.action_id not in {item.action_id for item in snapshot.candidates}
                or result is None or result.binding != main.binding or result.role != "main"
                or result.calls_started != 1 or result.error is not None
                or result.decision != main.decision):
            return False
        if self._registered_main is not None:
            return self._registered_main == main
        self._registered_main = main
        self._log("main_registered", {"main": main.model_dump(mode="json")}, main.binding)
        return True

    def _suggestion_expired(self) -> None:
        if self._policy_task is not None and self._policy_task is not asyncio.current_task() and not self._policy_task.done():
            self._policy_task.cancel()
        main = self._registered_main
        if main is not None and self._current(main.binding) and not self._second_call_started:
            self._reserve(DecisionProposal(
                binding=main.binding, candidate_snapshot_id=main.candidate_snapshot_id,
                action_id=main.decision.action_id, source="main", reason=main.decision.reason,
                cause_code="SUGGESTION_DEADLINE",
                review=self._review_resolution("SUGGESTION_DEADLINE"),
            ))
        else:
            self._fallback("SUGGESTION_DEADLINE")

    async def _decide(self, context: RequestContext, candidates: CandidateSnapshot | None = None) -> None:
        if not self._current(context.state.binding):
            return
        try:
            input = DecisionInput(context=context, candidates=candidates)
            if self.tools is None:
                outcome = await self.policy.decide(input)
            else:
                outcome = await self.policy.decide(input, calls=self._call_port)
        except asyncio.CancelledError:
            return
        except Exception:
            outcome = DecisionOutcome(status="unavailable", proposal=None,
                                      error=_error("POLICY_FAILED", "Rule policy failed"))
        self.post(ProposalReady(binding=context.state.binding, outcome=outcome))

    def _review_state(self) -> tuple[str, ModelDecision | None, tuple[str, ...], CallTicket | None] | None:
        """Derive whether a second step exists from this request's authority."""
        context = self._active
        if context is None or context.policy != "V2":
            return None
        call_ids = self._request_calls.get(context.state.binding.request_id, ())
        second = self._tickets[call_ids[1]] if len(call_ids) == 2 else None
        main = self._registered_main
        if main is not None and (main.trigger_codes or (second is not None and second.role == "review")):
            return "review", main.decision, main.trigger_codes, second
        first = self._call_results.get(call_ids[0]) if call_ids else None
        if (main is None and first is not None and first.role == "main"
                and first.calls_started == 1 and first.error is not None
                and first.error.code in {"MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN"}):
            return "repair", None, (first.error.code,), second
        return None

    def _review_resolution(self, cause: str) -> ReviewResult | None:
        """Build a truthful terminal record when the policy cannot finish."""
        state = self._review_state()
        if state is None:
            return None
        kind, before, triggers, second = state
        started = second is not None and second.call_id in self._started_calls
        result = self._call_results.get(second.call_id) if second is not None else None
        return ReviewResult(
            kind=kind, status="failed" if started else "skipped", trigger_codes=triggers,
            before=before, after=None, call_id=None if second is None else second.call_id,
            error=(result.error if result is not None and result.error is not None
                   else _error(cause, "The second decision step did not produce an accepted final proposal")),
        )

    def _review_matches(self, proposal: DecisionProposal) -> bool:
        review = proposal.review
        state = self._review_state()
        if review is None:
            return state is None
        if state is None or self._candidate_snapshot is None:
            return False
        kind, before, triggers, second = state
        if (proposal.candidate_snapshot_id != self._candidate_snapshot.candidate_snapshot_id
                or review.kind != kind or review.before != before or review.trigger_codes != triggers
                or review.call_id != (None if second is None else second.call_id)):
            return False
        started = second is not None and second.call_id in self._started_calls
        result = self._call_results.get(second.call_id) if second is not None else None
        if result is not None and (result.binding != proposal.binding or result.role != kind
                or result.candidate_snapshot_id != proposal.candidate_snapshot_id):
            return False
        if review.status == "skipped":
            expected_source = "main" if kind == "review" else "fallback"
            return (not started and proposal.source == expected_source and review.after is None
                    and review.error is not None
                    and (result is None or result.error == review.error))
        if review.status == "failed":
            return (started and proposal.source == "fallback" and review.after is None
                    and result is not None and result.error is not None
                    and result.error == review.error)
        if (not started or result is None or result.error is not None or result.decision is None
                or review.after != result.decision or review.error is not None
                or proposal.action_id != result.decision.action_id
                or proposal.reason != result.decision.reason):
            return False
        expected_status = ("repaired" if kind == "repair" else
                           "kept" if before.action_id == result.decision.action_id else "changed")
        return proposal.source == kind and review.status == expected_status

    def _proposal(self, outcome: DecisionOutcome) -> None:
        assert self._active is not None
        if self._active_submission is not None:
            return
        if self._suggest_deadline is not None and self.clock.monotonic() >= self._suggest_deadline:
            self._suggestion_expired()
            return
        proposal = outcome.proposal
        if (proposal is None or proposal.binding != self._active.state.binding
                or proposal.action_id not in {a.action_id for a in self._active.legal.actions}):
            self._fallback(outcome.error.code if outcome.error is not None else "INVALID_PROPOSAL")
            return
        if not self._review_matches(proposal):
            self._fallback("INVALID_PROPOSAL")
            return
        if proposal.source in ("main", "review", "repair"):
            snapshot = self._candidate_snapshot
            main = self._registered_main
            valid = (snapshot is not None and proposal.candidate_snapshot_id == snapshot.candidate_snapshot_id
                     and proposal.action_id in {candidate.action_id for candidate in snapshot.candidates})
            if proposal.source == "main":
                valid = (valid and not self._second_call_started and main is not None
                         and proposal.action_id == main.decision.action_id
                         and proposal.reason == main.decision.reason)
            else:
                valid = valid and proposal.review is not None and proposal.review.status in ("kept", "changed", "repaired")
            if not valid:
                self._fallback("INVALID_PROPOSAL")
                return
        elif proposal.candidate_snapshot_id is not None and (
                self._candidate_snapshot is None
                or proposal.candidate_snapshot_id != self._candidate_snapshot.candidate_snapshot_id):
            self._fallback("INVALID_PROPOSAL")
            return
        if proposal.source == "fallback" and proposal.action_id != self._active.fallback_action_id:
            self._fallback("INVALID_PROPOSAL")
            return
        self._reserve(proposal)

    def _fallback(self, cause: str) -> None:
        context = self._active
        if context is None or self._active_submission is not None:
            return
        review = self._review_resolution(cause)
        self._reserve(DecisionProposal(
            binding=context.state.binding,
            candidate_snapshot_id=(self._candidate_snapshot.candidate_snapshot_id
                                   if review is not None and self._candidate_snapshot is not None else None),
            action_id=context.fallback_action_id, source="fallback",
            reason="Using the request's prevalidated rule fallback", cause_code=cause, review=review,
        ))

    def _reserve(self, proposal: DecisionProposal) -> None:
        context = self._active
        if context is None or self._active_submission is not None or not self._current(proposal.binding):
            return
        assert self._final_deadline is not None
        if self.clock.monotonic() >= self._final_deadline:
            self._invalidate("FINAL_DEADLINE")
            return
        action = next((a for a in context.legal.actions if a.action_id == proposal.action_id), None)
        if action is None:
            return
        submission_id = f"{proposal.binding.request_id}:s1"
        intent = SubmissionIntent(
            submission_id=submission_id, binding=proposal.binding,
            action_id=proposal.action_id, wire_action=action.wire_action,
            stage=context.state.stage, platform_context=context.platform_context,
            final_deadline_mono=self._final_deadline,
        )
        record = ExecutionRecord(
            submission_id=submission_id, binding=proposal.binding,
            action_id=proposal.action_id, wire_action=action.wire_action, source=proposal.source,
            status="reserved", reserved_at=self.clock.utc_now(), write_started_at=None,
            sent_at=None, confirmed_at=None, feedback_event_id=None, error=None,
        )
        self._active_submission = submission_id
        self._intents[submission_id] = intent
        self._records[submission_id] = record
        self._decision_stage = "submitting"
        if proposal.review is not None:
            self._log("review_finished", {
                "binding": proposal.binding.model_dump(mode="json"),
                "candidate_snapshot_id": proposal.candidate_snapshot_id,
                "review": proposal.review.model_dump(mode="json"),
            }, proposal.binding)
        self._log("proposal_ready", {"proposal": proposal.model_dump(mode="json")}, proposal.binding)
        self._log("submission_reserved", {"execution": record.model_dump(mode="json")}, proposal.binding)
        self._touch()
        self._spawn(self._write(intent))

    def begin_write(self, intent: SubmissionIntent) -> GateDecision:
        """The sole write permission transition, adjacent to the actual send."""
        record = self._records.get(intent.submission_id)
        valid = (self._current(intent.binding) and record is not None
                 and record.status == "reserved" and self._active_submission == intent.submission_id
                 and self._intents.get(intent.submission_id) == intent
                 and self._room_id is not None and self._table.stage == intent.stage)
        if not valid or self.clock.monotonic() >= intent.final_deadline_mono:
            error = _error("SEND_DENIED", "Submission is no longer current or its deadline elapsed")
            if record is not None and record.status == "reserved":
                self._update(record, status="cancelled_before_write", error=error)
            return GateDecision(status="deny", error=error)
        assert record is not None
        self._write_rx[intent.submission_id] = self._rx_seq
        self._write_mono[intent.submission_id] = self.clock.monotonic()
        self._update(record, event_type="write_started", status="writing", write_started_at=self.clock.utc_now())
        self._handles.append(self.scheduler.call_at(
            self.clock.monotonic() + self.config.decision.confirm_timeout_s,
            lambda: self.post(ConfirmationDeadline(binding=intent.binding, submission_id=intent.submission_id)),
        ))
        return GateDecision(status="allow", error=None)

    async def _write(self, intent: SubmissionIntent) -> None:
        try:
            result = await self.platform.write(intent, self)
        except asyncio.CancelledError:
            record = self._records.get(intent.submission_id)
            started = record is not None and record.write_started_at is not None
            result = SendResult(submission_id=intent.submission_id,
                                status="unknown" if started else "not_started",
                                error=_error("SEND_CANCELLED", "Submission transport was cancelled"))
        except Exception:
            record = self._records.get(intent.submission_id)
            started = record is not None and record.write_started_at is not None
            result = SendResult(submission_id=intent.submission_id,
                                status="unknown" if started else "not_started",
                                error=_error("SEND_FAILED", "Submission transport failed"))
        self.post(SendFinished(binding=intent.binding, result=result))

    def _send_finished(self, event: SendFinished) -> None:
        record = self._records.get(event.result.submission_id)
        if (record is None or record.binding != event.binding
                or event.result.submission_id in self._send_finished_ids):
            return
        self._send_finished_ids.add(event.result.submission_id)
        result = event.result
        if result.status == "sent" and record.write_started_at is not None:
            if record.sent_at is not None:
                return
            # Confirmation may race ahead of send's completion; never regress it.
            status = "sent" if record.status == "writing" else record.status
            self._update(record, status=status, sent_at=self.clock.utc_now())
        elif result.status == "unknown" and record.status in ("writing", "sent"):
            self._update(record, status="unknown", error=result.error)
        elif result.status == "not_started" and record.status == "reserved":
            self._update(record, status="cancelled_before_write", error=result.error)
        self._log("send_finished", {
            "execution": self._records[record.submission_id].model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        }, record.binding)
        if result.status == "unknown":
            # The production adapter permanently seals its send channel after
            # an uncertain transport write. A late acceptance may confirm this
            # action, but cannot restore that channel's permission to continue.
            # Keep confirmed evidence intact while terminating the connection.
            self._interrupt(_error("SEND_UNCERTAIN", "Transport write is uncertain; this connection cannot continue"))
        elif (result.status == "not_started" and self._current(event.binding)
              and result.error is not None and result.error.code in {
                  "PLATFORM_NOT_READY", "CONNECTION_BINDING_MISMATCH", "SEND_FAILED",
              }):
            self._interrupt(_error("PLATFORM_WRITE_UNAVAILABLE", "The current platform connection cannot submit"))

    def _update(self, record: ExecutionRecord, *, event_type: str = "execution_updated", **changes: Any) -> None:
        updated = ExecutionRecord(**{**record.model_dump(), **changes})
        self._records[record.submission_id] = updated
        self._log(event_type, {"execution": updated.model_dump(mode="json")}, updated.binding)
        self._touch()

    @staticmethod
    def _same_action(left: Any, right: Any) -> bool:
        if left[0] == "PASS" or right[0] == "PASS":
            return tuple(left) == tuple(right)
        return left[:2] == right[:2] and Counter(left[2]) == Counter(right[2])

    def _confirm(self, event: PlatformEvent, event_id: str) -> None:
        expected_stage = {"play_observed": "play", "tribute_announced": "tribute",
                          "exchange_completed": "back"}.get(event.kind)
        if expected_stage is None:
            return
        candidates = []
        for record in self._records.values():
            intent = self._intents[record.submission_id]
            if (record.status not in ("writing", "sent", "unknown")
                    or record.binding.round_id != self._table.round_id
                    or intent.stage != expected_stage
                    or event.rx_seq <= self._write_rx.get(record.submission_id, event.rx_seq)
                    or event.received_mono < self._write_mono.get(record.submission_id, float("inf"))):
                continue
            if expected_stage == "play":
                matched = (event.payload["curPos"] == self.connection.seat
                           and self._same_action(record.wire_action, event.payload["curAction"]))
            elif expected_stage == "tribute":
                matched = any(row[0] == self.connection.seat and row[2] == record.wire_action[2][0]
                              for row in event.payload["result"])
            else:
                context = intent.platform_context
                assert context is not None
                paired = (context.tribute_pos, self.connection.seat, context.tribute_card) in self._table.tribute_transfers
                matched = paired and any(
                    row[0] == self.connection.seat and row[1] == context.tribute_pos
                    and row[2] == record.wire_action[2][0] for row in event.payload["result"])
            if matched:
                candidates.append(record)
        if len(candidates) == 1:
            self._update(candidates[0], status="confirmed", confirmed_at=event.received_at,
                         feedback_event_id=event_id, error=None)

    async def shutdown(self, timeout_s: float = 1.0) -> None:
        """Close locally within a finite window, even if a task ignores cancel."""
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown(timeout_s))
        # asyncio.wait never waits for a cancellation-resistant coroutine to end.
        await asyncio.wait({self._shutdown_task}, timeout=timeout_s + 0.05)

    async def _shutdown(self, timeout_s: float) -> None:
        if self._closed:
            return
        if not self.finished.is_set():
            self._interrupt(_error("SESSION_CLOSED", "Session closed before trusted match completion"))
        for record in tuple(self._records.values()):
            if record.status == "reserved":
                self._update(record, status="cancelled_before_write", error=_error("SESSION_CLOSED", "Session closed before write"))
            elif record.status in ("writing", "sent"):
                self._update(record, status="unknown", error=_error("SESSION_CLOSED", "Session closed without acceptance feedback"))
        self._closed = True
        for handle in self._handles:
            handle.cancel()
        for task in tuple(self._tasks):
            task.cancel()
        close_task = asyncio.create_task(self.platform.close())
        pending = set(self._tasks) | {close_task}
        _, remaining = await asyncio.wait(pending, timeout=timeout_s)
        for task in remaining:
            task.cancel()
        def consume(done: asyncio.Task[Any]) -> None:
            if not done.cancelled():
                done.exception()
        close_task.add_done_callback(consume)
