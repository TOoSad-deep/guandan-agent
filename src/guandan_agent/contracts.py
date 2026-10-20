"""Strict, immutable contracts for the B01/B02 rule-only foundation.

These DTOs check shape and local consistency. Platform legality, current request
authority, deadlines and state transitions remain the responsibility of their
owning modules. Model/candidate/report contracts are added with their batches.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta
import math
import re
from types import MappingProxyType
from typing import Annotated, Any, Literal, TypeVar
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)


class ContractModel(BaseModel):
    """Shared base; nested containers must also use frozen field types."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _array(value: Any) -> tuple[Any, ...]:
    # JSON arrays and Python tuples have the same public meaning. Do not accept
    # arbitrary iterables (strings, sets, generators) or coerce their members.
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected an array")
    return tuple(value)


T = TypeVar("T")
FrozenArray = Annotated[tuple[T, ...], BeforeValidator(_array)]
Seat = Annotated[int, Field(ge=0, le=3)]
Count = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonemptyStr = Annotated[str, StringConstraints(min_length=1)]
Card = Annotated[str, StringConstraints(pattern=r"^(?:[SHCD][2-9TJQKA]|SB|HR)$")]
Stage = Literal["play", "tribute", "back"]
Policy = Literal["V0", "V1", "V2"]
RuleSource = Literal["unique", "rule", "main", "review", "repair", "fallback"]


def _uuid_string(value: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError("expected a UUID string") from exc
    if str(parsed) != value.lower():
        raise ValueError("expected a hyphenated UUID string")
    return value


UUIDString = Annotated[str, AfterValidator(_uuid_string)]


def _utc_string(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("expected a UTC ISO 8601 timestamp") from exc
    if "T" not in value or parsed.utcoffset() != timedelta(0):
        raise ValueError("expected a UTC ISO 8601 timestamp")
    return value


UTCTimestamp = Annotated[str, AfterValidator(_utc_string)]


def _wire_action(value: Any) -> tuple[Any, ...]:
    parts = _array(value)
    if len(parts) != 3:
        raise ValueError("wire action must have exactly three entries")
    if parts[0] == "PASS" or parts[1] == "PASS" or parts[2] == "PASS":
        if parts != ("PASS", "PASS", "PASS"):
            raise ValueError('PASS must be exactly ["PASS", "PASS", "PASS"]')
        return parts
    cards = _array(parts[2])
    if not cards:
        raise ValueError("non-PASS action must contain cards")
    return parts[0], parts[1], cards


WireAction = Annotated[
    tuple[NonemptyStr, NonemptyStr, tuple[Card, ...]]
    | tuple[Literal["PASS"], Literal["PASS"], Literal["PASS"]],
    BeforeValidator(_wire_action),
]


def _freeze_json(value: Any) -> Any:
    """Copy only JSON values; keep dictionaries and arrays deeply read-only."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError("payload must contain only finite JSON values")


def _freeze_object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("payload must be a JSON object")
    return _freeze_json(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


FrozenJSONObject = Annotated[
    Mapping[str, Any],
    PlainValidator(_freeze_object),
    PlainSerializer(_json_value, return_type=dict[str, Any]),
    WithJsonSchema({"type": "object"}),
]


class RequestBinding(ContractModel):
    run_id: UUIDString
    match_id: UUIDString
    round_id: PositiveInt
    session_id: UUIDString
    seat: Seat
    connection_epoch: PositiveInt
    request_id: NonemptyStr
    state_version: Count

    @model_validator(mode="after")
    def request_belongs_to_session(self) -> RequestBinding:
        suffix = self.request_id.removeprefix(f"{self.session_id}:q")
        if not self.request_id.startswith(f"{self.session_id}:q") or not re.fullmatch(
            r"[1-9][0-9]*", suffix
        ):
            raise ValueError("request_id must be <session_id>:q<positive integer>")
        return self


class PublicPlay(ContractModel):
    event_id: NonemptyStr
    seat: Seat
    action: WireAction


class VisibleState(ContractModel):
    binding: RequestBinding
    stage: Stage
    self_hand: FrozenArray[Card]
    level_rank: NonemptyStr
    remaining_counts: Annotated[FrozenArray[Count | None], Field(min_length=4, max_length=4)]
    team_ranks: Annotated[FrozenArray[NonemptyStr | None], Field(min_length=2, max_length=2)]
    turn_seat: Seat
    leader_seat: Seat | None
    leading_action: WireAction | None
    finished_order: FrozenArray[Seat]
    history: FrozenArray[PublicPlay]
    history_complete: bool
    history_omitted: Count
    facts_status: Literal["trusted", "degraded"]

    @model_validator(mode="after")
    def consistent_public_facts(self) -> VisibleState:
        if (self.leader_seat is None) != (self.leading_action is None):
            raise ValueError("leader_seat and leading_action must be null together")
        if self.leading_action == ("PASS", "PASS", "PASS"):
            raise ValueError("a normalized leading action cannot be PASS")
        if self.turn_seat != self.binding.seat:
            raise ValueError("a request state must belong to the acting seat")
        if len(set(self.finished_order)) != len(self.finished_order):
            raise ValueError("finished_order cannot repeat a seat")
        if self.history_complete and self.history_omitted:
            raise ValueError("complete history cannot have omitted entries")
        return self


class LegalAction(ContractModel):
    action_id: NonemptyStr
    source_indices: Annotated[FrozenArray[Count], Field(min_length=1)]
    wire_action: WireAction

    @model_validator(mode="after")
    def consistent_origin(self) -> LegalAction:
        if tuple(sorted(set(self.source_indices))) != self.source_indices:
            raise ValueError("source_indices must be unique and increasing")
        if self.action_id != f"L{self.source_indices[0]:04d}":
            raise ValueError("action_id must use the first original index")
        return self


class ParsedActions(ContractModel):
    """Validated complete action list before a Session assigns its binding."""

    raw_count: PositiveInt
    actions: Annotated[FrozenArray[LegalAction], Field(min_length=1)]

    @model_validator(mode="after")
    def complete_exact_deduplication(self) -> ParsedActions:
        indices = [index for action in self.actions for index in action.source_indices]
        if (
            len(indices) != self.raw_count
            or len(set(indices)) != self.raw_count
            or min(indices) != 0
            or max(indices) != self.raw_count - 1
        ):
            raise ValueError("source_indices must cover each raw action exactly once")
        first_indices = [action.source_indices[0] for action in self.actions]
        if first_indices != sorted(first_indices):
            raise ValueError("actions must preserve their first original occurrence order")
        if len({action.wire_action for action in self.actions}) != len(self.actions):
            raise ValueError("exact duplicate wire actions must share one LegalAction")
        return self


class LegalActionSet(ParsedActions):
    binding: RequestBinding


class TributeContext(ContractModel):
    tribute_pos: Seat
    tribute_card: Card


def _check_stage_context(stage: Stage, context: TributeContext | None) -> None:
    if (stage == "back") != (context is not None):
        raise ValueError("platform_context is required only for back")


def _check_stage_action(stage: Stage, action: WireAction) -> None:
    if stage in ("tribute", "back"):
        if action[0] != stage or action[1] != stage or len(action[2]) != 1:
            raise ValueError("tribute/back action must contain its stage and one card")
    elif action[0] in ("tribute", "back"):
        raise ValueError("a play request cannot use a tribute/back action")


class RequestContext(ContractModel):
    state: VisibleState
    legal: LegalActionSet
    received_at: UTCTimestamp
    fallback_action_id: NonemptyStr
    policy: Policy
    platform_context: TributeContext | None

    @model_validator(mode="after")
    def same_request_and_owned_cards(self) -> RequestContext:
        if self.state.binding != self.legal.binding:
            raise ValueError("state and legal actions must have identical binding")
        if self.fallback_action_id not in {action.action_id for action in self.legal.actions}:
            raise ValueError("fallback_action_id must belong to the complete legal set")
        _check_stage_context(self.state.stage, self.platform_context)
        hand = Counter(self.state.self_hand)
        for action in self.legal.actions:
            _check_stage_action(self.state.stage, action.wire_action)
            if action.wire_action[0] != "PASS" and Counter(action.wire_action[2]) - hand:
                raise ValueError("legal action consumes cards not present in the own hand")
        return self


class ErrorInfo(ContractModel):
    """Only allowlisted, caller-authored messages; never raw provider exceptions."""

    code: NonemptyStr
    component: NonemptyStr
    safe_message: str
    related_event_id: NonemptyStr | None


class ActionFeatures(ContractModel):
    played_count: Count | None
    remaining_count: Count | None
    estimated_turns: Count | None
    single_count: Count | None
    split_cost: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    is_bomb: bool
    finishes_hand: bool
    beats_leader: bool
    rule_score: Annotated[float, Field(allow_inf_nan=False)]


class Candidate(ContractModel):
    action_id: NonemptyStr
    features: ActionFeatures
    keep_reasons: FrozenArray[NonemptyStr]


class CandidateSnapshot(ContractModel):
    binding: RequestBinding
    candidate_snapshot_id: NonemptyStr
    candidates: Annotated[FrozenArray[Candidate], Field(min_length=1)]
    rule_action_id: NonemptyStr
    tool_status: Literal["ok", "degraded"]
    degradation_codes: FrozenArray[NonemptyStr]

    @model_validator(mode="after")
    def snapshot_identity(self) -> CandidateSnapshot:
        ids = [c.action_id for c in self.candidates]
        if self.candidate_snapshot_id != f"{self.binding.request_id}:c1":
            raise ValueError("snapshot must belong to its request")
        if len(ids) != len(set(ids)) or self.rule_action_id not in ids:
            raise ValueError("candidates must be distinct and include the rule choice")
        return self


class ToolLimits(ContractModel):
    deadline_mono: Annotated[float, Field(allow_inf_nan=False)]
    deep_eval_limit: Count
    candidate_target: PositiveInt
    candidate_hard_limit: PositiveInt

    @model_validator(mode="after")
    def limits_match(self) -> ToolLimits:
        if self.candidate_target > self.candidate_hard_limit or self.deep_eval_limit > self.candidate_hard_limit:
            raise ValueError("tool limits exceed the candidate hard limit")
        return self


class ToolResult(ContractModel):
    snapshot: CandidateSnapshot | None
    error: ErrorInfo | None
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def has_outcome(self) -> ToolResult:
        if self.snapshot is None and self.error is None:
            raise ValueError("tool result must contain a snapshot or error")
        return self


class ModelDecision(ContractModel):
    action_id: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]


CallRole = Literal["main", "review", "repair"]


class ModelCandidate(ContractModel):
    action_id: NonemptyStr
    type: NonemptyStr
    rank: NonemptyStr
    cards: FrozenArray[Card]  # PASS projection uses [], never its wire placeholder.
    features: ActionFeatures
    keep_reasons: FrozenArray[NonemptyStr]


class ModelInput(ContractModel):
    binding: RequestBinding
    candidate_snapshot_id: NonemptyStr
    role: CallRole
    visible: VisibleState
    candidates: Annotated[FrozenArray[ModelCandidate], Field(min_length=1)]
    platform_context: TributeContext | None = None
    main: ModelDecision | None = None
    trigger_codes: FrozenArray[NonemptyStr] = ()
    validation_error: str | None = None

    @model_validator(mode="after")
    def input_identity(self) -> ModelInput:
        if self.visible.binding != self.binding or self.candidate_snapshot_id != f"{self.binding.request_id}:c1":
            raise ValueError("model input binding and snapshot mismatch")
        ids = [c.action_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("model candidates must be unique")
        _check_stage_context(self.visible.stage, self.platform_context)
        return self


class Usage(ContractModel):
    input_tokens: Count | None = None
    output_tokens: Count | None = None
    total_tokens: Count | None = None
    cost_amount: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    currency: str | None = None
    cost_basis: str | None = None


class ProviderMessage(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ProviderInput(ContractModel):
    call_id: NonemptyStr
    messages: Annotated[FrozenArray[ProviderMessage], Field(min_length=1)]
    deadline_mono: Annotated[float, Field(allow_inf_nan=False)]
    max_output_tokens: PositiveInt


class ProviderReply(ContractModel):
    call_id: NonemptyStr
    content: str | None
    usage: Usage
    transport_requests: Count | None
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    error: ErrorInfo | None


class ModelCallResult(ContractModel):
    binding: RequestBinding
    candidate_snapshot_id: NonemptyStr
    role: CallRole
    call_id: NonemptyStr | None
    decision: ModelDecision | None
    error: ErrorInfo | None
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    queue_wait_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    calls_started: Annotated[int, Field(ge=0, le=1)]
    transport_requests: Count | None
    usage: Usage
    input_bytes: Count | None = None
    history_omitted: Count | None = None


class BudgetView(ContractModel):
    active: bool
    suggestions_open: bool
    remaining_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    slots_used: Count
    max_slots: Annotated[int, Field(ge=0, le=2)]


class CallTicket(ContractModel):
    call_id: NonemptyStr
    binding: RequestBinding
    role: CallRole
    ordinal: Annotated[int, Field(ge=1, le=2)]
    deadline_mono: Annotated[float, Field(allow_inf_nan=False)]


class MainReady(ContractModel):
    binding: RequestBinding
    candidate_snapshot_id: NonemptyStr
    call_id: NonemptyStr
    decision: ModelDecision
    trigger_codes: FrozenArray[NonemptyStr] = ()


class ToolsReady(ContractModel):
    binding: RequestBinding
    result: ToolResult


class CallFinished(ContractModel):
    binding: RequestBinding
    result: ModelCallResult


class DecisionInput(ContractModel):
    """Validated request with the single registered candidate snapshot."""

    context: RequestContext
    candidates: CandidateSnapshot | None = None

    @model_validator(mode="after")
    def candidate_binding(self) -> DecisionInput:
        if self.candidates is not None and self.candidates.binding != self.context.state.binding:
            raise ValueError("candidate snapshot belongs to another request")
        return self


class ReviewResult(ContractModel):
    """Program-derived second-call outcome; it grants no submission authority."""

    kind: Literal["review", "repair"]
    status: Literal["skipped", "kept", "changed", "repaired", "failed"]
    trigger_codes: FrozenArray[NonemptyStr]
    before: ModelDecision | None
    after: ModelDecision | None
    call_id: NonemptyStr | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def coherent_outcome(self) -> ReviewResult:
        if (self.kind == "review") != (self.before is not None):
            raise ValueError("review requires a valid main; repair has no valid main")
        if len(self.trigger_codes) != len(set(self.trigger_codes)):
            raise ValueError("review triggers must be unique")
        if self.status in {"kept", "changed", "repaired"}:
            if self.after is None or self.call_id is None or self.error is not None:
                raise ValueError("successful second call requires a decision and call ID")
            if self.kind == "repair":
                if self.status != "repaired":
                    raise ValueError("repair success must be repaired")
            elif self.status == "repaired":
                raise ValueError("review success must be kept or changed")
            elif (self.before.action_id == self.after.action_id) != (self.status == "kept"):
                raise ValueError("kept/changed must match action ID comparison")
        else:
            if self.after is not None or self.error is None:
                raise ValueError("skipped/failed second call requires an error and no decision")
            if self.status == "failed" and self.call_id is None:
                raise ValueError("failed second call requires its started call ID")
        return self


class DecisionProposal(ContractModel):
    """Bounded decision suggestion; Session checks recorded call provenance."""

    binding: RequestBinding
    candidate_snapshot_id: NonemptyStr | None
    action_id: NonemptyStr
    source: RuleSource
    reason: str
    cause_code: NonemptyStr | None
    review: ReviewResult | None

    @model_validator(mode="after")
    def main_requires_snapshot(self) -> DecisionProposal:
        if self.source in {"main", "review", "repair"} and self.candidate_snapshot_id != f"{self.binding.request_id}:c1":
            raise ValueError("model proposal requires its request snapshot")
        if self.source in {"review", "repair"} and (
                self.review is None or self.review.kind != self.source
                or self.review.status not in {"kept", "changed", "repaired"}
                or self.review.after.action_id != self.action_id
                or self.review.after.reason != self.reason):
            raise ValueError("second-call proposal must match its successful review result")
        return self


class DecisionOutcome(ContractModel):
    status: Literal["proposed", "invalidated", "unavailable"]
    proposal: DecisionProposal | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def proposal_only_on_success(self) -> DecisionOutcome:
        if (self.status == "proposed") != (self.proposal is not None):
            raise ValueError("only a proposed outcome must contain a proposal")
        return self


class SubmissionIntent(ContractModel):
    submission_id: NonemptyStr
    binding: RequestBinding
    action_id: NonemptyStr
    wire_action: WireAction
    stage: Stage
    platform_context: TributeContext | None
    final_deadline_mono: Annotated[float, Field(allow_inf_nan=False)]

    @model_validator(mode="after")
    def frozen_stage_context(self) -> SubmissionIntent:
        _check_stage_context(self.stage, self.platform_context)
        _check_stage_action(self.stage, self.wire_action)
        return self


class SendResult(ContractModel):
    submission_id: NonemptyStr
    status: Literal["sent", "not_started", "unknown"]
    error: ErrorInfo | None


class GateDecision(ContractModel):
    status: Literal["allow", "deny"]
    error: ErrorInfo | None


class SeatConnection(ContractModel):
    url: NonemptyStr
    mode: Literal["create", "join"]
    room_id: Count | None
    user_name: NonemptyStr
    seat: Seat
    session_id: UUIDString
    connection_epoch: PositiveInt
    server_rounds: PositiveInt

    @model_validator(mode="after")
    def room_mode_is_explicit(self) -> SeatConnection:
        if self.mode == "create" and (self.room_id is not None or self.server_rounds != 1):
            raise ValueError("create requires room_id=null and server_rounds=1")
        if self.mode == "join" and self.room_id is None:
            raise ValueError("join requires the returned room_id")
        return self


class PlatformEvent(ContractModel):
    """Envelope only; P03 validates each kind's raw protocol payload."""

    session_id: UUIDString
    connection_epoch: PositiveInt
    rx_seq: PositiveInt
    received_at: UTCTimestamp
    received_mono: Annotated[float, Field(allow_inf_nan=False)]
    kind: Literal[
        "joined", "join_failed", "round_started", "action_requested", "auxiliary_notice",
        "play_observed", "tribute_announced", "exchange_completed", "anti_tribute",
        "round_result", "match_result", "match_closed", "unknown",
    ]
    payload: FrozenJSONObject
    # Original JSON values for seat-scoped evidence; strategies never receive
    # this transport envelope. Invalid non-JSON input stays in unknown.payload.
    raw_message: FrozenJSONObject | None = None


class ExecutionRecord(ContractModel):
    submission_id: NonemptyStr
    binding: RequestBinding
    action_id: NonemptyStr
    wire_action: WireAction
    source: RuleSource
    status: Literal[
        "reserved", "writing", "sent", "confirmed", "cancelled_before_write", "unknown", "rejected"
    ]
    reserved_at: UTCTimestamp
    write_started_at: UTCTimestamp | None
    sent_at: UTCTimestamp | None
    confirmed_at: UTCTimestamp | None
    feedback_event_id: NonemptyStr | None
    error: ErrorInfo | None


class EventDraft(ContractModel):
    schema_version: Literal["1.0"]
    run_id: UUIDString
    match_id: UUIDString | None
    session_id: UUIDString | None
    seat: Seat | None
    round_id: PositiveInt | None
    request_id: NonemptyStr | None
    event_type: NonemptyStr
    payload: FrozenJSONObject

    @model_validator(mode="after")
    def event_request_ownership(self) -> EventDraft:
        if self.request_id is not None:
            if self.session_id is None or not re.fullmatch(
                re.escape(self.session_id) + r":q[1-9][0-9]*", self.request_id
            ):
                raise ValueError("request-level events must identify their owning session")
        return self


class LogEvent(EventDraft):
    event_id: NonemptyStr
    seq: PositiveInt
    recorded_at: UTCTimestamp

    @model_validator(mode="after")
    def event_id_matches_sequence(self) -> LogEvent:
        if self.event_id != f"{self.run_id}:e{self.seq}":
            raise ValueError("event_id must be <run_id>:e<seq>")
        return self


class EventQueryScope(ContractModel):
    """An online query always selects exactly one run, session and seat."""

    run_id: UUIDString
    session_id: UUIDString
    seat: Seat


class EventPage(ContractModel):
    items: FrozenArray[LogEvent]
    next_after_seq: Count | None
    has_more: bool
    retained_from_seq: PositiveInt
    log_complete: bool


class AppendResult(ContractModel):
    accepted: bool
    seq: PositiveInt
    event_id: NonemptyStr
    error: ErrorInfo | None


class RunSummary(ContractModel):
    schema_version: Literal["1.0"]
    run_id: UUIDString
    config_id: NonemptyStr
    closed: bool
    log_complete: bool
    last_persisted_seq: Count
    pending_call_ids: FrozenArray[NonemptyStr]
    unconfirmed_submission_ids: FrozenArray[NonemptyStr]
    error: ErrorInfo | None


class TableView(ContractModel):
    """Request-independent projection of one seat's current permitted facts."""

    stage: Literal["waiting", "play", "tribute", "back", "settled"]
    self_hand: FrozenArray[Card]
    level_rank: NonemptyStr | None
    team_ranks: Annotated[FrozenArray[NonemptyStr | None], Field(min_length=2, max_length=2)]
    remaining_counts: Annotated[FrozenArray[Count | None], Field(min_length=4, max_length=4)]
    leader_seat: Seat | None
    leading_action: WireAction | None
    finished_order: FrozenArray[Seat]
    facts_status: Literal["trusted", "degraded"]

    @model_validator(mode="after")
    def paired_leader(self) -> TableView:
        if (self.leader_seat is None) != (self.leading_action is None):
            raise ValueError("leader_seat and leading_action must be null together")
        return self


class SessionView(ContractModel):
    run_id: UUIDString
    match_id: UUIDString
    round_id: Count  # Zero until this connection has seen its first beginning.
    session_id: UUIDString
    seat: Seat
    state_version: Count
    view_version: Count
    updated_at: UTCTimestamp
    status: NonemptyStr
    state: TableView | None
    active_request_id: NonemptyStr | None
    decision_stage: NonemptyStr | None
    latest_execution: ExecutionRecord | None
    log_complete: bool
    error: ErrorInfo | None = None


class RunSessionView(ContractModel):
    session_id: UUIDString
    match_id: UUIDString
    seat: Seat
    status: NonemptyStr


class RunView(ContractModel):
    run_id: UUIDString
    status: Literal["starting", "running", "stopping", "completed", "interrupted"]
    config_id: NonemptyStr
    active_match_id: UUIDString | None
    sessions: FrozenArray[RunSessionView]
    log_complete: bool
    error: ErrorInfo | None = None


class RunListItem(RunView):
    """No private hands in the run picker; label is operator metadata only."""

    read_only: bool
    label: str


class DecisionSummary(ContractModel):
    request_id: NonemptyStr
    opened_seq: PositiveInt
    round_id: PositiveInt
    stage: Stage
    source: RuleSource | None
    execution_status: str | None
    updated_at: UTCTimestamp
    log_complete: bool


class DecisionPage(ContractModel):
    items: FrozenArray[DecisionSummary]
    next_after_seq: Count | None
    has_more: bool
    retained_from_seq: PositiveInt
    log_complete: bool


class DecisionDetail(ContractModel):
    state: VisibleState
    candidates: CandidateSnapshot | None
    proposal: DecisionProposal | None
    review: ReviewResult | None
    execution: ExecutionRecord | None
    calls: FrozenArray[ModelCallResult]
    log_complete: bool
    candidate_actions: FrozenArray[LegalAction] = ()
    # Derived at read time; never replaces or validates recorded model prose.
    program_explanation: FrozenArray[str] = ()


class ReviewReportItem(ContractModel):
    request_id: NonemptyStr
    session_id: UUIDString
    evidence_event_ids: FrozenArray[NonemptyStr]
    at_time_facts: FrozenArray[str]
    later_facts: FrozenArray[str]
    assessment: str
    suggestion: str
    # Legacy reports contain model-written summaries. Provenance is a claim;
    # readers must recompute program facts against the original source.
    facts_source: Literal["model", "program"] = "model"
    facts_verified: bool = False
    # The read layer recomputes these fields from source ownership. Stored/model
    # flags cannot authorize a link to another session or a missing decision.
    evidence_available: bool = False
    evidence_error: ErrorInfo | None = None


class ReviewReport(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    report_id: NonemptyStr
    kind: Literal["review"] = "review"
    run_id: UUIDString
    config_id: NonemptyStr
    status: Literal["completed", "partial", "failed"]
    created_at: UTCTimestamp
    items: FrozenArray[ReviewReportItem]
    usage: Usage
    errors: FrozenArray[ErrorInfo]
    coverage: FrozenJSONObject = Field(default_factory=dict)
    generation: FrozenJSONObject = Field(default_factory=dict)


class OfflineCallRecord(ContractModel):
    """Offline telemetry has no online request binding or action authority."""

    call_id: NonemptyStr
    calls_started: Annotated[int, Field(ge=0, le=1)]
    transport_requests: Count | None
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    usage: Usage
    error: ErrorInfo | None


class OfflineTaskView(ContractModel):
    task_id: UUIDString
    source_run_id: UUIDString
    status: Literal["running", "completed", "partial", "failed"]
    slots_used: Count
    calls_started: Count
    calls: FrozenArray[OfflineCallRecord]
    error: ErrorInfo | None
    report_ready: bool = False


class EvaluationReport(ContractModel):
    """Read-only envelope; metric computation belongs to B09/P21."""

    schema_version: Literal["1.0"] = "1.0"
    report_id: NonemptyStr
    kind: Literal["evaluation"] = "evaluation"
    run_ids: Annotated[FrozenArray[UUIDString], Field(min_length=1)]
    config_ids: Annotated[FrozenArray[NonemptyStr], Field(min_length=1)]
    status: Literal["completed", "partial", "failed"]
    created_at: UTCTimestamp
    unit: Literal["round", "match"]
    cohort: FrozenJSONObject
    counts: FrozenJSONObject
    performance: FrozenJSONObject
    latency: FrozenJSONObject
    model_usage: FrozenJSONObject
    quality: FrozenJSONObject

    @model_validator(mode="after")
    def source_pairs(self) -> EvaluationReport:
        if len(self.run_ids) != len(self.config_ids) or len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("evaluation source run/config pairs must be distinct and aligned")
        return self


class ReportSummary(ContractModel):
    report_id: NonemptyStr
    kind: Literal["review", "evaluation"]
    status: Literal["completed", "partial", "failed"]
    created_at: UTCTimestamp


class ReportPage(ContractModel):
    items: FrozenArray[ReportSummary]
    log_complete: bool
    errors: FrozenArray[ErrorInfo] = ()


class ProposalReady(ContractModel):
    binding: RequestBinding
    outcome: DecisionOutcome


class SuggestionDeadline(ContractModel):
    binding: RequestBinding


class SubmissionDeadline(ContractModel):
    binding: RequestBinding


class ConfirmationDeadline(ContractModel):
    binding: RequestBinding
    submission_id: NonemptyStr


class SendFinished(ContractModel):
    binding: RequestBinding
    result: SendResult


class Stop(ContractModel):
    run_id: UUIDString
    session_id: UUIDString
    reason: NonemptyStr = "stopped"


class ConnectionLost(ContractModel):
    run_id: UUIDString
    session_id: UUIDString
    reason: NonemptyStr = "connection_closed"


# Only the rule-run events needed in B03. Model/tool events arrive in B05/B07.
SessionEvent = (
    PlatformEvent | ProposalReady | SuggestionDeadline | SubmissionDeadline
    | ConfirmationDeadline | SendFinished | Stop | ConnectionLost | ToolsReady | MainReady | CallFinished
)
