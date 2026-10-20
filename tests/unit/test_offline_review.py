"""Offline reviews have one bounded capability call and source-owned evidence."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from guandan_agent.config import RunConfig
from guandan_agent.contracts import ErrorInfo, LogEvent, ProviderReply, Usage
from guandan_agent.offline.review import (
    LEGACY_PROMPT_VERSION, FACTS_PROMPT_VERSION as PROMPT_VERSION,
    PROMPT_VERSION as CURRENT_PROMPT_VERSION, _current_input, _messages, _select,
    generate_review, verify_program_facts,
)
from guandan_agent.recording.store import ClosedRunLog


SOURCE = Path(__file__).resolve().parents[1] / "fixtures/audit/valid_scripted_run"
CREATED = "2026-10-19T12:00:00Z"


def load_source():
    return asyncio.run(ClosedRunLog.load(SOURCE, cache_events=20000))


def clone(source, *, events=None, closed=True, complete=True, evicted=0):
    events = source.cached_events if events is None else tuple(events)
    return ClosedRunLog(run_id=source.run_id, config_id=source.config_id, events=events,
                        summary=source.summary.model_copy(update={"closed": closed, "log_complete": complete}),
                        evicted_through=evicted, max_query_limit=200, config=source.config)


def mutate(source, change):
    rows = [item.model_dump(mode="json") for item in source.cached_events]
    change(rows)
    return clone(source, events=[LogEvent.model_validate_json(json.dumps(item)) for item in rows])


def valid_output(inputs):
    items = []
    for item in inputs:
        items.append({"request_id": item["request_id"],
                      "evidence_event_ids": [item["anchor_event_id"]],
                      "at_time_facts": ["仅本座位的请求时手牌可见。"], "later_facts": [],
                      "assessment": "需结合可见信息判断，单例不证明策略强弱。",
                      "suggestion": "可比较保留的合法选项，但不能断言换牌必胜。"})
    return {"items": items}


class Calls:
    task_id = "offline-review-test"

    def __init__(self, mutate_output=None, *, content=None, error=None, raises=None):
        self.messages = []
        self.inputs = []
        self.change = mutate_output
        self.content = content
        self.error = error
        self.raises = raises

    async def call(self, messages):
        self.messages.append(messages)
        self.inputs = json.loads(messages[-1].content)["decisions"]
        if self.raises:
            raise self.raises
        output = valid_output(self.inputs)
        if json.loads(messages[-1].content).get("prompt_version") in {PROMPT_VERSION, CURRENT_PROMPT_VERSION}:
            for item in output["items"]:
                del item["at_time_facts"], item["later_facts"]
        if self.change:
            self.change(output, self.inputs)
        return ProviderReply(call_id="offline-call", content=self.content if self.content is not None else json.dumps(output),
                             usage=Usage(input_tokens=11, output_tokens=13, total_tokens=24),
                             transport_requests=1, elapsed_ms=1.0, error=self.error)


def run(source, calls=None, config=None, **kwargs):
    calls = calls or Calls()
    kwargs.setdefault("prompt_version", LEGACY_PROMPT_VERSION)
    report = asyncio.run(generate_review(source, config or RunConfig(), calls,
                                       report_id=calls.task_id, created_at=CREATED, **kwargs))
    return report, calls


def multiple_requests(source, count=4):
    """Repeat one tiny recorded decision with new request/event identities."""
    anchor = next(item for item in source.cached_events if item.event_type == "request_opened" and item.seat == 0)
    original = [item for item in source.cached_events if item.request_id == anchor.request_id]
    rows = list(source.cached_events)
    seq = rows[-1].seq
    for index in range(2, count + 1):
        for item in original:
            seq += 1
            text = item.model_dump_json().replace(anchor.request_id, f"{anchor.session_id}:q{index}")
            value = json.loads(text)
            value.update(seq=seq, event_id=f"{source.run_id}:e{seq}")
            rows.append(LogEvent.model_validate_json(json.dumps(value)))
    return clone(source, events=rows)


def test_single_call_uses_source_identity_and_own_offline_usage():
    source = load_source()
    report, calls = run(source)
    assert report.status == "completed"
    assert len(calls.messages) == len(report.items) == 1
    assert report.run_id == source.run_id and report.config_id == source.config_id
    assert report.report_id == calls.task_id
    assert report.usage.total_tokens == 24 and report.usage.cost_amount is None
    assert report.coverage["selection_scope"] == "selected_requests"
    assert report.coverage["selected_request_ids"] == (report.items[0].request_id,)
    assert report.generation["prompt_version"] == "b09-offline-v1"
    assert report.generation["offline_config_id"] == RunConfig().config_id
    anchor = next(row for row in source.cached_events if row.event_type == "request_opened")
    assert report.items[0].session_id == anchor.session_id and report.items[0].evidence_available
    assert calls.inputs[0]["recorded_decision"]["execution_status"] == "confirmed"
    assert calls.inputs[0]["recorded_judgment"]["classification"] == "recorded_judgment_not_verified_fact"


def test_default_selection_is_deterministic_bounded_and_one_session():
    source = multiple_requests(load_source(), 5)
    first, calls = run(source)
    second, _ = run(source)
    assert first == second
    assert len(first.items) == len(calls.inputs) == 3
    assert len({item.session_id for item in first.items}) == 1
    assert first.coverage["selection_limited"]
    assert first.coverage["available_request_count"] == 5
    assert first.status == "completed"  # Explicit selected scope; not whole game.


def test_explicit_selection_and_nonzero_seat_fallback():
    source = load_source()
    anchor = next(row for row in source.cached_events if row.event_type == "request_opened" and row.seat == 1)
    report, calls = run(source, request_ids=(anchor.request_id,))
    assert report.status == "completed" and report.items[0].session_id == anchor.session_id
    assert calls.inputs[0]["at_time"]["turn_seat"] == 1
    no_zero = clone(source, events=[row for row in source.cached_events if row.seat != 0])
    report, calls = run(no_zero)
    assert report.items[0].session_id == anchor.session_id


@pytest.mark.parametrize("kind", ["cross_session", "missing", "empty", "duplicate", "too_many", "owner"])
def test_invalid_explicit_selection_never_calls(kind):
    source = load_source()
    anchors = [row for row in source.cached_events if row.event_type == "request_opened"]
    first, other = anchors[:2]
    ids = {"cross_session": (first.request_id, other.request_id), "missing": ("no-request",),
           "empty": (), "duplicate": (first.request_id, first.request_id),
           "too_many": tuple(row.request_id for row in anchors[:4]), "owner": (first.request_id,)}[kind]
    kwargs = {"request_ids": ids}
    if kind == "owner":
        kwargs["session_id"] = other.session_id
    report, calls = run(source, **kwargs)
    assert report.status == "failed" and not calls.messages
    assert any(error.code == "REVIEW_SELECTION_INVALID" for error in report.errors)


def test_not_closed_or_wrong_task_identity_never_calls():
    source = clone(load_source(), closed=False)
    report, calls = run(source)
    assert report.status == "failed" and not calls.messages
    assert report.errors[0].code == "RUN_NOT_CLOSED"
    calls = Calls()
    report = asyncio.run(generate_review(load_source(), RunConfig(), calls,
                                       report_id="foreign-task", created_at=CREATED))
    assert report.errors[0].code == "REVIEW_TASK_ID_MISMATCH" and not calls.messages


@pytest.mark.parametrize("incomplete", ["gap", "window", "no_anchor"])
def test_incomplete_evidence_is_partial_or_failed_without_fabrication(incomplete):
    source = load_source()
    if incomplete == "gap":
        source = clone(source, complete=False)
    else:
        anchor = next(row for row in source.cached_events if row.event_type == "request_opened")
        retained = anchor.seq + int(incomplete == "no_anchor")
        source = clone(source, events=[row for row in source.cached_events if row.seq >= retained], evicted=retained - 1)
        if incomplete == "no_anchor":
            source = clone(source, events=[row for row in source.cached_events if row.session_id == anchor.session_id], evicted=retained - 1)
    report, calls = run(source)
    if incomplete == "no_anchor":
        assert report.status == "failed" and not calls.messages
    else:
        assert report.status == "partial" and len(calls.messages) == 1
        assert not report.items[0].evidence_available
        assert report.items[0].evidence_error.code == "EVIDENCE_UNAVAILABLE"


@pytest.mark.parametrize("prompt_version", [LEGACY_PROMPT_VERSION, PROMPT_VERSION])
def test_input_contains_only_typed_own_state_actions_and_public_later_fields(prompt_version):
    source = load_source()
    def inject(rows):
        for row in rows:
            if row["event_type"] == "platform_received":
                row["payload"]["event"]["raw_message"] = {"other_hands": "HIDDEN_RAW_MARKER"}
                row["payload"]["event"]["payload"]["restCards"] = [[1, ["HIDDEN_REST_MARKER"]]]
                row["payload"]["event"]["payload"]["other_hands"] = "HIDDEN_NESTED_MARKER"
    source = mutate(source, inject)
    report, calls = run(source, prompt_version=prompt_version)
    payload = calls.messages[0][-1].content
    assert "HIDDEN_" not in payload and "restCards" not in payload and "raw_message" not in payload
    assert "config_id" not in payload and "session_id" not in payload
    assert calls.inputs[0]["at_time"]["self_hand"] == ["S8"]
    assert "SJ" not in payload  # Other seat's revealed leftover is excluded even after settlement.
    assert report.status == "completed"


def test_malformed_later_public_values_are_excluded():
    source = load_source()
    def corrupt(rows):
        for row in rows:
            if row["event_type"] == "platform_received" and row["payload"]["event"]["kind"] == "play_observed":
                row["payload"]["event"]["payload"]["curAction"] = {"other_hand": "HIDDEN_MARKER"}
    report, calls = run(mutate(source, corrupt))
    assert report.status == "completed"
    assert all(row["kind"] != "play_observed" for row in calls.inputs[0]["later_public_facts"])
    assert "HIDDEN_MARKER" not in calls.messages[0][-1].content


def test_legacy_v0_alternatives_come_from_original_legal_set_and_are_bounded():
    source = load_source()
    def expand(rows):
        anchor = next(row for row in rows if row["event_type"] == "request_opened")
        context = anchor["payload"]["context"]
        context["state"]["self_hand"] = [f"S{x}" for x in "23456789TJQKA"]
        context["legal"]["raw_count"] = 13
        context["legal"]["actions"] = [
            {"action_id": f"L{i:04d}", "source_indices": [i], "wire_action": ["Single", rank, [f"S{rank}"]]}
            for i, rank in enumerate("82345679TJQKA")]
    report, calls = run(mutate(source, expand))
    item = calls.inputs[0]
    assert report.status == "completed"
    assert len(item["alternatives"]) == 6 and item["alternatives_omitted"] == 7
    assert item["alternatives_source"] == "original_legal_actions_bounded_subset"
    assert item["alternatives"][0]["wire_action"] == ["Single", "8", ["S8"]]


@pytest.mark.parametrize("mutation", ["missing", "foreign", "other_private", "before", "later_unlabelled", "later_unreferenced", "missing_anchor", "duplicate_reference"])
def test_invalid_evidence_is_rejected(mutation):
    source = load_source()
    anchor = next(row for row in source.cached_events if row.event_type == "request_opened")
    other = next(row for row in source.cached_events if row.event_type == "request_opened" and row.session_id != anchor.session_id)
    before = next(row for row in source.cached_events if row.session_id == anchor.session_id and row.seq < anchor.seq)
    def change(output, inputs):
        item = output["items"][0]
        if mutation in {"missing", "foreign", "other_private", "before"}:
            item["evidence_event_ids"].append({"missing": "missing-event", "foreign": other.event_id,
                                               "other_private": other.event_id, "before": before.event_id}[mutation])
        elif mutation == "later_unlabelled":
            item["evidence_event_ids"].append(inputs[0]["later_public_facts"][0]["event_id"])
        elif mutation == "later_unreferenced":
            item["later_facts"] = ["此后某玩家出牌。"]
        elif mutation == "missing_anchor":
            item["evidence_event_ids"] = [inputs[0]["direct_evidence_event_ids"][-1]]
        else:
            item["evidence_event_ids"].append(item["evidence_event_ids"][0])
    report, calls = run(source, Calls(change))
    assert report.status == "failed" and not report.items and len(calls.messages) == 1
    assert any(error.code == "REVIEW_ITEM_INVALID" for error in report.errors)


def test_later_fact_needs_allowed_reference_and_is_kept_separate():
    def change(output, inputs):
        output["items"][0]["evidence_event_ids"].append(inputs[0]["later_public_facts"][0]["event_id"])
        output["items"][0]["later_facts"] = ["事后公开记录出现出牌通知。"]
    report, _ = run(load_source(), Calls(change))
    assert report.status == "completed" and report.items[0].later_facts


def test_another_request_private_context_is_not_later_public_evidence():
    source = multiple_requests(load_source(), 2)
    def change(output, inputs):
        output["items"][0]["evidence_event_ids"].append(inputs[1]["anchor_event_id"])
        output["items"][0]["later_facts"] = ["声称来自另一请求的当时手牌。"]
    report, _ = run(source, Calls(change))
    assert report.status == "partial" and len(report.items) == 1
    assert report.items[0].request_id.endswith(":q2")


@pytest.mark.parametrize("field", ["session_id", "run_id", "report_id", "config_id", "usage", "evidence_available"])
def test_model_cannot_author_metadata(field):
    def change(output, _):
        output["items"][0][field] = "injected"
    report, _ = run(load_source(), Calls(change))
    assert report.status == "failed" and not report.items


@pytest.mark.parametrize("content", ["", "not JSON", "```json\n{}\n```", "{}", '{"items":[]}', '{"items":[],"usage":{}}', '{"items":[],"items":[]}'])
def test_invalid_json_or_empty_output_never_retries(content):
    report, calls = run(load_source(), Calls(content=content))
    assert report.status == "failed" and len(calls.messages) == 1 and report.usage.total_tokens == 24


@pytest.mark.parametrize("problem", ["wrong_request", "empty_assessment", "long_fact", "wrong_type", "extra_item"])
def test_invalid_items_allow_valid_subset(problem):
    source = multiple_requests(load_source(), 3)
    def change(output, inputs):
        if problem == "wrong_request":
            output["items"][1]["request_id"] = "unselected:q1"
        elif problem == "empty_assessment":
            output["items"][1]["assessment"] = "  "
        elif problem == "long_fact":
            output["items"][1]["at_time_facts"] = ["x" * 241]
        elif problem == "wrong_type":
            output["items"][1]["suggestion"] = 123
        else:
            output["items"][1] = dict(output["items"][0])
    report, calls = run(source, Calls(change))
    assert report.status == "partial" and len(report.items) == 2 and len(calls.messages) == 1
    assert report.coverage["reviewed_request_ids"] == tuple(item.request_id for item in report.items)


def test_transport_error_and_exception_are_safe_and_preserve_only_known_usage():
    error = ErrorInfo(code="MODEL_TIMEOUT", component="provider", safe_message="调用超时", related_event_id=None)
    report, calls = run(load_source(), Calls(error=error))
    assert report.status == "failed" and report.usage.total_tokens == 24 and len(calls.messages) == 1
    report, calls = run(load_source(), Calls(raises=RuntimeError("SENSITIVE_EXCEPTION_DETAIL")))
    assert report.status == "failed" and report.usage.total_tokens is None and len(calls.messages) == 1
    assert "SENSITIVE" not in report.model_dump_json()


def test_cancellation_propagates_to_runtime_authority():
    calls = Calls(raises=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run(load_source(), calls)
    assert len(calls.messages) == 1


@pytest.mark.parametrize("prompt_version", [LEGACY_PROMPT_VERSION, PROMPT_VERSION])
def test_input_budget_prevents_call_when_minimal_evidence_does_not_fit(prompt_version):
    config = RunConfig(model={"max_input_bytes": 128})
    report, calls = run(load_source(), config=config, prompt_version=prompt_version)
    assert report.status == "failed" and not calls.messages
    assert any(error.code == "REVIEW_INPUT_TOO_LARGE" for error in report.errors)


def test_input_budget_reduces_scope_with_explicit_partial_coverage():
    source = multiple_requests(load_source(), 3)
    _, full_calls = run(source)
    full_size = len(json.dumps([m.model_dump(mode="json") for m in full_calls.messages[0]],
                               ensure_ascii=False, separators=(",", ":")).encode())
    config = RunConfig(model={"max_input_bytes": full_size - 1000})
    report, calls = run(source, config=config)
    assert report.status == "partial" and len(calls.messages) == 1
    measured = len(json.dumps([m.model_dump(mode="json") for m in calls.messages[0]],
                              ensure_ascii=False, separators=(",", ":")).encode())
    assert measured <= config.model.max_input_bytes
    assert report.coverage["selection_limited"]
    assert any(error.code == "REVIEW_INPUT_REDUCED" for error in report.errors)


@pytest.mark.parametrize("kind", ["context", "execution", "duplicate_anchor"])
def test_invalid_recorded_binding_or_action_is_not_reviewed(kind):
    source = load_source()
    anchor = next(row for row in source.cached_events if row.event_type == "request_opened")
    def change(rows):
        for row in rows:
            if row["request_id"] != anchor.request_id:
                continue
            if kind == "context" and row["event_type"] == "request_opened":
                row["payload"]["context"]["state"]["self_hand"] = ["S2"]
            if kind == "execution" and "execution" in row["payload"]:
                row["payload"]["execution"]["wire_action"] = ["Single", "2", ["S2"]]
        if kind == "duplicate_anchor":
            extra = json.loads(anchor.model_dump_json())
            seq = rows[-1]["seq"] + 1
            extra.update(seq=seq, event_id=f"{source.run_id}:e{seq}")
            rows.append(extra)
    report, calls = run(mutate(source, change), session_id=anchor.session_id)
    assert report.status == "failed" and not calls.messages
    assert any(error.code == "REVIEW_SOURCE_EVIDENCE_INVALID" for error in report.errors)


@pytest.mark.parametrize("prompt_version", [LEGACY_PROMPT_VERSION, PROMPT_VERSION])
def test_same_source_is_not_mutated_by_review_generation(prompt_version):
    source = load_source()
    before = tuple(item.model_dump_json() for item in source.cached_events)
    summary = source.summary.model_dump_json()
    run(source, prompt_version=prompt_version)
    assert before == tuple(item.model_dump_json() for item in source.cached_events)
    assert summary == source.summary.model_dump_json()


def test_legacy_prompt_messages_remain_byte_identical():
    _, calls = run(load_source())
    raw = json.dumps([message.model_dump(mode="json") for message in calls.messages[0]],
                     ensure_ascii=False, separators=(",", ":")).encode()
    assert len(raw) == 3641
    assert hashlib.sha256(raw).hexdigest() == "3e7245ba01256a110e1ce0c284146323b39a8cc0e8c15e9978e258e1903b55b8"


def test_default_v2_owns_facts_and_references_and_separates_execution_time():
    source, calls = load_source(), Calls()
    report = asyncio.run(generate_review(source, RunConfig(), calls,
                                        report_id=calls.task_id, created_at=CREATED, prompt_version=PROMPT_VERSION))
    assert report.status == "completed" and report.generation["prompt_version"] == PROMPT_VERSION
    item = report.items[0]
    assert item.facts_source == "program" and item.facts_verified
    assert verify_program_facts(source, item)
    assert all("confirmed" not in text for text in item.at_time_facts)
    assert any("执行状态=confirmed" in text for text in item.later_facts)
    assert any("黑桃8（S8）" in text for text in item.at_time_facts)
    assert any("队伍0（座位0、2）获胜" in text for text in item.later_facts)
    assert set(calls.inputs[0]["later_public_facts"][0]) == {"event_id", "kind", "facts"}
    assert calls.inputs[0]["semantics"]["seat_roles"]["partner"] == 2


@pytest.mark.parametrize("field", ["at_time_facts", "later_facts", "facts_source", "facts_verified"])
def test_v2_rejects_model_authored_facts_or_provenance(field):
    def inject(output, _):
        output["items"][0][field] = ["伪造事实"] if field.endswith("facts") else "program"
    report, calls = run(load_source(), Calls(inject), prompt_version=PROMPT_VERSION)
    assert report.status == "failed" and not report.items and len(calls.messages) == 1
    assert any(error.code == "REVIEW_ITEM_INVALID" for error in report.errors)


def test_v2_compact_input_does_not_change_program_facts_or_reintroduce_history():
    source = load_source()
    complete, _ = run(source, prompt_version=PROMPT_VERSION)
    selected, _, _ = _select(source, None, None)
    compact = _messages([_current_input(source, item, compact=True) for item in selected], prompt_version=PROMPT_VERSION)
    limit = len(json.dumps([m.model_dump(mode="json") for m in compact], ensure_ascii=False, separators=(",", ":")).encode())
    report, calls = run(source, config=RunConfig(model={"max_input_bytes": limit}), prompt_version=PROMPT_VERSION)
    assert report.status == "partial" and len(calls.messages) == 1
    assert len(calls.inputs[0]["later_public_facts"]) == 1
    assert report.items[0].at_time_facts == complete.items[0].at_time_facts
    assert report.items[0].later_facts == complete.items[0].later_facts
    assert report.items[0].evidence_event_ids == complete.items[0].evidence_event_ids
    assert verify_program_facts(source, report.items[0])
    assert len(calls.inputs[0]["semantics"]["history"]) == len(calls.inputs[0]["at_time"]["history"]) <= 1

    omitted_id = complete.items[0].evidence_event_ids[-1]
    def cite_omitted(output, _):
        output["items"][0]["evidence_event_ids"].append(omitted_id)
    invalid, _ = run(source, Calls(cite_omitted), config=RunConfig(model={"max_input_bytes": limit}),
                     prompt_version=PROMPT_VERSION)
    assert invalid.status == "failed" and not invalid.items


def test_v2_exchange_facts_keep_unknown_counts_and_ignore_historical_judgment():
    source = load_source()
    anchor = next(row for row in source.cached_events if row.event_type == "request_opened")
    def exchange(rows):
        for row in rows:
            if row["request_id"] != anchor.request_id:
                continue
            if row["event_type"] == "request_opened":
                context = row["payload"]["context"]
                context["state"].update(stage="back", self_hand=["D2"] * 27, level_rank="3",
                                        remaining_counts=[27, None, None, None])
                context["platform_context"] = {"tribute_pos": 2, "tribute_card": "SB"}
                context["legal"]["actions"][0]["wire_action"] = ["back", "back", ["D2"]]
            if "execution" in row["payload"]:
                row["payload"]["execution"]["wire_action"] = ["back", "back", ["D2"]]
            if row["event_type"] == "proposal_ready":
                row["payload"]["proposal"]["reason"] = "错误旧理由：大王SB，后卫阶段普通首出，剩26张。"
    source = mutate(source, exchange)
    report, calls = run(source, session_id=anchor.session_id, prompt_version=PROMPT_VERSION)
    assert report.status == "completed"
    item = report.items[0]
    facts = "\n".join((*item.at_time_facts, *item.later_facts))
    assert "阶段：还贡（back）" in facts and "手牌共27张" in facts and "座位1=未知" in facts
    assert "小王（SB）" in facts and "不能按普通出牌减一" in facts
    assert "剩26张" not in facts and "后卫" not in facts and "大王SB" not in facts
    assert "错误旧理由" in calls.inputs[0]["recorded_judgment"]["reason"]
    assert verify_program_facts(source, item)


def test_v2_incomplete_source_cannot_mark_facts_verified():
    source = clone(load_source(), complete=False)
    report, _ = run(source, prompt_version=PROMPT_VERSION)
    assert report.status == "partial" and report.items[0].facts_source == "program"
    assert not report.items[0].facts_verified and not verify_program_facts(source, report.items[0])


def test_unsupported_offline_prompt_version_does_not_call():
    report, calls = run(load_source(), prompt_version="invented-version")
    assert report.status == "failed" and not calls.messages
    assert report.errors[0].code == "REVIEW_PROMPT_UNSUPPORTED"
