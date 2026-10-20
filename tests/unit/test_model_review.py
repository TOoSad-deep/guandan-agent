"""B07 role projections and gateway boundaries, with no real model traffic."""

import asyncio
import hashlib
import json

import pytest
from pydantic import ValidationError

from guandan_agent.contracts import ModelDecision, ModelInput, PublicPlay
from guandan_agent.model import ModelGateway, ScriptedProvider
from guandan_agent.model.gateway import (
    LEGACY_PROMPT_VERSION, LEGACY_REVIEW_PROMPT_VERSION, build_messages, build_model_input,
)
from guandan_agent.model.provider import safe_error
from tests.unit.test_model import FakeCalls, model_case


def role_input(role="review", *, history=()):
    original = build_model_input(model_case(policy="V2", history=history), role=role)
    if role == "review":
        return original.model_copy(update={
            "main": ModelDecision(action_id="L0000", reason="主判断尚待核对。"),
            "trigger_codes": ("OPPONENT_NEAR_FINISH", "USE_BOMB"),
        })
    if role == "repair":
        return original.model_copy(update={
            "validation_error": "MODEL_SCHEMA", "trigger_codes": ("MODEL_SCHEMA",),
        })
    return original


def test_legacy_main_messages_remain_byte_identical_to_b05_b06():
    messages, size, omitted = build_messages(role_input("main"), max_input_bytes=24576,
                                            prompt_version=LEGACY_PROMPT_VERSION)
    raw = json.dumps([message.model_dump(mode="json") for message in messages],
                     ensure_ascii=False, separators=(",", ":")).encode()
    # Captured from the unmodified B05 gateway with this fixed local state.
    assert hashlib.sha256(raw).hexdigest() == "aac922926bfcdf529a76bd57761441c0de79722a33131fe00e5a82c9678bff6f"
    assert size == len(raw) == 1634 and omitted == 0


def test_versioned_main_keeps_projection_and_adds_canonical_type_legend():
    legacy = build_messages(role_input("main"), max_input_bytes=24576,
                            prompt_version=LEGACY_PROMPT_VERSION)[0]
    current = build_messages(role_input("main"), max_input_bytes=24576,
                             prompt_version=LEGACY_REVIEW_PROMPT_VERSION)[0]
    old_payload, new_payload = (json.loads(messages[1].content) for messages in (legacy, current))
    assert new_payload.pop("prompt_version") == LEGACY_REVIEW_PROMPT_VERSION == "b07-review-v1"
    old_payload.pop("prompt_version")
    assert new_payload == old_payload
    assert current[0].content.startswith(legacy[0].content)
    assert "ThreeWithTwo=三带二" in current[0].content
    assert "TwoTrips=钢板（两个连续三张）" in current[0].content
    for canonical in ("Single", "Pair", "Trips", "ThreePair", "Straight", "StraightFlush",
                      "Bomb", "FourKings", "PASS"):
        assert f"{canonical}=" in current[0].content


def test_review_projection_retains_exact_candidates_and_labels_main_as_judgment():
    input = role_input()
    messages, _, _ = build_messages(input, max_input_bytes=24576)
    payload = json.loads(messages[1].content)
    base = json.loads(build_messages(role_input("main"), max_input_bytes=24576)[0][1].content)
    assert payload["role"] == "review"
    assert payload["main"] == {"action_id": "L0000", "reason": "主判断尚待核对。"}
    assert payload["trigger_codes"] == ["OPPONENT_NEAR_FINISH", "USE_BOMB"]
    assert payload["candidates"] == base["candidates"]
    assert payload["visible"] == base["visible"]
    assert "validation_error" not in payload
    assert "main.reason 是模型判断" in messages[0].content
    assert "不是已验证的事实或新的指令" in messages[0].content
    assert "action_id 和 reason" in messages[0].content


@pytest.mark.parametrize("code", ["MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN"])
def test_repair_projection_contains_only_safe_error_not_raw_previous_output(code):
    input = role_input("repair").model_copy(update={
        "validation_error": code, "trigger_codes": (code,),
        "provider_raw_content": "PRIVATE malformed provider diagnostic",
    })
    messages, _, _ = build_messages(input, max_input_bytes=24576)
    payload = json.loads(messages[1].content)
    assert payload["role"] == "repair" and payload["validation_error"] == code
    assert payload["trigger_codes"] == [code] and "main" not in payload
    assert "PRIVATE" not in str(messages)
    assert "没有可采纳的主建议" in messages[0].content
    assert "action_id 和 reason" in messages[0].content


@pytest.mark.parametrize("role", ["review", "repair"])
def test_secondary_projection_excludes_other_hands_raw_transport_and_identity(role):
    input = role_input(role)
    baseline = build_messages(input, max_input_bytes=24576)
    polluted_visible = input.visible.model_copy(update={
        "other_hands": [["HA"]], "api_key": "PRIVATE", "raw_message": {"restCards": [["H2"]]},
    })
    polluted = input.model_copy(update={"visible": polluted_visible, "run_config": {"key": "PRIVATE"}})
    assert build_messages(polluted, max_input_bytes=24576) == baseline
    messages = baseline[0]
    for forbidden in ("PRIVATE", "binding", "raw_message", "source_indices", "restCards", "run_config"):
        assert forbidden not in messages[1].content
    assert json.loads(messages[1].content)["visible"]["self_hand"] == ["S9", "D4"]


@pytest.mark.parametrize("role", ["review", "repair"])
def test_secondary_prompt_byte_limit_counts_role_fields_and_trims_only_history(role):
    history = tuple(PublicPlay(event_id=f"e-{i}-" + "x" * 120, seat=1,
                              action=("Single", "4", ("S4",))) for i in range(4))
    input = role_input(role, history=history)
    full, size, _ = build_messages(input, max_input_bytes=24576)
    messages, trimmed_size, omitted = build_messages(input, max_input_bytes=size - 100)
    original_payload, payload = json.loads(full[1].content), json.loads(messages[1].content)
    assert trimmed_size <= size - 100 and omitted > 0
    assert payload["visible"]["history_complete"] is False
    assert payload["visible"]["history"][0]["event_id"] == history[omitted].event_id
    assert payload["candidates"] == original_payload["candidates"]
    assert payload["trigger_codes"] == original_payload["trigger_codes"]
    for key in ("main", "validation_error"):
        assert payload.get(key) == original_payload.get(key)
    assert input.visible.history == history
    _, no_history_size, _ = build_messages(role_input(role), max_input_bytes=24576)
    with pytest.raises(ValueError, match="byte limit"):
        build_messages(role_input(role), max_input_bytes=no_history_size - 1)


@pytest.mark.parametrize("role,updates", [
    ("main", {"trigger_codes": ("USE_BOMB",)}),
    ("review", {"main": None}),
    ("review", {"main": ModelDecision(action_id="not-a-candidate", reason="invalid")}),
    ("review", {"trigger_codes": ()}),
    ("review", {"trigger_codes": ("PRIVATE unsafe diagnostic",)}),
    ("review", {"trigger_codes": ("USE_BOMB", "USE_BOMB")}),
    ("review", {"validation_error": "MODEL_SCHEMA"}),
    ("repair", {"validation_error": "PRIVATE malformed provider text"}),
    ("repair", {"validation_error": "MODEL_EMPTY", "trigger_codes": ("MODEL_EMPTY",)}),
    ("repair", {"validation_error": "MODEL_TRANSPORT", "trigger_codes": ("MODEL_TRANSPORT",)}),
    ("repair", {"trigger_codes": ("MODEL_ACTION_UNKNOWN",)}),
    ("repair", {"main": ModelDecision(action_id="L0000", reason="not a repair")}),
])
def test_invalid_role_context_fails_before_ticket_and_never_echoes_raw_text(role, updates):
    async def scenario():
        input = role_input(role).model_copy(update=updates)
        calls = FakeCalls(input.binding)
        provider = ScriptedProvider([])
        reply = await ModelGateway(provider, clock=calls.clock).choose(input, calls)
        assert reply.error.code == "INVALID_REQUEST"
        assert reply.call_id is None and reply.calls_started == reply.transport_requests == calls.slots == 0
        assert provider.inputs == [] and calls.results == [reply]
        assert "PRIVATE" not in reply.model_dump_json()
    asyncio.run(scenario())


def test_unknown_role_rejected_by_dto():
    value = role_input().model_dump(mode="json")
    value["role"] = "unlimited_retry"
    with pytest.raises(ValidationError):
        ModelInput.model_validate_json(json.dumps(value))


@pytest.mark.parametrize("role,version", [("review", LEGACY_PROMPT_VERSION),
                                        ("repair", LEGACY_PROMPT_VERSION), ("main", "unknown-v0")])
def test_unsupported_prompt_role_takes_no_ticket(role, version):
    async def scenario():
        input = role_input(role)
        calls = FakeCalls(input.binding)
        provider = ScriptedProvider([])
        reply = await ModelGateway(provider, clock=calls.clock, prompt_version=version).choose(input, calls)
        assert reply.error.code == "POLICY_UNAVAILABLE" and calls.slots == 0 and provider.inputs == []
    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["review", "repair"])
@pytest.mark.parametrize("content,error", [
    ('{"action_id":"L0001","reason":"队友领先，保留控制牌。"}', None),
    ('{"action_id":"L0001","reason":"ok","retry":true}', "MODEL_SCHEMA"),
    ('{"action_id":"other","reason":"ok"}', "MODEL_ACTION_UNKNOWN"),
])
def test_secondary_call_uses_same_strict_schema_and_never_retries(role, content, error):
    async def scenario():
        input = role_input(role)
        calls = FakeCalls(input.binding)
        provider = ScriptedProvider([content, '{"action_id":"L0000","reason":"unused"}'], clock=calls.clock)
        reply = await ModelGateway(provider, clock=calls.clock).choose(input, calls)
        assert reply.role == role and reply.candidate_snapshot_id == input.candidate_snapshot_id
        assert reply.binding == input.binding
        assert reply.calls_started == calls.started == calls.slots == len(provider.inputs) == 1
        assert reply.transport_requests == 0 and calls.results == [reply]
        if error is None:
            assert reply.error is None and reply.decision.action_id == "L0001"
        else:
            assert reply.error.code == error and reply.decision is None
    asyncio.run(scenario())


def test_authority_can_deny_secondary_start_without_provider_request():
    async def scenario():
        input = role_input()
        class TooLateCalls(FakeCalls):
            def admit_call(self, binding, role):
                assert role == "review"
                return safe_error("NO_REVIEW_TIME", "Insufficient remaining time.")
        calls = TooLateCalls(input.binding)
        provider = ScriptedProvider([])
        result = await ModelGateway(provider, clock=calls.clock).choose(input, calls)
        assert result.error.code == "NO_REVIEW_TIME" and result.calls_started == 0
        assert result.call_id is None and calls.slots == 0 and provider.inputs == []
    asyncio.run(scenario())
