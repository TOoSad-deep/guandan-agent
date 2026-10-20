"""Budget and dependency boundaries for the fixed factual-repair retest."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from guandan_agent.contracts import ModelDecision, ProviderReply, PublicPlay, Usage
from guandan_agent.model.gateway import build_messages
from guandan_agent.model.provider import safe_error
from scripts.run_model_retest15 import MAX_INPUT_BYTES, main, run_plan, validate_plan
from tests.unit.test_model_review import role_input


def plan(*, history=()):
    review_input = role_input(history=history)
    main_input = review_input.model_copy(update={"role": "main", "main": None, "trigger_codes": ()})
    source = {key: getattr(main_input.binding, key) for key in ("run_id", "session_id", "request_id")}
    candidates = [item.action_id for item in main_input.candidates]

    def case(identity, group, value):
        return {"case_id": identity, "group": group, "source": source,
                "messages": [message.model_dump(mode="json") for message in build_messages(value, max_input_bytes=MAX_INPUT_BYTES, prompt_version="b09-facts-v2")[0]],
                "expected": {"candidate_ids": candidates}, "max_output_tokens": 384, "timeout_s": 20,
                "prompt_version": "b09-facts-v2"}

    cases = [case(f"main_{index}", "main", main_input) for index in range(3)]
    for index in range(2):
        review = case(f"review_{index}", "review", review_input)
        review.update(model_input=review_input.model_dump(mode="json"), dependency={"main_case_id": f"main_{index}"})
        cases.append(review)
    for index in range(10):
        cases.append({"case_id": f"offline_{index}", "group": "offline", "source": source,
                      "expected": {"request_ids": [source["request_id"]]}, "max_output_tokens": 1600,
                      "timeout_s": 20, "prompt_version": "b09-offline-v2", "messages": [
                          {"role": "system", "content": "Return JSON analysis."},
                          {"role": "user", "content": json.dumps({"prompt_version": "b09-offline-v2",
                              "scope": "selected_requests_only", "decisions": [{"request_id": source["request_id"]}]})},
                      ]})
    return {"global_call_limit": 15, "model": "deepseek-flash", "cases": cases}


class Provider:
    def __init__(self, *, fail_case=None, main_content=None, status=None, transport=1, cancel=False):
        self.inputs, self.closed = [], False
        self.fail_case, self.main_content, self.status = fail_case, main_content, status
        self.transport, self.cancel = transport, cancel

    async def complete(self, request):
        self.inputs.append(request)
        if self.cancel:
            raise asyncio.CancelledError()
        if request.call_id == self.fail_case:
            raise RuntimeError("DO_NOT_LEAK_EXCEPTION_OR_KEY")
        content = '{"items":[{}]}' if request.call_id.startswith("offline") else '{"action_id":"L0000","reason":"实际主建议"}'
        if request.call_id.startswith("main") and self.main_content is not None:
            content = self.main_content
        error = safe_error("MODEL_TRANSPORT", f"Provider request failed with HTTP status {self.status}.") if self.status else None
        return ProviderReply(call_id=request.call_id, content=content, usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
                             transport_requests=self.transport, elapsed_ms=1.0, error=error)

    async def aclose(self):
        self.closed = True


def test_fifteen_fixed_slots_failures_consume_budget_and_artifacts_are_durable(tmp_path):
    provider = Provider(fail_case="offline_0")
    directory = tmp_path / "new"
    summary = asyncio.run(run_plan(plan(), directory, mode="scripted", provider=provider))
    assert summary["calls_started"] == summary["results_recorded"] == len(provider.inputs) == 15
    assert summary["cases_recorded"] == 15 and summary["skipped_cases"] == 0
    assert summary["stop_reason"] == "schedule_completed" and summary["remaining_cases"] == 0
    assert summary["provider_errors"] == 1 and summary["usage"]["total_tokens"] is None
    assert summary["known_usage_subtotal"]["total_tokens"] == 42
    assert summary["transport_requests"] is None and summary["known_transport_requests"] == 14
    assert summary["real_provider"] is summary["real_platform"] is False
    assert provider.closed
    text = (directory / "calls.jsonl").read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 30 and "DO_NOT_LEAK" not in text
    assert [row["case_id"] for row in rows if row["event"] == "begin"] == [case["case_id"] for case in plan()["cases"]]
    assert len(list(directory.glob("*.input.json"))) == len(list(directory.glob("*.result.json"))) == 15


@pytest.mark.parametrize("invalid", ["exception", "not JSON", '{"action_id":"L9999","reason":"wrong"}'])
def test_review_without_valid_main_is_skipped_and_never_replaced(tmp_path, invalid):
    provider = Provider(fail_case="main_0" if invalid == "exception" else None,
                        main_content=None if invalid == "exception" else invalid)
    directory = tmp_path / "new"
    summary = asyncio.run(run_plan(plan(), directory, mode="scripted", provider=provider))
    skip_count = 1 if invalid == "exception" else 2
    assert summary["calls_started"] == summary["results_recorded"] == 15 - skip_count
    assert summary["skipped_cases"] == skip_count and summary["cases_recorded"] == 15
    assert summary["remaining_cases"] == 0 and summary["stop_reason"] == "schedule_completed"
    assert "review_0" not in [request.call_id for request in provider.inputs]
    skipped = json.loads((directory / "review_0.result.json").read_text())
    assert skipped["event"] == "skipped" and skipped["reply"] is None
    assert not (directory / "review_0.input.json").exists()


def test_review_reprojects_actual_main_and_trims_history_to_byte_limit(tmp_path):
    history = tuple(PublicPlay(event_id=f"event-{index}-" + "x" * 150, seat=1,
                              action=("Single", "4", ("S4",))) for index in range(90))
    manifest = plan(history=history)
    reason = "实" * 240
    provider = Provider(main_content=json.dumps({"action_id": "L0000", "reason": reason}, ensure_ascii=False))
    directory = tmp_path / "new"
    asyncio.run(run_plan(manifest, directory, mode="scripted", provider=provider))
    actual = json.loads((directory / "review_0.input.json").read_text())
    payload = json.loads(actual["messages"][-1]["content"])
    original = json.loads(manifest["cases"][3]["messages"][-1]["content"])
    assert payload["main"] == {"action_id": "L0000", "reason": reason}
    assert payload["candidates"] == original["candidates"]
    assert actual["input_bytes"] <= MAX_INPUT_BYTES
    assert actual["history_omitted"] > original["visible"]["history_omitted"]
    assert payload["semantic_facts"]["history_omitted"] == actual["history_omitted"]
    assert manifest["cases"][3]["model_input"]["main"]["reason"] == "主判断尚待核对。"


@pytest.mark.parametrize("status", [401, 402])
def test_auth_or_credit_failure_stops_without_refill(tmp_path, status):
    provider = Provider(status=status)
    summary = asyncio.run(run_plan(plan(), tmp_path / "new", mode="scripted", provider=provider))
    assert len(provider.inputs) == summary["calls_started"] == 1
    assert summary["stop_reason"] == "provider_auth_or_credit_failure" and summary["remaining_cases"] == 14


def test_transport_budget_violation_stops_batch(tmp_path):
    provider = Provider(transport=2)
    summary = asyncio.run(run_plan(plan(), tmp_path / "new", mode="scripted", provider=provider))
    assert len(provider.inputs) == 1 and summary["stop_reason"] == "transport_budget_violation"


def test_existing_directory_is_rejected_before_calls(tmp_path):
    provider = Provider()
    with pytest.raises(FileExistsError):
        asyncio.run(run_plan(plan(), tmp_path, mode="scripted", provider=provider))
    assert provider.inputs == []


def test_cancelled_begin_remains_recorded_and_consumes_slot(tmp_path):
    provider, directory = Provider(cancel=True), tmp_path / "new"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_plan(plan(), directory, mode="scripted", provider=provider))
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["calls_started"] == 1 and summary["results_recorded"] == 0
    assert summary["stop_reason"] == "cancelled" and summary["usage"]["total_tokens"] is None
    assert provider.closed


def test_check_mode_does_not_read_credentials_construct_provider_or_write_output(tmp_path, monkeypatch):
    manifest = tmp_path / "plan.json"
    manifest.write_text(json.dumps(plan()))
    import scripts.run_model_retest15 as runner

    class NoEnvironment:
        def get(self, *_):
            raise AssertionError("Check mode must not access credentials")

    monkeypatch.setattr(runner, "os", SimpleNamespace(environ=NoEnvironment()))
    assert main(["--plan", str(manifest)]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["plan.json"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(global_call_limit=16),
    lambda p: p.update(model="other-model"),
    lambda p: p["cases"][0].update(group="offline"),
    lambda p: p["cases"][0].update(timeout_s=True),
    lambda p: p["cases"][0].update(max_output_tokens=385),
    lambda p: p["cases"][0].update(case_id="../unsafe"),
    lambda p: p["cases"][0].update(prompt_version="b07-review-v1"),
    lambda p: p["cases"][3].update(dependency={"main_case_id": "missing"}),
    lambda p: p["cases"][3]["dependency"].update(fallback_main={"action_id": "L0000", "reason": "seed"}),
    lambda p: p["cases"][3].update(source={**p["cases"][3]["source"], "request_id": "other"}),
    lambda p: p["cases"][3]["expected"].update(candidate_ids=["L0001"]),
    lambda p: p["cases"][3]["model_input"]["candidates"][0].update(cards=["S4"]),
    lambda p: p["cases"][5]["expected"].update(request_ids=["other"]),
    lambda p: p["cases"][0]["messages"][0].update(content="x" * 25000),
])
def test_invalid_plan_cannot_start_calls(tmp_path, mutation):
    manifest, provider = plan(), Provider()
    mutation(manifest)
    with pytest.raises(ValueError):
        asyncio.run(run_plan(manifest, tmp_path / "new", mode="scripted", provider=provider))
    assert not (tmp_path / "new").exists() and provider.inputs == []


def test_matching_candidate_ids_with_different_cards_is_rejected():
    manifest = plan()
    payload = json.loads(manifest["cases"][0]["messages"][-1]["content"])
    payload["candidates"][0]["cards"] = ["H9"]
    manifest["cases"][0]["messages"][-1]["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="same public state"):
        validate_plan(manifest)
