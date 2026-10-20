"""Offline checks for the additional-call replay harness; no provider credentials."""
import asyncio
import copy
import json

import pytest

from guandan_agent.contracts import ProviderReply, Usage
from guandan_agent.model.provider import safe_error, unknown_usage
from scripts.run_model_validation25 import check_output, main, run_plan, validate_plan


def plan():
    return {"global_call_limit": 25, "model": "deepseek-flash", "cases": [
        {"case_id": f"case-{i:02}", "group": "main", "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": '{"candidates":[{"action_id":"L0000"}]}'},
        ], "expected": {"candidate_ids": ["L0000"]}, "max_output_tokens": 384,
         "timeout_s": 20} for i in range(25)]}


class Provider:
    def __init__(self, *, fail_at=None, status=None, transport=1, unknown=False, cancel=False):
        self.inputs = []
        self.closed = False
        self.fail_at, self.status, self.transport = fail_at, status, transport
        self.unknown, self.cancel = unknown, cancel

    async def complete(self, request):
        self.inputs.append(request)
        if self.cancel:
            raise asyncio.CancelledError()
        if len(self.inputs) == self.fail_at:
            raise RuntimeError("DO_NOT_LEAK_EXCEPTION_OR_KEY")
        error = safe_error("MODEL_TRANSPORT", f"Provider request failed with HTTP status {self.status}.") if self.status else None
        return ProviderReply(call_id=request.call_id, content='{"action_id":"L0000","reason":"测试"}',
                             usage=unknown_usage() if self.unknown else Usage(input_tokens=2, output_tokens=1, total_tokens=3),
                             transport_requests=self.transport, elapsed_ms=1.0, error=error)

    async def aclose(self):
        self.closed = True


def test_exact_schedule_failures_count_no_retry_and_durable_artifacts(tmp_path):
    provider = Provider(fail_at=3)
    output = tmp_path / "new"
    result = asyncio.run(run_plan(plan(), output, mode="scripted", provider=provider))
    assert result["calls_started"] == result["results_recorded"] == len(provider.inputs) == 25
    assert result["stop_reason"] == "schedule_completed"
    assert result["provider_errors"] == 1
    assert result["usage"]["total_tokens"] is None
    assert result["known_usage_subtotal"]["total_tokens"] == 72
    assert result["transport_requests"] is None
    assert result["known_transport_requests"] == 24
    assert provider.closed
    journal = (output / "calls.jsonl").read_text()
    rows = [json.loads(line) for line in journal.splitlines()]
    assert len(rows) == 50
    assert [row["case_id"] for row in rows if row["event"] == "begin"] == [case["case_id"] for case in plan()["cases"]]
    assert "DO_NOT_LEAK" not in journal
    assert len(list(output.glob("*.input.json"))) == len(list(output.glob("*.result.json"))) == 25


@pytest.mark.parametrize("status", [401, 402])
def test_fatal_auth_or_credit_stops_without_refill(tmp_path, status):
    provider = Provider(status=status)
    result = asyncio.run(run_plan(plan(), tmp_path / "new", mode="scripted", provider=provider))
    assert len(provider.inputs) == result["calls_started"] == 1
    assert result["stop_reason"] == "provider_auth_or_credit_failure"
    assert result["remaining_cases"] == 24


def test_multiple_transports_stops_the_batch(tmp_path):
    provider = Provider(transport=2)
    result = asyncio.run(run_plan(plan(), tmp_path / "new", mode="scripted", provider=provider))
    assert len(provider.inputs) == 1
    assert result["stop_reason"] == "transport_budget_violation"


def test_unknown_usage_stays_unknown(tmp_path):
    result = asyncio.run(run_plan(plan(), tmp_path / "new", mode="scripted", provider=Provider(unknown=True)))
    assert result["usage"] == dict.fromkeys(("input_tokens", "output_tokens", "total_tokens"))
    assert result["known_usage_subtotal"]["total_tokens"] == 0
    assert result["cost_amount"] is None
    assert result["real_provider"] is result["real_platform"] is False


def test_refuses_even_empty_existing_directory_before_call(tmp_path):
    provider = Provider()
    with pytest.raises(FileExistsError):
        asyncio.run(run_plan(plan(), tmp_path, mode="scripted", provider=provider))
    assert provider.inputs == []


def test_cancelled_call_consumes_slot_and_retains_checkpoint(tmp_path):
    provider = Provider(cancel=True)
    directory = tmp_path / "new"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_plan(plan(), directory, mode="scripted", provider=provider))
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["calls_started"] == 1 and summary["results_recorded"] == 0
    assert summary["stop_reason"] == "cancelled"
    assert summary["transport_requests"] is None and summary["usage"]["total_tokens"] is None
    assert provider.closed


def test_check_mode_does_not_read_key_or_construct_provider(tmp_path, monkeypatch):
    manifest = tmp_path / "plan.json"
    manifest.write_text(json.dumps(plan()))
    import scripts.run_model_validation25 as runner
    monkeypatch.setattr(runner.os, "environ", {})
    assert main(["--plan", str(manifest)]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["plan.json"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(global_call_limit=26),
    lambda p: p["cases"].append(copy.deepcopy(p["cases"][0])),
    lambda p: p["cases"][0].update(timeout_s=float("inf")),
    lambda p: p["cases"][0].update(max_output_tokens=1600),
    lambda p: p["cases"][0].update(case_id="../unsafe"),
    lambda p: p["cases"][0].update(timeout_s=True),
])
def test_invalid_manifest_cannot_start_calls(tmp_path, mutation):
    manifest = plan()
    mutation(manifest)
    provider = Provider()
    with pytest.raises(ValueError):
        asyncio.run(run_plan(manifest, tmp_path / "new", mode="scripted", provider=provider))
    assert provider.inputs == [] and not (tmp_path / "new").exists()


@pytest.mark.parametrize("fail_first", [False, True])
def test_review_depends_on_valid_real_result_or_explicit_seed(tmp_path, fail_first):
    manifest = plan()
    review = manifest["cases"][1]
    review["group"] = "review"
    review["messages"][-1]["content"] = '{"main":{"action_id":"L0000","reason":"placeholder"}}'
    review["dependency"] = {"main_case_id": "case-00", "fallback_main": {"action_id": "L0000", "reason": "seed"}}
    validate_plan(manifest)
    provider = Provider(fail_at=1 if fail_first else None)
    directory = tmp_path / "new"
    asyncio.run(run_plan(manifest, directory, mode="scripted", provider=provider))
    used = json.loads(provider.inputs[1].messages[-1].content)
    assert used["main"]["reason"] == ("seed" if fail_first else "测试")
    row = json.loads((directory / "case-01.result.json").read_text())
    assert row["dependency_fallback_used"] is fail_first
    assert manifest["cases"][1]["messages"][-1]["content"].endswith('"placeholder"}}')


@pytest.mark.parametrize("content,json_valid,schema_valid,member", [
    ('{"action_id":"L0000","reason":"ok"}', True, True, True),
    ('{"action_id":"L9999","reason":"ok"}', True, True, False),
    ('{"action_id":"L0000","reason":"ok","extra":1}', True, False, None),
    ('{"action_id":"L0000","action_id":"L0001","reason":"ok"}', False, None, None),
    ('not JSON', False, None, None),
])
def test_structure_and_membership_are_not_semantic_acceptance(content, json_valid, schema_valid, member):
    reply = ProviderReply(call_id="c", content=content, usage=unknown_usage(), transport_requests=1, elapsed_ms=1.0, error=None)
    result = check_output(plan()["cases"][0], reply)
    assert (result["json_valid"], result["schema_valid"], result["candidate_member"]) == (json_valid, schema_valid, member)
    assert result["semantic_judgment"] == "not_assessed"
