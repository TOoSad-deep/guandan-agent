"""B10 audits use actual runtime logs over a synthetic local room, never services."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from guandan_agent.clock import SystemClock
from guandan_agent.config import load_config
from guandan_agent.hand import HandTools
from guandan_agent.model.scripted import ScriptedProvider
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler
from scripts.verify_b10_run import canonical, digest, main, verify_run
from tests.support.scripted_room import ScriptedRooms

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def frozen_plan(tmp_path):
    """Standalone unit-test freeze; no generated real-match artifacts required."""
    entries = {}
    for policy in ("V0", "V1", "V2"):
        config = load_config(PROJECT / f"configs/b10-{policy.lower()}.toml")
        entries[policy] = {"config_id": config.config_id, "config": config.model_dump(mode="json")}
    path = tmp_path / "synthetic-frozen-plan.json"
    path.write_text(json.dumps({"batch": "B10", "tested_seat": 0, "configs": entries}))
    return path


def make_run(directory: Path, policy: str, *, steps: list[str] | None = None) -> Path:
    async def run():
        config = load_config(PROJECT / f"configs/b10-{policy.lower()}.toml")
        clock = SystemClock()
        rooms = ScriptedRooms(clock, pair_options=True)
        provider = ScriptedProvider(steps or ['{"action_id":"L0001","reason":"Synthetic legal choice"}'] * 2, clock=clock)
        manager = RunManager(clock=clock, scheduler=AsyncioScheduler(clock), recording_root=directory,
                             platform_factory=rooms.factory, provider_factory=lambda _: provider,
                             tools_factory=lambda _: HandTools(clock=clock))
        initial = manager.start(config)
        final = await asyncio.wait_for(manager.wait(initial.run_id), 4)
        assert final.status == "completed", final
        return directory / initial.run_id
    return asyncio.run(run())


@pytest.fixture
def v2(tmp_path):
    return make_run(tmp_path, "V2")


def mutate_events(directory, change):
    path = directory / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    change(events)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def select(events, kind):
    return next(event for event in events if event["event_type"] == kind)


@pytest.mark.parametrize("policy,source,calls", [("V0", "rule", 0), ("V1", "main", 1), ("V2", "review", 2)])
def test_runtime_generated_full_match_audits_all_three_policies(tmp_path, policy, source, calls):
    directory = make_run(tmp_path, policy)
    report = verify_run(directory)
    assert report["passed"], report["reasons"]
    assert report["counts"]["matches"] == 1 and report["counts"]["confirmed_chains"] == 3
    assert report["model"]["calls_started"] == calls
    assert report["path_index"][source][0]["seat"] == 0
    assert not report["accepted_live"] and not report["live_provenance_valid"]
    assert report["model"]["transport_requests"] == 0  # ScriptedProvider has no actual HTTP.


def test_repaired_invalid_main_has_distinct_confirmed_path(tmp_path):
    directory = make_run(tmp_path, "V2", steps=["not-json", '{"action_id":"L0001","reason":"Repaired"}'])
    report = verify_run(directory)
    assert report["passed"], report["reasons"]
    assert report["path_coverage"]["repair"] == {"observed": True, "count": 1}
    assert report["path_coverage"]["review"]["count"] == 0
    assert report["model"]["calls"][0]["error_code"] == "MODEL_SCHEMA"


def test_failed_second_call_remains_confirmed_fallback(tmp_path):
    directory = make_run(tmp_path, "V2", steps=['{"action_id":"L0001","reason":"Main"}', "not-json"])
    report = verify_run(directory)
    assert report["passed"], report["reasons"]
    assert report["path_coverage"]["fallback"]["count"] == 1
    assert report["path_coverage"]["review"]["count"] == 0


@pytest.mark.parametrize("change,fragment", [
    (lambda es: select(es, "request_opened")["payload"]["context"]["legal"]["actions"][0].update(source_indices=[99]), "original actionList"),
    (lambda es: select(es, "candidates_prepared")["payload"]["snapshot"]["candidates"][1].update(action_id="L9999"), "legal subset"),
    (lambda es: select(es, "main_registered")["payload"]["main"].update(call_id="wrong"), "bound successful result"),
    (lambda es: select(es, "main_registered")["payload"]["main"].update(trigger_codes=[]), "risk triggers"),
    (lambda es: select(es, "model_call_started")["payload"].update(ordinal=2), "start/admission"),
    (lambda es: select(es, "model_call_admitted")["payload"].update(run_slots_used=21), "admission counter"),
    (lambda es: select(es, "model_budget_summary")["payload"].update(calls_started=21), "budget counters"),
    (lambda es: select(es, "model_call_finished")["payload"]["result"].update(transport_requests=2), "transport exceeded"),
    (lambda es: select(es, "model_call_finished")["payload"].update(late=True), "bound successful result"),
    (lambda es: select(es, "review_finished")["payload"].update(candidate_snapshot_id="wrong"), "Review result/proposal"),
    (lambda es: select(es, "proposal_ready")["payload"]["proposal"].update(source="main"), "differs from submission"),
    (lambda es: select(es, "run_finished")["payload"].update(status="interrupted"), "did not finish completed"),
])
def test_forged_model_and_proposal_lineage_is_rejected(v2, change, fragment):
    mutate_events(v2, change)
    report = verify_run(v2)
    assert not report["passed"]
    assert any(fragment in reason for reason in report["reasons"]), report["reasons"]


def test_raw_notify_controls_confirmation_not_local_success(v2):
    def change(events):
        event = next(row for row in events if row["event_type"] == "platform_received" and
                     row["payload"]["event"]["kind"] == "play_observed")
        event["payload"]["event"]["raw_message"]["curAction"] = ["Pair", "A", ["SA", "HA"]]
    mutate_events(v2, change)
    report = verify_run(v2)
    assert not report["passed"]
    assert any("PLAY acceptance" in reason for reason in report["reasons"])


def test_second_write_rejected_even_with_final_confirmation(v2):
    mutate_events(v2, lambda es: select(es, "execution_updated").update(event_type="write_started"))
    report = verify_run(v2)
    assert not report["passed"]
    assert any("one reserve/write/send" in reason for reason in report["reasons"])


def test_wrong_raw_settlement_rejected(v2):
    def change(events):
        event = next(row for row in events if row["event_type"] == "platform_received" and
                     row["payload"]["event"]["raw_message"].get("stage") == "gameResult")
        event["payload"]["event"]["raw_message"]["victory"] = 1
    mutate_events(v2, change)
    report = verify_run(v2)
    assert not report["passed"]
    assert any("Settlement differs" in reason or "conflicting gameResult" in reason for reason in report["reasons"])


def test_incomplete_or_uncommitted_record_rejected(v2):
    (v2 / "summary.pending").write_text("not closed")
    report = verify_run(v2)
    assert not report["passed"]
    assert any("summary.pending" in reason for reason in report["reasons"])


def test_declaration_cannot_turn_scripted_model_into_live_evidence(v2):
    config = json.loads((v2 / "config.json").read_text())
    declaration = v2 / "declaration.json"
    declaration.write_text(json.dumps({"run_id": config["run_id"], "config_id": config["config_id"],
                                      "platform": "OpenGuanDanPlatform", "network_mode": "live",
                                      "provider": "deepseek", "model": "deepseek-flash"}))
    report = verify_run(v2, provenance=declaration)
    assert not report["passed"] and not report["accepted_live"]
    assert any("observed transport" in reason for reason in report["reasons"])


@pytest.mark.parametrize("change,fragment", [
    (lambda config: config["run"].update(split="development"), "one final match"),
    (lambda config: config["batch"].update(max_model_calls=21), "online budget 0/20"),
    (lambda config: config["offline"].update(max_model_calls=1), "offline budget zero"),
    (lambda config: config["run"]["seats"][1].update(policy="V1"), "three V0 seats"),
    (lambda config: config["versions"].update(prompt="b09-facts-v2"), "final shared rules"),
])
def test_unfrozen_evaluation_conditions_are_rejected(v2, change, fragment):
    path = v2 / "config.json"
    config_doc = json.loads(path.read_text())
    change(config_doc["config"])
    path.write_text(json.dumps(config_doc))
    report = verify_run(v2)
    assert not report["passed"]
    assert any(fragment in reason for reason in report["reasons"]), report["reasons"]


def test_unknown_usage_is_preserved_as_unknown(v2):
    mutate_events(v2, lambda es: select(es, "model_call_finished")["payload"]["result"]["usage"].update(total_tokens=None))
    report = verify_run(v2)
    assert report["passed"], report["reasons"]
    assert report["model"]["usage"]["total_tokens"] is None


def test_duplicate_model_result_is_rejected(v2):
    def change(events):
        first = select(events, "model_call_finished")
        second = [event for event in events if event["event_type"] == "model_call_finished"][1]
        second["payload"]["result"]["call_id"] = first["payload"]["result"]["call_id"]
    mutate_events(v2, change)
    report = verify_run(v2)
    assert not report["passed"]
    assert any("duplicate final model call" in reason for reason in report["reasons"])


def test_full_frozen_config_matches_without_claiming_synthetic_run_is_live(v2, frozen_plan):
    report = verify_run(v2, frozen_plan=frozen_plan)
    assert report["passed"] and report["frozen_config_valid"], report["reasons"]
    assert report["frozen_plan"]["version"] == "V2"
    assert not report["accepted_live"]


def test_without_freeze_live_declaration_cannot_grant_final_acceptance(tmp_path):
    directory = make_run(tmp_path, "V0")
    config = json.loads((directory / "config.json").read_text())
    declaration = tmp_path / "synthetic-declaration.json"
    declaration.write_text(json.dumps({"run_id": config["run_id"], "config_id": config["config_id"],
                                      "platform": "OpenGuanDanPlatform", "network_mode": "live",
                                      "synthetic_unit_test": True}))
    report = verify_run(directory, provenance=declaration)
    assert report["passed"] and report["live_provenance_valid"]
    assert not report["frozen_config_valid"] and not report["accepted_live"]


@pytest.mark.parametrize("change", [
    lambda config: config["model"].update(timeout_s=90.0),
    lambda config: config["model"].update(max_output_tokens=8192),
    lambda config: config["tools"].update(deep_eval_limit=0),
    lambda config: config["model"].update(name="different-model"),
])
def test_internally_rehashed_config_tampering_fails_frozen_comparison(v2, frozen_plan, change):
    """Recompute every internal config reference so only external freeze catches it."""
    path = v2 / "config.json"
    config_doc = json.loads(path.read_text())
    change(config_doc["config"])
    config_id = digest(canonical(config_doc["config"]).encode())
    config_doc["config_id"] = config_id
    path.write_text(json.dumps(config_doc))
    summary_path = v2 / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["config_id"] = config_id
    summary_path.write_text(json.dumps(summary))
    def change_events(events):
        for event in events:
            if "config_id" in event["payload"]:
                event["payload"]["config_id"] = config_id
            if event["event_type"] == "run_started":
                event["payload"]["config"] = config_doc["config"]
    mutate_events(v2, change_events)
    structural = verify_run(v2)
    assert structural["passed"], structural["reasons"]
    assert not structural["accepted_live"]
    frozen = verify_run(v2, frozen_plan=frozen_plan)
    assert not frozen["passed"] and not frozen["frozen_config_valid"] and not frozen["accepted_live"]
    assert any("frozen plan version/config_id/full" in reason for reason in frozen["reasons"])


def test_swapped_frozen_version_entry_is_rejected(v2, tmp_path, frozen_plan):
    plan = json.loads(frozen_plan.read_text())
    plan["configs"]["V2"] = plan["configs"]["V1"]
    path = tmp_path / "wrong-plan.json"
    path.write_text(json.dumps(plan))
    report = verify_run(v2, frozen_plan=path)
    assert not report["passed"] and not report["frozen_config_valid"]


def test_cli_never_overwrites_core_evidence(v2, capsys, frozen_plan):
    before = (v2 / "config.json").read_bytes()
    with pytest.raises(SystemExit):
        main([str(v2), "--output", str(v2 / "config.json")])
    assert (v2 / "config.json").read_bytes() == before
    assert main([str(v2), "--frozen-plan", str(frozen_plan), "--output", str(v2 / "reports/audit.json")]) == 0
    report = json.loads((v2 / "reports/audit.json").read_text())
    assert report["passed"] and report["frozen_config_valid"]
