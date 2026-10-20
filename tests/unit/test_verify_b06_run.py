"""B06 checker rejects plausible success markers without their raw evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.verify_b06_run import main, verify_run

SOURCE = Path(__file__).resolve().parents[1] / "fixtures/audit/b06_synthetic_short_chain"


@pytest.fixture
def evidence(tmp_path):
    for name in ("config.json", "events.jsonl", "summary.json"):
        shutil.copyfile(SOURCE / name, tmp_path / name)
    config = json.loads((tmp_path / "config.json").read_text())
    provenance = tmp_path / "synthetic-test-declaration.json"
    provenance.write_text(json.dumps({"run_id": config["run_id"], "provider": "deepseek",
                          "platform": "OpenGuanDanPlatform", "network_mode": "live",
                          "model": config["config"]["model"]["name"], "synthetic_unit_test": True}))
    return tmp_path, provenance


def mutate_events(run_dir, mutate):
    path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(events)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def select(events, kind):
    return next(e for e in events if e["event_type"] == kind)


def test_complete_synthetic_chain_exercises_acceptance_without_real_services(evidence):
    run_dir, provenance = evidence
    report = verify_run(run_dir, provenance=provenance)
    assert report["passed"], report["reasons"]
    assert report["chain_passed"] and report["live_provenance_valid"]
    assert report["model"]["calls_started"] == report["model"]["transport_requests"] == 1
    assert report["model"]["usage"]["total_tokens"] == 14
    assert report["counts"]["confirmed_main_chains"] == 1
    assert report["chains"][0]["confirmation_event_id"].endswith(":e27")
    assert report["chains"][0]["action_id"] == "L0001"
    assert report["claims"]["winner"] is None
    assert set(report["files_sha256"]) == {"config.json", "events.jsonl", "summary.json"}


def test_no_provenance_cannot_be_claimed_as_live(evidence):
    report = verify_run(evidence[0])
    assert report["chain_passed"]
    assert not report["passed"] and not report["live_provenance_valid"]
    assert any("provenance" in reason for reason in report["reasons"])


@pytest.mark.parametrize("field,value", [("run_id", "another-run"), ("provider", "scripted"),
                                         ("platform", "ScriptedPlatform"), ("network_mode", "mock"),
                                         ("model", "another-model")])
def test_provenance_must_match_actual_run_declaration(evidence, field, value):
    run_dir, provenance = evidence
    data = json.loads(provenance.read_text()); data[field] = value
    provenance.write_text(json.dumps(data))
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"] and not report["live_provenance_valid"]


@pytest.mark.parametrize("transport", [0, None, 2])
def test_scripted_unknown_or_retried_transport_cannot_pass_live_chain(evidence, transport):
    run_dir, provenance = evidence
    mutate_events(run_dir, lambda es: select(es, "model_call_finished")["payload"]["result"].update(transport_requests=transport))
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"] and not report["chain_passed"]
    if transport is None:
        assert report["model"]["transport_requests"] is None
        assert report["model"]["known_transport_requests"] == 0


def test_unknown_usage_remains_null_not_estimated(evidence):
    run_dir, provenance = evidence
    mutate_events(run_dir, lambda es: select(es, "model_call_finished")["payload"]["result"]["usage"].update(total_tokens=None))
    report = verify_run(run_dir, provenance=provenance)
    assert report["passed"], report["reasons"]
    assert report["model"]["usage"]["total_tokens"] is None
    assert report["model"]["usage"]["input_tokens"] == 11


@pytest.mark.parametrize("mutation,expected", [
    (lambda es: select(es, "request_opened")["payload"]["context"]["legal"]["actions"][0].update(source_indices=[9]), "original actionList"),
    (lambda es: select(es, "candidates_prepared")["payload"]["snapshot"]["candidates"][1].update(action_id="L9999"), "legal subset"),
    (lambda es: select(es, "main_registered")["payload"]["main"].update(candidate_snapshot_id="wrong"), "snapshot binding"),
    (lambda es: select(es, "main_registered")["payload"]["main"].update(call_id="wrong"), "call identity"),
    (lambda es: select(es, "model_call_started")["payload"].update(ordinal=2), "one main attempt"),
    (lambda es: select(es, "model_budget_summary")["payload"].update(calls_started=5), "budget counters"),
    (lambda es: select(es, "model_call_finished")["payload"]["result"].update(error={"code": "TIMEOUT"}), "successful result"),
    (lambda es: select(es, "run_finished")["payload"].update(status="completed"), "record interrupted"),
])
def test_forged_lineage_and_unbounded_accounting_rejected(evidence, mutation, expected):
    run_dir, provenance = evidence
    mutate_events(run_dir, mutation)
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"]
    assert any(expected in reason for reason in report["reasons"]), report["reasons"]


def test_raw_confirmation_must_match_even_if_all_normalized_markers_succeed(evidence):
    run_dir, provenance = evidence
    def mutate(events):
        e = next(e for e in events if e["event_type"] == "platform_received" and e["payload"]["event"]["kind"] == "play_observed")
        e["payload"]["event"]["raw_message"]["curAction"] = ["Single", "A", ["SA"]]
    mutate_events(run_dir, mutate)
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"] and not report["chain_passed"]
    assert any("Raw PLAY" in reason for reason in report["reasons"])


def test_exact_wire_action_cannot_be_changed_even_to_equivalent_card_order(evidence):
    run_dir, provenance = evidence
    def mutate(events):
        for e in events:
            if "execution" in e["payload"]:
                e["payload"]["execution"]["wire_action"][2].reverse()
    mutate_events(run_dir, mutate)
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"]
    assert any("outside original legal set" in reason for reason in report["reasons"])


def test_unknown_non_main_submission_is_reported_without_becoming_a_loss(evidence):
    run_dir, provenance = evidence
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text()); summary["unconfirmed_submission_ids"] = ["other-seat:q2:s1"]
    path.write_text(json.dumps(summary))
    mutate_events(run_dir, lambda es: select(es, "run_closed")["payload"].update(unconfirmed_submission_ids=["other-seat:q2:s1"]))
    report = verify_run(run_dir, provenance=provenance)
    assert report["passed"]
    assert report["unconfirmed_submission_ids"] == ["other-seat:q2:s1"]
    assert report["claims"]["winner"] is None


def test_pending_summary_and_missing_confirmation_cannot_pass(evidence):
    run_dir, provenance = evidence
    (run_dir / "summary.pending").touch()
    def mutate(events):
        for e in events:
            execution = e["payload"].get("execution")
            if execution and execution["status"] == "confirmed":
                execution["feedback_event_id"] = "missing"
    mutate_events(run_dir, mutate)
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"]
    assert any("summary.pending" in reason for reason in report["reasons"])
    assert any("original confirmation" in reason for reason in report["reasons"])


def test_sensitive_fields_are_flagged_without_returning_value(evidence):
    run_dir, provenance = evidence
    data = json.loads(provenance.read_text()); data["api_key"] = "synthetic-never-a-real-key"
    provenance.write_text(json.dumps(data))
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"]
    assert "synthetic-never-a-real-key" not in json.dumps(report)
    assert any("credential" in reason for reason in report["reasons"])


def test_cli_preserves_core_evidence_and_reports_malformed_input(evidence, capsys):
    run_dir, provenance = evidence
    out = run_dir / "reports" / "b06-audit.json"
    assert main([str(run_dir), "--provenance", str(provenance), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["passed"]
    original = (run_dir / "summary.json").read_bytes()
    with pytest.raises(SystemExit):
        main([str(run_dir), "--provenance", str(provenance), "--output", str(run_dir / "summary.json")])
    assert (run_dir / "summary.json").read_bytes() == original
    (run_dir / "events.jsonl").write_text('{"broken":')
    assert main([str(run_dir), "--provenance", str(provenance), "--output", str(out)]) == 1
    assert "Malformed or missing evidence" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["candidates_prepared", "model_call_admitted", "model_call_started",
                                    "model_call_finished", "main_registered", "write_started",
                                    "send_finished"])
def test_missing_chain_node_returns_failure_instead_of_crashing(evidence, missing):
    run_dir, provenance = evidence
    mutate_events(run_dir, lambda es: select(es, missing).update(event_type="missing_node"))
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"]
    assert report["reasons"]


def test_empty_log_returns_failure_instead_of_crashing(evidence):
    run_dir, provenance = evidence
    (run_dir / "events.jsonl").write_text("")
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"] and not report["chain_passed"]


def test_late_result_cannot_be_registered_as_successful_main_chain(evidence):
    run_dir, provenance = evidence
    mutate_events(run_dir, lambda es: select(es, "model_call_finished")["payload"].update(late=True))
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"] and not report["chain_passed"]
    assert any("suggestion deadline" in reason for reason in report["reasons"])


def test_consistently_forged_binding_cannot_override_raw_state_sequence(evidence):
    run_dir, provenance = evidence
    def change(value):
        if isinstance(value, dict):
            if "binding" in value:
                value["binding"]["state_version"] += 1
            for item in value.values():
                change(item)
        elif isinstance(value, list):
            for item in value:
                change(item)
    mutate_events(run_dir, change)
    report = verify_run(run_dir, provenance=provenance)
    assert not report["passed"]
    assert any("original connection/round/state" in reason for reason in report["reasons"])
