"""Acceptance checker must reject plausible but unsupported success records."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.verify_v0_run import main, verify_run

SOURCE = Path(__file__).resolve().parents[1] / "fixtures/audit/valid_scripted_run"


@pytest.fixture
def run_dir(tmp_path):
    for filename in ("config.json", "events.jsonl", "summary.json"):
        shutil.copyfile(SOURCE / filename, tmp_path / filename)
    return tmp_path


def rewrite(run_dir, mutation):
    path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(events)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_scripted_two_matches_have_six_independently_confirmed_chains(run_dir):
    result = verify_run(run_dir)
    assert result["passed"], result["reasons"]
    assert result["counts"]["matches"] == 2
    assert result["counts"]["sessions"] == 8
    assert result["counts"]["confirmed_chains"] == 6
    assert len({c["confirmation_event_id"] for c in result["chains"]}) == 6
    assert result["coverage"]["anti-tribute"] == 0  # No fabricated natural coverage.


def test_action_outside_original_legal_set_is_rejected(run_dir):
    def mutate(events):
        first = next(e["request_id"] for e in events if e["event_type"] == "request_opened")
        for e in events:
            if e["request_id"] == first and "execution" in e["payload"]:
                e["payload"]["execution"]["wire_action"] = ["Single", "A", ["SA"]]
    rewrite(run_dir, mutate)
    result = verify_run(run_dir)
    assert not result["passed"]
    assert any("outside request legal set" in r for r in result["reasons"])


def test_second_write_marker_is_rejected_even_with_confirmed_final_state(run_dir):
    def mutate(events):
        event = next(e for e in events if e["event_type"] == "execution_updated")
        event["event_type"] = "write_started"
        event["payload"]["execution"]["status"] = "writing"
    rewrite(run_dir, mutate)
    result = verify_run(run_dir)
    assert not result["passed"]
    assert any("one reserve/write/send" in r for r in result["reasons"])


def test_raw_notify_must_confirm_even_if_parsed_payload_and_final_status_claim_success(run_dir):
    def mutate(events):
        event = next(e for e in events if e["event_type"] == "platform_received"
                     and e["payload"]["event"]["kind"] == "play_observed")
        event["payload"]["event"]["raw_message"]["curAction"] = ["Single", "A", ["SA"]]
    rewrite(run_dir, mutate)
    result = verify_run(run_dir)
    assert not result["passed"]
    assert any("PLAY acceptance does not match" in r for r in result["reasons"])


def test_missing_confirmation_reference_is_rejected(run_dir):
    def mutate(events):
        for e in events:
            execution = e["payload"].get("execution")
            if execution and execution["status"] == "confirmed":
                execution["feedback_event_id"] = "missing-evidence"
    rewrite(run_dir, mutate)
    result = verify_run(run_dir)
    assert not result["passed"]
    assert any("missing original confirmation" in r for r in result["reasons"])


@pytest.mark.parametrize("pending", [False, True])
def test_incomplete_log_cannot_pass_from_success_markers(run_dir, pending):
    if pending:
        (run_dir / "summary.pending").write_text("uncommitted")
    else:
        path = run_dir / "summary.json"
        summary = json.loads(path.read_text())
        summary["log_complete"] = False
        path.write_text(json.dumps(summary))
    result = verify_run(run_dir)
    assert not result["passed"]
    assert any("log-complete" in r or "summary.pending" in r for r in result["reasons"])


def test_cli_writes_report_and_returns_nonzero_on_unreadable_evidence(run_dir, tmp_path, capsys):
    output = tmp_path / "reports" / "audit.json"
    assert main([str(run_dir), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["passed"]
    (run_dir / "events.jsonl").write_text('{"broken":')
    assert main([str(run_dir), "--output", str(output)]) == 1
    assert not json.loads(output.read_text())["passed"]
    assert "Malformed or missing evidence" in capsys.readouterr().out
