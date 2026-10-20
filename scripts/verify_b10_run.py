#!/usr/bin/env python3
"""Independent B10 full-match audit of frozen config, wire evidence and model lineage.

The raw-frame/settlement checks are adapted from verify_v0_run; the historical
checker remains unchanged. No source record is rewritten or error filtered.
A provenance declaration is operator evidence, never a network packet capture.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from guandan_agent.config import RunConfig
from guandan_agent.contracts import CandidateSnapshot, DecisionProposal, MainReady, ModelCallResult, RequestContext, ReviewResult

PASS = ["PASS", "PASS", "PASS"]
BINDING = ("run_id", "match_id", "round_id", "session_id", "seat", "connection_epoch", "request_id", "state_version")
ENVELOPE = ("run_id", "match_id", "round_id", "session_id", "seat", "request_id")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def same_action(left: list, right: list) -> bool:
    if left == PASS or right == PASS:
        return left == right
    return left[:2] == right[:2] and Counter(left[2]) == Counter(right[2])


def legal_actions(rows: list) -> list[dict]:
    groups: dict[str, dict] = {}
    for i, row in enumerate(rows):
        key = canonical(row)
        if key not in groups:
            groups[key] = {"action_id": f"L{i:04d}", "source_indices": [], "wire_action": row}
        groups[key]["source_indices"].append(i)
    return list(groups.values())


def _sources(policy: str) -> set[str]:
    return {"unique", "rule", "fallback"} | ({"main"} if policy == "V1" else
            {"main", "review", "repair"} if policy == "V2" else set())


def _config(config: dict, check: Any, report: dict) -> None:
    RunConfig.model_validate(config)
    seats = {str(seat["seat"]): seat["policy"] for seat in config["run"]["seats"]}
    policy = seats.get("0")
    report.update(policy=policy, seat_policies=seats)
    check(config["run"]["split"] == "final" and config["run"]["matches"] == 1,
          "B10 requires one final match")
    check(policy in {"V0", "V1", "V2"} and all(seats.get(str(s)) == "V0" for s in (1, 2, 3))
          and config["run"]["evaluation_team"] == 0, "B10 requires seat 0 under evaluation and three V0 seats")
    check(config["batch"]["max_model_calls"] == (0 if policy == "V0" else 20)
          and config["offline"]["max_model_calls"] == 0, "B10 requires online budget 0/20 and offline budget zero")
    check(config["model"]["max_retries"] == 0 and config["model"]["thinking"] is False
          and config["model"]["concurrency"] == 1, "B10 requires the frozen nonthinking zero-retry provider profile")
    check(config["review"]["enabled"] == (policy == "V2"), "Review configuration differs from evaluated policy")
    check(config["versions"] == {"policy": "b05-rules-v1", "candidate": "b05-candidates-v1",
          "fallback": "b02-cheap-v1", "prompt": "b09-facts-v3"}, "B10 requires final shared rules/candidate/prompt versions")


def _provenance(path: Path, config_doc: dict, check: Any, report: dict) -> None:
    raw = path.read_bytes()
    declaration = json.loads(raw)
    config = config_doc["config"]
    valid = (declaration.get("run_id") == config_doc["run_id"]
             and declaration.get("config_id") == config_doc["config_id"]
             and declaration.get("platform") == "OpenGuanDanPlatform"
             and declaration.get("network_mode") == "live")
    if report["policy"] != "V0":
        valid = valid and declaration.get("provider") == "deepseek" and declaration.get("model") == config["model"]["name"]
        # Successful model-backed actions require observed provider transport;
        # a declaration alone cannot relabel ScriptedProvider evidence as live.
        for chain in report["model_chains"]:
            for call_id in chain["call_ids"]:
                call = next(row for row in report["model"]["calls"] if row["call_id"] == call_id)
                check(call["transport_requests"] == 1, "Live model chain requires one observed transport per call")
        check(report["model"]["calls_started"] > 0 and report["model"]["known_transport_requests"] > 0,
              "Live model version has no observed provider transport")
    report["provenance_sha256"] = digest(raw)
    report["live_provenance_valid"] = bool(valid)
    check(valid, "Live execution provenance is missing or mismatched")


def _frozen_config(path: Path, config_doc: dict, check: Any, report: dict) -> None:
    """Compare against the separately frozen full effective configuration."""
    raw = path.read_bytes()
    plan = json.loads(raw)
    policy = report["policy"]
    entry = plan["configs"][policy]
    expected = entry["config"]
    effective = RunConfig.model_validate(expected).model_dump(mode="json")
    expected_id = digest(canonical(expected).encode())
    valid = (plan.get("batch") == "B10" and plan.get("tested_seat") == 0
             and canonical(expected) == canonical(effective)
             and entry["config_id"] == expected_id == config_doc["config_id"]
             and canonical(config_doc["config"]) == canonical(expected))
    report["frozen_config_valid"] = bool(valid)
    report["frozen_plan"] = {"path": str(path.resolve()), "sha256": digest(raw),
                             "version": policy, "expected_config_id": entry["config_id"],
                             "scope": "version and complete effective configuration"}
    check(valid, "Run does not match the frozen plan version/config_id/full effective configuration")


def _risks(context: dict, decision: dict) -> list[str]:
    """Independent small transcription of the frozen three risk rules."""
    state = context["state"]
    if state["stage"] != "play":
        return []
    seat = state["binding"]["seat"]
    action = next(row for row in context["legal"]["actions"] if row["action_id"] == decision["action_id"])
    triggers = []
    if any(s % 2 != seat % 2 and s not in state["finished_order"] and count in (1, 2)
           for s, count in enumerate(state["remaining_counts"])):
        triggers.append("OPPONENT_NEAR_FINISH")
    if action["wire_action"][0] in {"Bomb", "StraightFlush", "FourKings"}:
        triggers.append("USE_BOMB")
    if state["leader_seat"] == (seat + 2) % 4 and action["wire_action"][0] != "PASS":
        triggers.append("BEAT_PARTNER")
    return triggers


def _model_audit(events: list[dict], config: dict, check: Any, report: dict) -> None:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    admitted, started, finished, results = {}, {}, {}, {}
    summaries = []
    for event in events:
        kind, payload, request_id = event["event_type"], event["payload"], event["request_id"]
        if request_id is not None:
            grouped[request_id][kind].append(event)
        if kind in {"model_call_admitted", "model_call_started"}:
            target = admitted if kind == "model_call_admitted" else started
            call_id = payload["call_id"]
            check(call_id not in target, "Duplicate model admission/start", event)
            target[call_id] = event
        elif kind == "model_call_finished":
            result = payload["result"]
            ModelCallResult.model_validate(result)
            call_id = result["call_id"]
            check(call_id is not None and call_id not in finished, "Missing/duplicate final model call ID", event)
            finished[call_id] = event
            results[call_id] = dict(result)
        elif kind == "model_call_usage_updated":
            result = results.get(payload["call_id"])
            check(result is not None, "Usage update lacks a prior result", event)
            if result is not None:
                check(all(value is None or payload["usage"].get(key) == value for key, value in result["usage"].items()),
                      "Usage update changed known telemetry", event)
                check(result["transport_requests"] is None or result["transport_requests"] == payload["transport_requests"],
                      "Usage update changed known transport count", event)
                result["usage"], result["transport_requests"] = payload["usage"], payload["transport_requests"]
        elif kind == "model_budget_summary":
            summaries.append(event)
    check(set(started) <= set(admitted) and set(finished) == set(admitted), "Model admission/start/result ledger mismatch")
    check(len(started) <= len(admitted) <= config["batch"]["max_model_calls"], "Run model budget exceeded")
    check(len(summaries) == 1 and summaries[0]["payload"] == {"slots_used": len(admitted), "calls_started": len(started)},
          "Final model budget counters disagree with ledger")
    for ordinal, event in enumerate(admitted.values(), 1):
        check(event["payload"]["run_slots_used"] == ordinal, "Run admission counter mismatch", event)
    for ordinal, event in enumerate(started.values(), 1):
        call_id = event["payload"]["call_id"]
        admission = admitted.get(call_id)
        check(event["payload"]["run_calls_started"] == ordinal and event["payload"]["calls_started"] == 1,
              "Run started counter mismatch", event)
        check(admission is not None and admission["seq"] < event["seq"] and
              all(admission[key] == event[key] for key in ENVELOPE) and
              all(admission["payload"][key] == event["payload"][key] for key in ("call_id", "role", "ordinal")),
              "Call start/admission binding or order mismatch", event)
    for call_id, result in results.items():
        admission = admitted.get(call_id)
        check(admission is not None and admission["seq"] < finished[call_id]["seq"]
              and all(result["binding"][key] == admission[key] == finished[call_id][key] for key in ENVELOPE)
              and result["role"] == admission["payload"]["role"], "Call result/admission binding or order mismatch")
        check(result["calls_started"] == int(call_id in started), "Model result/start accounting mismatch")
        check(result["transport_requests"] is None or type(result["transport_requests"]) is int and 0 <= result["transport_requests"] <= 1,
              "Provider transport exceeded one request per call")
        check(call_id in started or result["transport_requests"] in (None, 0), "Unstarted call reported transport")
        check(call_id not in started or started[call_id]["seq"] < finished[call_id]["seq"], "Model result preceded call start")
    paths = {source: [] for source in ("unique", "rule", "fallback", "main", "review", "repair")}
    model_chains = []
    for request_id, kinds in grouped.items():
        if not kinds["request_opened"]:
            check(not any(kinds[k] for k in ("model_call_admitted", "model_call_started", "model_call_finished")),
                  "Model event has no request")
            continue
        event = kinds["request_opened"][0]
        context = event["payload"]["context"]
        RequestContext.model_validate(context)
        binding = context["state"]["binding"]
        limit = {"V0": 0, "V1": 1, "V2": 2}[context["policy"]]
        admissions = kinds["model_call_admitted"]
        check(len(admissions) <= limit, "Per-request model budget exceeded", event)
        snapshot = None
        if kinds["candidates_prepared"]:
            check(len(kinds["candidates_prepared"]) == 1, "Duplicate candidate snapshots", event)
            node = kinds["candidates_prepared"][0]
            snapshot = node["payload"]["snapshot"]
            CandidateSnapshot.model_validate(snapshot)
            ids = [row["action_id"] for row in snapshot["candidates"]]
            legal_ids = {row["action_id"] for row in context["legal"]["actions"]}
            check(snapshot["binding"] == binding and all(node[key] == binding[key] for key in ENVELOPE)
                  and event["seq"] < node["seq"], "Candidate binding/order mismatch", node)
            check(len(ids) <= config["tools"]["candidate_hard_limit"] and set(ids) <= legal_ids,
                  "Candidate set is not a bounded legal subset", node)
        if admissions:
            check(snapshot is not None, "Model call lacks registered candidates", event)
        for ordinal, admission in enumerate(admissions, 1):
            payload = admission["payload"]
            expected_role = "main" if ordinal == 1 else payload["role"]
            check(payload["ordinal"] == payload["slots_used"] == ordinal and
                  payload["call_id"] == f"{request_id}:m{ordinal}" and payload["role"] == expected_role and
                  (ordinal == 1 or payload["role"] in {"review", "repair"}), "Model role/ordinal/identity mismatch", admission)
            check(all(admission[key] == binding[key] for key in ENVELOPE), "Admission request binding mismatch", admission)
            if snapshot is not None:
                check(kinds["candidates_prepared"][0]["seq"] < admission["seq"], "Call preceded candidate registration", admission)
            result = results.get(payload["call_id"])
            if result is not None:
                check(result["binding"] == binding and result["candidate_snapshot_id"] == f"{request_id}:c1",
                      "Result candidate snapshot binding mismatch", finished[payload["call_id"]])
                if result["decision"] is not None and snapshot is not None:
                    check(result["error"] is None and result["decision"]["action_id"] in {row["action_id"] for row in snapshot["candidates"]},
                          "Model decision is outside candidates or has error")
            if ordinal == 2:
                first_id = f"{request_id}:m1"
                first = results.get(first_id)
                check(first is not None and first["calls_started"] == 1 and finished[first_id]["seq"] < admission["seq"],
                      "Second call preceded a completed main", admission)
                if first is not None and payload["role"] == "repair":
                    check(first["decision"] is None and (first["error"] or {}).get("code") in {"MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN"},
                          "Repair lacks repairable main failure", admission)
                elif payload["role"] == "review":
                    check(len(kinds["main_registered"]) == 1 and kinds["main_registered"][0]["seq"] < admission["seq"],
                          "Review lacks prior registered main", admission)
        main = None
        check(len(kinds["main_registered"]) <= 1, "Duplicate main registration", event)
        if kinds["main_registered"]:
            node = kinds["main_registered"][0]
            main = node["payload"]["main"]
            MainReady.model_validate(main)
            call_id = main["call_id"]
            result = results.get(call_id)
            check(main["binding"] == binding and main["candidate_snapshot_id"] == f"{request_id}:c1"
                  and call_id == f"{request_id}:m1" and result is not None and result["decision"] == main["decision"]
                  and result["error"] is None and call_id in started and finished[call_id]["payload"]["late"] is False
                  and finished[call_id]["seq"] < node["seq"], "Registered main lacks valid bound successful result", node)
            expected_triggers = _risks(context, main["decision"]) if context["policy"] == "V2" else []
            check(main["trigger_codes"] == expected_triggers, "Main risk triggers disagree with visible state", node)
        proposals = kinds["proposal_ready"]
        check(len(proposals) == 1, "Exactly one adopted proposal is required per request", event)
        if len(proposals) != 1 or not kinds["submission_reserved"]:
            continue
        proposal_event = proposals[0]
        proposal = proposal_event["payload"]["proposal"]
        DecisionProposal.model_validate(proposal)
        final = kinds["submission_reserved"][0]["payload"]["execution"]
        check(proposal["binding"] == binding and proposal["action_id"] == final["action_id"]
              and proposal["source"] == final["source"] and proposal_event["seq"] < kinds["submission_reserved"][0]["seq"],
              "Adopted proposal differs from submission", proposal_event)
        source = final["source"]
        if source == "rule":
            check(context["policy"] == "V0" and snapshot is not None and proposal["action_id"] == snapshot["rule_action_id"],
                  "Rule submission differs from registered rule choice", proposal_event)
        elif source == "fallback":
            check(proposal["action_id"] == context["fallback_action_id"], "Fallback differs from prevalidated choice", proposal_event)
        elif source == "unique":
            check(len(context["legal"]["actions"]) == 1 and not admissions, "Unique path cannot consume model calls", proposal_event)
        elif source == "main":
            check(main is not None and proposal["action_id"] == main["decision"]["action_id"]
                  and proposal["reason"] == main["decision"]["reason"] and kinds["main_registered"][0]["seq"] < proposal_event["seq"],
                  "Main submission differs from registered main", proposal_event)
            check(not any(row["payload"]["role"] in {"review", "repair"} for row in kinds["model_call_started"]),
                  "Main reused after a second call started", proposal_event)
        review = proposal["review"]
        review_nodes = kinds["review_finished"]
        check(len(review_nodes) == int(review is not None), "Review result count differs from adopted proposal", proposal_event)
        if review is not None:
            ReviewResult.model_validate(review)
            node = review_nodes[0]
            check(node["payload"]["binding"] == binding and node["payload"]["candidate_snapshot_id"] == f"{request_id}:c1"
                  and node["payload"]["review"] == review and node["seq"] < proposal_event["seq"],
                  "Review result/proposal binding mismatch", node)
            if review["kind"] == "review":
                check(main is not None and review["before"] == main["decision"] and review["trigger_codes"] == main["trigger_codes"]
                      and bool(review["trigger_codes"]), "Review lacks matching risk-triggered main", node)
            else:
                first = results.get(f"{request_id}:m1")
                check(first is not None and first["decision"] is None and
                      review["trigger_codes"] == [(first["error"] or {}).get("code")] and
                      (first["error"] or {}).get("code") in {"MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN"}, "Repair lacks invalid main", node)
            call_id = review["call_id"]
            second = results.get(call_id)
            if review["status"] in {"kept", "changed", "repaired"}:
                check(call_id == f"{request_id}:m2" and call_id in started and second is not None and
                      second["role"] == review["kind"] and second["error"] is None and second["decision"] == review["after"] and
                      finished[call_id]["payload"]["late"] is False and finished[call_id]["seq"] < node["seq"] and source == review["kind"],
                      "Second-call submission lacks valid bound successful result", node)
            elif review["status"] == "failed":
                check(call_id in started and second is not None and second["error"] == review["error"] and source == "fallback",
                      "Failed second call must match result and fallback", node)
            else:
                check(call_id not in started and source == ("main" if review["kind"] == "review" else "fallback"),
                      "Skipped second call has incompatible source/start", node)
        elif source in {"review", "repair"}:
            check(False, "Second-call source has no review result", proposal_event)
        chain = next((row for row in report["chains"] if row["request_id"] == request_id), None)
        if chain is not None:
            indexed = {**chain, "seat": binding["seat"], "policy": context["policy"], "action_id": proposal["action_id"],
                       "candidate_snapshot_id": proposal["candidate_snapshot_id"], "proposal_event_id": proposal_event["event_id"],
                       "model_call_ids": [row["payload"]["call_id"] for row in admissions],
                       "model_result_event_ids": [row["event_id"] for row in kinds["model_call_finished"]],
                       "main_registered_event_ids": [row["event_id"] for row in kinds["main_registered"]],
                       "review_finished_event_ids": [row["event_id"] for row in review_nodes]}
            paths[source].append(indexed)
            if source in {"main", "review", "repair"}:
                model_chains.append({**indexed, "call_ids": indexed["model_call_ids"]})
    known_transport = sum(row["transport_requests"] or 0 for row in results.values())
    unknown_transport = [call_id for call_id in started if results.get(call_id, {}).get("transport_requests") is None]
    usage = {name: sum(results[call_id]["usage"][name] for call_id in started)
             if all(results.get(call_id, {}).get("usage", {}).get(name) is not None for call_id in started) else None
             for name in ("input_tokens", "output_tokens", "total_tokens")}
    report["model"] = {"slots_used": len(admitted), "calls_started": len(started),
                       "transport_requests": None if unknown_transport else known_transport,
                       "known_transport_requests": known_transport, "unknown_transport_call_ids": unknown_transport,
                       "usage": usage, "calls": [{"call_id": call_id, "role": row["role"],
                         "transport_requests": row["transport_requests"], "usage": row["usage"],
                         "error_code": (row["error"] or {}).get("code")} for call_id, row in results.items()]}
    report["model_chains"] = model_chains
    report["path_index"] = paths
    report["path_coverage"] = {source: {"observed": bool(rows), "count": len(rows)} for source, rows in paths.items()}


def verify_run(run_dir: Path, *, provenance: Path | None = None,
               frozen_plan: Path | None = None) -> dict:
    """Return a JSON-safe report, retaining specific failures for interrupted runs."""
    reasons: list[str] = []
    report: dict[str, Any] = {"schema_version": "1.0", "run_dir": str(run_dir.resolve()),
                              "passed": False, "accepted_live": False, "live_provenance_valid": False,
                              "frozen_config_valid": False, "frozen_plan": None, "reasons": reasons,
                              "acceptance_boundary": "passed checks record structure; accepted_live additionally requires live provenance and an explicit matching frozen plan"}

    def check(condition: Any, message: str, event: dict | None = None) -> None:
        if not condition:
            reasons.append(f"{event['event_id']}: {message}" if event else message)

    try:
        # Full files are read once; no bounded UI/history cache is used here.
        config_bytes = (run_dir / "config.json").read_bytes()
        event_bytes = (run_dir / "events.jsonl").read_bytes()
        summary_bytes = (run_dir / "summary.json").read_bytes()
        config_doc, summary = json.loads(config_bytes), json.loads(summary_bytes)
        events = [json.loads(line) for line in event_bytes.splitlines()]
        report["files_sha256"] = {"config.json": digest(config_bytes), "events.jsonl": digest(event_bytes),
                                  "summary.json": digest(summary_bytes)}
        _audit(events, config_doc, summary, run_dir, check, report)
        _model_audit(events, config_doc["config"], check, report)
        if frozen_plan is not None:
            _frozen_config(frozen_plan, config_doc, check, report)
        if provenance is not None:
            _provenance(provenance, config_doc, check, report)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, StopIteration) as exc:
        reasons.append(f"Malformed or missing evidence: {type(exc).__name__}")
    report["passed"] = not reasons
    report["accepted_live"] = report["passed"] and report["live_provenance_valid"] and report["frozen_config_valid"]
    return report


def _audit(events: list[dict], config_doc: dict, summary: dict, run_dir: Path,
           check: Any, report: dict) -> None:
    config, run_id = config_doc["config"], config_doc["run_id"]
    config_id = digest(canonical(config).encode())
    report.update(run_id=run_id, config_id=config_id)
    check(config_doc["config_id"] == config_id == summary["config_id"], "Configuration hash mismatch")
    check(summary["run_id"] == run_id, "Summary run ownership mismatch")
    check(summary["closed"] is True and summary["log_complete"] is True, "Run is not closed and log-complete")
    check(not (run_dir / "summary.pending").exists(), "Uncommitted summary.pending exists")
    check(not summary["pending_call_ids"] and not summary["unconfirmed_submission_ids"] and summary["error"] is None,
          "Summary contains pending calls, unconfirmed submissions, or an error")
    _config(config, check, report)
    check(sorted(s["seat"] for s in config["run"]["seats"]) == [0, 1, 2, 3], "Four configured seats required")
    check(events and events[-1]["seq"] == summary["last_persisted_seq"], "Summary sequence differs from complete log")
    by_id = {e["event_id"]: e for e in events}
    check(len(by_id) == len(events), "Duplicate event IDs")
    check([e["seq"] for e in events] == list(range(1, len(events) + 1)), "Event sequence gap or duplication")
    starts = [e for e in events if e["event_type"] == "run_started"]
    finishes = [e for e in events if e["event_type"] == "run_finished"]
    closes = [e for e in events if e["event_type"] == "run_closed"]
    check(len(starts) == len(finishes) == len(closes) == 1, "One start, finish, and close are required")
    if starts:
        start = starts[0]
        check(start["payload"]["config"] == config and start["payload"]["config_id"] == config_id,
              "run_started configuration differs from frozen config", start)
        environment = start["payload"]["environment"]
        hashes = environment["source_files_sha256"]
        expected = digest(json.dumps(hashes, sort_keys=True).encode())
        check(bool(hashes) and environment["source_sha256"] == expected, "Source manifest digest mismatch", start)
        check(all(isinstance(v, str) and len(v) == 64 for v in hashes.values()), "Invalid source hash entry", start)
        package = Path(__file__).resolve().parents[1] / "src" / "guandan_agent"
        current = {str(p.relative_to(package)): digest(p.read_bytes()) for p in sorted(package.rglob("*.py"))}
        report["source"] = {"recorded_sha256": environment["source_sha256"], "manifest_valid": expected == environment["source_sha256"],
                            "matches_current_source": current == hashes, "file_count": len(hashes)}
    if finishes:
        check(finishes[0]["payload"] == {"status": "completed", "error": None}, "Run did not finish completed", finishes[0])
    if closes:
        check(closes[0] == events[-1] and closes[0]["payload"] == {
            "log_complete": True, "pending_call_ids": [], "unconfirmed_submission_ids": []}, "Close record is incomplete or not final", closes[0])

    sessions: dict[str, dict] = {}
    last_frame: dict[str, dict] = {}
    rx = Counter()
    round_number = Counter()
    state_version = Counter()
    active: dict[str, str | None] = {}
    request_ordinals = Counter()
    requests: dict[str, dict] = {}
    executions: dict[str, list[dict]] = defaultdict(list)
    rounds: dict[tuple, list[dict]] = defaultdict(list)
    results: dict[str, list[dict]] = defaultdict(list)
    closed: dict[str, list[dict]] = defaultdict(list)
    completed: dict[str, list[dict]] = defaultdict(list)
    match_completed: dict[str, list[dict]] = defaultdict(list)
    tribute: dict[tuple, list[dict]] = defaultdict(list)
    coverage: dict[str, set] = defaultdict(set)
    coverage_evidence: dict[str, list[str]] = defaultdict(list)
    raw_requests: set[str] = set()
    types = Counter()

    for e in events:
        kind, payload, sid = e["event_type"], e["payload"], e["session_id"]
        types[kind] += 1
        check(e["run_id"] == run_id and e["event_id"] == f"{run_id}:e{e['seq']}", "Event run/ID mismatch", e)
        check(kind not in {"session_interrupted", "log_gap"}, f"Forbidden incomplete-run event {kind}", e)
        if kind == "session_started":
            check(sid not in sessions, "Session was started twice", e)
            sessions[sid] = e
            check(payload["config_id"] == config_id and payload["seat"] == e["seat"], "Session config/seat mismatch", e)
            check(payload["rules_version"] == config["platform"]["rules_version"] and payload["adapter_version"] == config["platform"]["adapter_version"], "Session version mismatch", e)
        elif kind == "platform_received":
            frame, raw = payload["event"], payload["event"]["raw_message"]
            check(frame["session_id"] == sid, "Platform session mismatch", e)
            rx[sid] += 1
            check(frame["rx_seq"] == rx[sid], "Platform receive sequence is not continuous", e)
            if sid in sessions:
                session = sessions[sid]
                check(e["match_id"] == session["match_id"] and e["seat"] == session["seat"] and frame["connection_epoch"] == session["payload"]["connection_epoch"], "Platform connection binding mismatch", e)
            check(frame["kind"] not in {"unknown", "join_failed"}, "Invalid platform frame", e)
            if raw.get("type") == "notify" and raw.get("stage") == "beginning":
                round_number[sid] += 1
                check(raw["myPos"] == e["seat"], "Beginning seat mismatch", e)
                if round_number[sid] > 1:
                    coverage["cross_round_sessions"].add(sid)
                    coverage_evidence["cross_round_sessions"].append(e["event_id"])
            if raw.get("type") == "act" and raw.get("stage") in {"play", "tribute", "back"}:
                raw_requests.add(e["event_id"])
            if frame["kind"] not in {"joined", "auxiliary_notice", "match_result", "match_closed"}:
                state_version[sid] += 1
                active[sid] = None
            last_frame[sid] = e
            if raw.get("type") == "notify":
                stage = raw["stage"]
                key = (e["match_id"], round_number[sid])
                if stage == "episodeOver":
                    rounds[key].append(e)
                elif stage == "gameResult":
                    results[e["match_id"]].append(e)
                elif stage == "gameOver":
                    closed[e["match_id"]].append(e)
                elif stage == "tribute":
                    tribute[(sid, round_number[sid])].append(e)
                if stage in {"tribute", "back", "anti-tribute"}:
                    coverage[stage].add(key)
                    coverage_evidence[stage].append(e["event_id"])
        elif kind == "request_opened":
            context = payload["context"]
            state, legal = context["state"], context["legal"]
            binding = state["binding"]
            request = binding["request_id"]
            check(request not in requests, "Duplicate request ID", e)
            frame_event = last_frame[sid]
            frame, raw = frame_event["payload"]["event"], frame_event["payload"]["event"]["raw_message"]
            request_ordinals[sid] += 1
            expected = {"run_id": run_id, "match_id": e["match_id"], "round_id": round_number[sid],
                        "session_id": sid, "seat": e["seat"], "connection_epoch": sessions[sid]["payload"]["connection_epoch"],
                        "request_id": f"{sid}:q{request_ordinals[sid]}", "state_version": state_version[sid]}
            check(binding == expected == legal["binding"], "Request binding differs from current connection/round/state", e)
            check(all(e[k] == binding[k] for k in ("run_id", "match_id", "round_id", "session_id", "seat", "request_id")), "Request envelope binding mismatch", e)
            check(frame["kind"] == "action_requested" and raw["type"] == "act" and state["stage"] == raw["stage"], "Request lacks matching original action frame", e)
            expected_legal = {"binding": binding, "raw_count": len(raw["actionList"]), "actions": legal_actions(raw["actionList"])}
            check(legal == expected_legal, "Complete legal set differs from original actionList", e)
            check(Counter(state["self_hand"]) == Counter(raw["handCards"]), "Request hand differs from original frame", e)
            check(state["facts_status"] == "trusted" and context["policy"] == report["seat_policies"].get(str(e["seat"])), "Request facts/policy invalid", e)
            check(context["received_at"] == frame["received_at"], "Request receive time mismatch", e)
            check(payload["final_budget_ms"] == config["decision"]["total_s"] * 1000, "Final budget differs from config", e)
            check(context["fallback_action_id"] in {a["action_id"] for a in legal["actions"]}, "Fallback outside legal set", e)
            if raw["stage"] == "back":
                check(context["platform_context"] == {"tribute_pos": raw["tributePos"], "tribute_card": raw["tribute"]}, "Back context differs from platform", e)
            else:
                check(context["platform_context"] is None, "Unexpected tribute context", e)
            for row in raw["actionList"]:
                check(row == PASS or not (Counter(row[2]) - Counter(raw["handCards"])), "Legal action consumes absent cards", e)
            if legal["raw_count"] > len(legal["actions"]):
                coverage["duplicate_action_requests"].add(request)
                coverage_evidence["duplicate_action_requests"].append(e["event_id"])
            requests[request] = {"event": e, "frame": frame_event, "context": context}
            active[sid] = request
        elif kind == "request_invalidated":
            if active.get(sid) == e["request_id"]:
                active[sid] = None
        elif kind == "proposal_ready":
            proposal = payload["proposal"]
            request = requests[e["request_id"]]
            check(proposal["binding"] == request["context"]["state"]["binding"], "Proposal binding mismatch", e)
            check(proposal["source"] in _sources(request["context"]["policy"]), "Proposal source violates seat policy", e)
        elif kind in {"submission_reserved", "write_started", "send_finished", "execution_updated"}:
            execution = payload["execution"]
            request = requests[execution["binding"]["request_id"]]
            context = request["context"]
            check(execution["binding"] == context["state"]["binding"], "Execution binding mismatch", e)
            check(all(e[k] == execution["binding"][k] for k in ("run_id", "match_id", "round_id", "session_id", "seat", "request_id")), "Execution envelope ownership mismatch", e)
            action = next((a for a in context["legal"]["actions"] if a["action_id"] == execution["action_id"]), None)
            check(action is not None and action["wire_action"] == execution["wire_action"], "Execution action is outside request legal set", e)
            check(execution["source"] in _sources(context["policy"]), "Execution source violates seat policy", e)
            check(execution["status"] not in {"unknown", "cancelled_before_write"} and execution["error"] is None, "Submission has unknown/cancelled/error state", e)
            if kind == "write_started":
                check(active.get(sid) == e["request_id"], "Write used a stale/inactivated request", e)
                elapsed = (instant(execution["write_started_at"]) - instant(context["received_at"])).total_seconds() * 1000
                check(0 <= elapsed < request["event"]["payload"]["final_budget_ms"], "Write started outside final deadline", e)
            executions[execution["submission_id"]].append(e)
        elif kind in {"round_result", "match_result"}:
            evidence = by_id[payload["evidence_event_id"]]
            raw = evidence["payload"]["event"]["raw_message"]
            stage = "episodeOver" if kind == "round_result" else "gameResult"
            check(evidence["event_type"] == "platform_received" and raw["type"] == "notify" and raw["stage"] == stage,
                  "Settlement lacks original notify", e)
            check(all(evidence[k] == e[k] for k in ("match_id", "round_id", "session_id", "seat")) and evidence["seq"] < e["seq"], "Settlement evidence ownership/order mismatch", e)
            check(payload["result"] == {k: v for k, v in raw.items() if k not in {"type", "stage"}}, "Settlement differs from raw notify", e)
        elif kind == "session_completed":
            completed[e["match_id"]].append(e)
        elif kind == "match_completed":
            match_completed[e["match_id"]].append(e)

    # Verify each final execution from raw acceptance evidence and exact lineage.
    chains, stages, sources, request_submissions = [], Counter(), Counter(), Counter()
    for submission, history in executions.items():
        final = history[-1]["payload"]["execution"]
        request_id = final["binding"]["request_id"]
        request_submissions[request_id] += 1
        request = requests[request_id]
        context, frame_event = request["context"], request["frame"]
        sid, stage = final["binding"]["session_id"], context["state"]["stage"]
        counts = Counter(e["event_type"] for e in history)
        check(counts["submission_reserved"] == counts["write_started"] == counts["send_finished"] == 1, f"{submission}: expected one reserve/write/send completion")
        check(final["status"] == "confirmed", f"{submission}: final status is not confirmed")
        check(submission == f"{request_id}:s1", f"{submission}: invalid submission ID")
        check(final["source"] != "unique" or len(context["legal"]["actions"]) == 1,
              f"{submission}: unique source requires a singleton complete legal set")
        writes = [e for e in history if e["event_type"] == "write_started"]
        if not writes or not final["feedback_event_id"]:
            check(False, f"{submission}: missing write or confirmation evidence")
            continue
        write = writes[0]
        evidence = by_id.get(final["feedback_event_id"])
        if evidence is None or evidence["event_type"] != "platform_received":
            check(False, f"{submission}: missing original confirmation frame")
            continue
        frame, raw = evidence["payload"]["event"], evidence["payload"]["event"]["raw_message"]
        check(all(evidence[k] == final["binding"][k] for k in ("run_id", "match_id", "round_id", "session_id", "seat")), "Confirmation evidence has wrong ownership", evidence)
        check(frame["connection_epoch"] == final["binding"]["connection_epoch"] and frame["rx_seq"] > frame_event["payload"]["event"]["rx_seq"], "Confirmation connection/receive order mismatch", evidence)
        check(write["seq"] < evidence["seq"] <= history[-1]["seq"], "Confirmation event sequence precedes write", evidence)
        check(instant(final["reserved_at"]) <= instant(final["write_started_at"]) <= instant(frame["received_at"])
              and final["confirmed_at"] == frame["received_at"], "Confirmation time order mismatch", evidence)
        check(raw.get("type") == "notify" and raw.get("stage") == stage, "Wrong acceptance notify stage", evidence)
        action, seat = final["wire_action"], final["binding"]["seat"]
        if stage == "play":
            check(raw.get("curPos") == seat and same_action(action, raw["curAction"]), "PLAY acceptance does not match action multiset/seat", evidence)
        elif stage == "tribute":
            check(any(row[0] == seat and row[2] == action[2][0] for row in raw["result"]), "Tribute acceptance mismatch", evidence)
        else:
            ctx = context["platform_context"]
            pair = [ctx["tribute_pos"], seat, ctx["tribute_card"]]
            announcements = tribute[(sid, final["binding"]["round_id"])]
            check(any(a["seq"] < request["event"]["seq"] and pair in a["payload"]["event"]["raw_message"]["result"] for a in announcements), "Back lacks the preceding tribute pairing", evidence)
            check([seat, ctx["tribute_pos"], action[2][0]] in raw["result"], "Back sender/recipient/card mismatch", evidence)
        stable = ("submission_id", "binding", "action_id", "wire_action", "source", "reserved_at")
        check(all(all(e["payload"]["execution"][key] == final[key] for key in stable) for e in history), f"{submission}: execution identity changed")
        check(all(e["payload"].get("result", {}).get("status", "sent") == "sent" for e in history), f"{submission}: send did not complete sent")
        stages[stage] += 1
        sources[final["source"]] += 1
        chains.append({"request_id": request_id, "submission_id": submission, "stage": stage, "source": final["source"],
                       "request_event_id": request["event"]["event_id"], "raw_request_event_id": frame_event["event_id"],
                       "write_event_id": write["event_id"], "confirmation_event_id": evidence["event_id"]})
    check(set(request_submissions) == set(requests) and all(n == 1 for n in request_submissions.values()), "Every request must have exactly one submission")
    used_frames = [r["frame"]["event_id"] for r in requests.values()]
    check(set(used_frames) == raw_requests and len(used_frames) == len(raw_requests),
          "Every original act frame must create exactly one request")
    check(types["platform_received"] > 0 and requests, "No platform/action evidence")

    match_rows = []
    match_ids = {e["match_id"] for e in events if e["event_type"] == "match_started"}
    check(len(match_ids) == config["run"]["matches"] == types["match_started"], "Match count differs from frozen config")
    for match in sorted(match_ids):
        members = [e for e in sessions.values() if e["match_id"] == match]
        check(sorted(e["seat"] for e in members) == [0, 1, 2, 3], f"{match}: expected four independent seat sessions")
        check(len({e["payload"]["room_id"] for e in members}) == 1, f"{match}: sessions joined different rooms")
        for name, rows in (("gameResult", results[match]), ("gameOver", closed[match]), ("session completion", completed[match])):
            check(sorted(e["seat"] for e in rows) == [0, 1, 2, 3], f"{match}: requires four matching {name} records")
        raw_results = [e["payload"]["event"]["raw_message"] for e in results[match]]
        raw_closes = [e["payload"]["event"]["raw_message"] for e in closed[match]]
        check(all(type(raw.get("victory")) is int and raw["victory"] in (0, 1) for raw in raw_results),
              f"{match}: final winner must be a valid team number")
        check(len({canonical(v) for v in raw_results}) == 1, f"{match}: conflicting gameResult")
        check(len({canonical(v) for v in raw_closes}) == 1, f"{match}: conflicting gameOver")
        check(len(match_completed[match]) == 1, f"{match}: expected one coordinator match completion")
        for member in members:
            sid = member["session_id"]
            result = next((e for e in results[match] if e["session_id"] == sid), None)
            close = next((e for e in closed[match] if e["session_id"] == sid), None)
            completion = next((e for e in completed[match] if e["session_id"] == sid), None)
            check(result and close and completion and result["seq"] < close["seq"] < completion["seq"], f"{sid}: invalid settlement/closure order")
            if result and completion:
                normalized = {k: v for k, v in result["payload"]["event"]["raw_message"].items() if k not in {"type", "stage"}}
                check(completion["payload"]["match_result"] == normalized, f"{sid}: completion result differs from raw result")
                if match_completed[match]:
                    check(match_completed[match][0]["payload"]["result"] == normalized, f"{match}: coordinator result differs from raw result")
        match_rounds = [(key, rows) for key, rows in rounds.items() if key[0] == match]
        check(bool(match_rounds), f"{match}: no episodeOver evidence")
        check(sorted(key[1] for key, _ in match_rounds) == list(range(1, len(match_rounds) + 1)), f"{match}: round number gap")
        for key, rows in match_rounds:
            check(sorted(e["seat"] for e in rows) == [0, 1, 2, 3], f"{key}: round result lacks four seats")
            check(len({canonical(e["payload"]["event"]["raw_message"]) for e in rows}) == 1, f"{key}: conflicting episodeOver")
            for row in rows:
                order = row["payload"]["event"]["raw_message"].get("order")
                check(isinstance(order, list) and len(order) == 4 and all(type(seat) is int for seat in order)
                      and set(order) == {0, 1, 2, 3}, f"{key}: round order must contain each seat once")
        match_rows.append({"match_id": match, "sessions": {str(e["seat"]): e["session_id"] for e in members},
                           "rounds": len(match_rounds), "result_event_ids": [e["event_id"] for e in results[match]],
                           "closed_event_ids": [e["event_id"] for e in closed[match]],
                           "round_result_event_ids": [[e["event_id"] for e in rows] for _, rows in match_rounds],
                           "victory": raw_results[0]["victory"] if raw_results else None})
    report.update(counts={"events": len(events), "matches": len(match_ids), "sessions": len(sessions),
                          "rounds": len(rounds), "requests": len(requests), "submissions": len(executions),
                          "confirmed_chains": len(chains), "by_stage": dict(stages), "by_source": dict(sources),
                          "model_call_started": types["model_call_started"]},
                  coverage={name: len(coverage[name]) for name in ("tribute", "back", "anti-tribute", "cross_round_sessions", "duplicate_action_requests")},
                  coverage_evidence=dict(coverage_evidence), matches=match_rows, chains=chains,
                  limits=["Only the recorded client run is audited; no network packet capture or strategy-strength claim.",
                          "Deadline check uses persisted UTC timestamps; runtime monotonic enforcement is separately unit-tested.",
                          "Current source equality is reported separately so immutable historical records remain auditable."])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--frozen-plan", type=Path,
                        help="Explicit pre-run freeze; required with provenance for accepted_live")
    args = parser.parse_args(argv)
    report = verify_run(args.run_dir, provenance=args.provenance, frozen_plan=args.frozen_plan)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() in {(args.run_dir / name).resolve() for name in ("config.json", "events.jsonl", "summary.json", "summary.pending")}:
            parser.error("Report must not overwrite core run evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({key: report[key] for key in ("passed", "accepted_live", "frozen_config_valid", "run_dir", "reasons", "counts", "coverage") if key in report}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
