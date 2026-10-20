#!/usr/bin/env python3
"""Offline B06 audit: a bounded real main decision and raw platform acceptance.

The provenance file is the operator's execution declaration, not a packet capture.
A positive HTTP request count plus this declaration distinguishes the live run
from scripted evidence. This script never loads credentials or contacts services.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ENVELOPE = ("run_id", "match_id", "session_id", "seat", "round_id", "request_id")
PASS = ["PASS", "PASS", "PASS"]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _legal(rows: list) -> list[dict]:
    groups: dict[str, dict] = {}
    for index, row in enumerate(rows):
        key = _canonical(row)
        if key not in groups:
            groups[key] = {"action_id": f"L{index:04d}", "source_indices": [], "wire_action": row}
        groups[key]["source_indices"].append(index)
    return list(groups.values())


def _same_action(left: list, right: list) -> bool:
    return left == right if PASS in (left, right) else left[:2] == right[:2] and Counter(left[2]) == Counter(right[2])


def _secret_locations(value: Any, path: str = "") -> list[str]:
    """Heuristic only; no secret value is read, printed, or included in a report."""
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            location = f"{path}.{key}"
            if key.lower() in {"api_key", "apikey", "authorization", "deepseek_api_key"} and item:
                found.append(location)
            found.extend(_secret_locations(item, location))
        return found
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _secret_locations(item, f"{path}[{i}]")]
    if isinstance(value, str) and re.search(r"\bsk-[A-Za-z0-9_-]{20,}|\bBearer\s+[A-Za-z0-9_-]{20,}", value):
        return [path]
    return []


def verify_run(run_dir: Path, *, provenance: Path | None = None) -> dict:
    reasons: list[str] = []
    report: dict[str, Any] = {"schema_version": "1.0", "run_dir": str(run_dir.resolve()),
                              "passed": False, "chain_passed": False, "live_provenance_valid": False,
                              "reasons": reasons, "claims": {"full_match": False, "winner": None,
                              "strategy_strength": None, "formal_time_limit_compliance": None}}

    def check(condition: Any, message: str, event: dict | None = None) -> None:
        if not condition:
            reasons.append(f"{event['event_id']}: {message}" if event else message)

    try:
        raw = {name: (run_dir / name).read_bytes() for name in ("config.json", "events.jsonl", "summary.json")}
        report["files_sha256"] = {name: _sha(data) for name, data in raw.items()}
        config_doc, summary = json.loads(raw["config.json"]), json.loads(raw["summary.json"])
        events = [json.loads(line) for line in raw["events.jsonl"].splitlines()]
        locations = _secret_locations(config_doc, "config") + _secret_locations(summary, "summary") + _secret_locations(events, "events")
        report["secret_scan"] = {"method": "heuristic_without_loading_credentials", "locations": locations,
                                 "exact_secret_comparison": False}
        check(not locations, "Potential credential material present in core evidence")
        _audit(run_dir, config_doc, summary, events, check, report)
        if provenance is not None:
            evidence = provenance.read_bytes()
            declaration = json.loads(evidence)
            report["provenance_sha256"] = _sha(evidence)
            valid = (declaration.get("run_id") == config_doc["run_id"]
                     and declaration.get("provider") == "deepseek"
                     and declaration.get("platform") == "OpenGuanDanPlatform"
                     and declaration.get("network_mode") == "live"
                     and declaration.get("model") == config_doc["config"]["model"]["name"])
            report["live_provenance_valid"] = valid
            check(valid, "Live DeepSeek/platform execution declaration is missing or mismatched")
            check(not _secret_locations(declaration, "provenance"), "Potential credential material in provenance")
        else:
            check(False, "Live execution provenance is required; scripted evidence alone cannot pass B06")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, StopIteration) as exc:
        # Never echo untrusted malformed input, paths, or raw exception text.
        reasons.append(f"Malformed or missing evidence: {type(exc).__name__}")
    report["passed"] = not reasons
    return report


def _audit(run_dir: Path, config_doc: dict, summary: dict, events: list[dict], check: Any, report: dict) -> None:
    config, run_id = config_doc["config"], config_doc["run_id"]
    config_id = _sha(_canonical(config).encode())
    report.update(run_id=run_id, config_id=config_id)
    check(config_id == config_doc["config_id"] == summary["config_id"], "Configuration hash mismatch")
    check(summary["run_id"] == run_id, "Summary run ownership mismatch")
    check(summary["closed"] is True and summary["log_complete"] is True, "Run is not closed and log-complete")
    check(not (run_dir / "summary.pending").exists(), "Uncommitted summary.pending exists")
    check(not summary["pending_call_ids"], "Model calls remain pending at close")
    check(summary["error"] is None, "Recording summary contains an error")
    check(config["run"]["split"] == "development" and config["run"]["matches"] == 1, "B06 requires one development short run")
    check(sorted(s["seat"] for s in config["run"]["seats"]) == [0, 1, 2, 3], "Four configured seats required")
    check(any(s["policy"] == "V1" for s in config["run"]["seats"])
          and all(s["policy"] in {"V0", "V1"} for s in config["run"]["seats"]), "B06 requires V1 with no V2 seats")
    limit = config["batch"]["max_model_calls"]
    check(type(limit) is int and 0 < limit <= 4 and config["offline"]["max_model_calls"] == 0, "B06 model budget must be online <=4 and offline zero")
    check(config["model"]["max_retries"] == 0 and config["review"]["enabled"] is False, "B06 must disable model retries and review")
    check(events and summary["last_persisted_seq"] == events[-1]["seq"], "Summary sequence differs from complete log")
    check([e["seq"] for e in events] == list(range(1, len(events) + 1)), "Event sequence gap or duplication")
    by_id = {e["event_id"]: e for e in events}
    check(len(by_id) == len(events), "Duplicate event IDs")
    kinds: dict[str, list[dict]] = defaultdict(list)
    requests: dict[str, dict] = {}
    last_frames: dict[str, dict] = {}
    sessions: dict[str, dict] = {}
    rx, rounds, versions, ordinals = Counter(), Counter(), Counter(), Counter()
    histories: dict[str, list[dict]] = defaultdict(list)
    by_request: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    admitted: dict[str, dict] = {}
    started: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for event in events:
        kind, payload, sid = event["event_type"], event["payload"], event["session_id"]
        kinds[kind].append(event)
        check(event["run_id"] == run_id and event["event_id"] == f"{run_id}:e{event['seq']}", "Event run/ID mismatch", event)
        check(kind != "log_gap", "Log gap prevents acceptance", event)
        if event["request_id"] is not None:
            by_request[event["request_id"]][kind].append(event)
        if kind == "session_started":
            check(sid not in sessions, "Duplicate session start", event)
            sessions[sid] = event
        elif kind == "platform_received":
            frame = payload["event"]
            rx[sid] += 1
            check(frame["session_id"] == sid and frame["rx_seq"] == rx[sid], "Platform receive ownership/sequence mismatch", event)
            raw = frame["raw_message"]
            if raw.get("type") == "notify" and raw.get("stage") == "beginning":
                rounds[sid] += 1
            if frame["kind"] not in {"joined", "auxiliary_notice", "match_result", "match_closed"}:
                versions[sid] += 1
            last_frames[sid] = event
        elif kind == "request_opened":
            binding = payload["context"]["state"]["binding"]
            ordinals[sid] += 1
            expected = {"run_id": run_id, "match_id": sessions[sid]["match_id"],
                        "session_id": sid, "seat": sessions[sid]["seat"], "round_id": rounds[sid],
                        "connection_epoch": sessions[sid]["payload"]["connection_epoch"],
                        "request_id": f"{sid}:q{ordinals[sid]}", "state_version": versions[sid]}
            check(binding == expected, "Request binding differs from original connection/round/state", event)
            check(binding["request_id"] not in requests, "Duplicate request ID", event)
            requests[binding["request_id"]] = {"event": event, "raw_event": last_frames.get(sid), "context": payload["context"]}
        elif "execution" in payload:
            execution = payload["execution"]
            histories[execution["submission_id"]].append(event)
        elif kind in {"model_call_admitted", "model_call_started"}:
            target = admitted if kind == "model_call_admitted" else started
            call_id = payload["call_id"]
            check(call_id not in target, "Duplicate model admission/start", event)
            target[call_id] = event
            check(payload["role"] == "main" and payload["ordinal"] == 1
                  and call_id == f"{event['request_id']}:m1", "B06 allows one main attempt per request", event)
        elif kind == "model_call_finished":
            result = payload["result"]
            call_id = result["call_id"]
            check(call_id not in results, "Duplicate final model result", event)
            results[call_id] = dict(result)
        elif kind == "model_call_usage_updated":
            result = results.get(payload["call_id"])
            check(result is not None, "Usage update lacks a prior result", event)
            if result is not None:
                check(all(value is None or payload["usage"].get(key) == value for key, value in result["usage"].items()), "Usage update changed previously known telemetry", event)
                check(result["transport_requests"] is None or result["transport_requests"] == payload["transport_requests"], "Usage update changed known transport count", event)
                result["usage"], result["transport_requests"] = payload["usage"], payload["transport_requests"]

    for kind in ("run_started", "run_finished", "run_closed", "model_budget_summary"):
        check(len(kinds[kind]) == 1, f"Exactly one {kind} record required")
    if kinds["run_started"]:
        start = kinds["run_started"][0]
        check(start["payload"]["config"] == config and start["payload"]["config_id"] == config_id, "run_started differs from frozen configuration", start)
        environment = start["payload"]["environment"]
        manifest = environment["source_files_sha256"]
        check(bool(manifest) and environment["source_sha256"] == _sha(json.dumps(manifest, sort_keys=True).encode()), "Source manifest digest mismatch", start)
        check(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in manifest.values()), "Invalid source manifest entry", start)
        report["source"] = {"sha256": environment["source_sha256"], "files": len(manifest)}
    if kinds["run_finished"]:
        finish = kinds["run_finished"][0]
        check(finish["payload"]["status"] == "interrupted", "Short B06 run must record interrupted, not a game result", finish)
        report["stop_error_code"] = (finish["payload"]["error"] or {}).get("code")
    check(not any(kinds[k] for k in ("match_completed", "match_result", "session_completed")), "B06 short acceptance cannot claim a completed match")
    if kinds["run_closed"]:
        close = kinds["run_closed"][0]
        check(close == events[-1] and close["payload"]["log_complete"] is True, "Run close is incomplete or not final", close)
        check(close["payload"]["pending_call_ids"] == summary["pending_call_ids"]
              and close["payload"]["unconfirmed_submission_ids"] == summary["unconfirmed_submission_ids"], "Close/summary pending fields disagree", close)
    check(set(started) <= set(admitted) and set(results) <= set(admitted), "Model start/result lacks admission")
    check(set(started) <= set(results), "Started model call has no terminal result")
    check(1 <= len(started) <= len(admitted) <= limit <= 4, "Model admissions/starts exceed B06 budget or no call started")
    check(all(count == 1 for count in Counter(e["request_id"] for e in admitted.values()).values()), "More than one model admission per request")
    for index, event in enumerate(admitted.values(), 1):
        check(event["payload"]["run_slots_used"] == index and event["payload"]["slots_used"] == 1, "Admission budget counter mismatch", event)
    for index, event in enumerate(started.values(), 1):
        check(event["payload"]["run_calls_started"] == index and event["payload"]["calls_started"] == 1, "Started budget counter mismatch", event)
        admission = admitted[event["payload"]["call_id"]]
        check(admission["seq"] < event["seq"], "Call started before admission", event)
        check(all(admission[key] == event[key] for key in ENVELOPE), "Call start/admission ownership mismatch", event)
    if kinds["model_budget_summary"]:
        check(kinds["model_budget_summary"][0]["payload"] == {"slots_used": len(admitted), "calls_started": len(started)}, "Final model budget counters disagree with events")
    for call_id, result in results.items():
        admission = admitted[call_id]
        check(result["role"] == "main" and all(result["binding"][key] == admission[key] for key in ENVELOPE), "Result/admission binding mismatch")
        check(result["calls_started"] == int(call_id in started), "Result/start accounting mismatch")
        check(call_id in started or result["transport_requests"] in (None, 0), "Unstarted call reports a transport request")
        check(result["transport_requests"] is None or (type(result["transport_requests"]) is int and 0 <= result["transport_requests"] <= 1), "Provider transport exceeded one request per admitted call")
    unknown = [call_id for call_id in started if results.get(call_id, {}).get("transport_requests") is None]
    known_transport = sum(r["transport_requests"] for r in results.values() if r["transport_requests"] is not None)
    check(known_transport <= len(started) <= 4, "Observed transport count exceeds B06 budget")
    token_fields = ("input_tokens", "output_tokens", "total_tokens")
    usage = {key: sum(results[c]["usage"][key] for c in started) if all(results.get(c, {}).get("usage", {}).get(key) is not None for c in started) else None for key in token_fields}
    report["model"] = {"slots_used": len(admitted), "calls_started": len(started),
                       "transport_requests": None if unknown else known_transport,
                       "known_transport_requests": known_transport, "unknown_transport_call_ids": unknown,
                       "usage": usage, "calls": [{"call_id": c, "transport_requests": r["transport_requests"],
                       "usage": r["usage"], "error_code": (r["error"] or {}).get("code")} for c, r in results.items()]}
    report["unconfirmed_submission_ids"] = summary["unconfirmed_submission_ids"]
    report["tools"] = dict(Counter(e["payload"]["snapshot"]["tool_status"] for e in kinds["candidates_prepared"] if e["payload"].get("snapshot")))
    chains = []
    for submission_id, history in histories.items():
        final = history[-1]["payload"]["execution"]
        if final["source"] == "main" and final["status"] == "confirmed":
            before = len(report["reasons"])
            chain = _chain(final, history, requests, by_request, by_id, results, config, check)
            chain["passed"] = len(report["reasons"]) == before
            chains.append(chain)
    report["chains"] = chains
    report["chain_passed"] = any(c["passed"] for c in chains)
    check(report["chain_passed"], "No complete real main decision and raw confirmation chain")
    report["counts"] = {"events": len(events), "requests": len(requests), "confirmed_main_chains": sum(c["passed"] for c in chains),
                        "executions": len(histories), "execution_statuses": dict(Counter(h[-1]["payload"]["execution"]["status"] for h in histories.values()))}


def _chain(final: dict, history: list[dict], requests: dict, grouped: dict, by_id: dict, results: dict, config: dict, check: Any) -> dict:
    binding = final["binding"]
    request_id, submission_id = binding["request_id"], final["submission_id"]
    request = requests[request_id]
    event, raw_event, context = request["event"], request["raw_event"], request["context"]
    check(raw_event is not None, "Request has no original platform frame", event)
    frame = raw_event["payload"]["event"]
    raw = frame["raw_message"]
    check(binding == context["state"]["binding"] == context["legal"]["binding"], "Request/execution binding mismatch", event)
    check(all(event[key] == binding[key] for key in ENVELOPE), "Request envelope ownership mismatch", event)
    check(frame["session_id"] == binding["session_id"] and frame["connection_epoch"] == binding["connection_epoch"], "Raw request connection mismatch", raw_event)
    check(frame["kind"] == "action_requested" and raw["type"] == "act" and raw["stage"] == context["state"]["stage"], "Request lacks original act frame", raw_event)
    check(context["policy"] == "V1" and context["state"]["facts_status"] == "trusted", "Main request must use trusted V1 state", event)
    check(context["legal"]["raw_count"] == len(raw["actionList"]) and context["legal"]["actions"] == _legal(raw["actionList"]), "Complete legal set differs from original actionList", event)
    check(Counter(context["state"]["self_hand"]) == Counter(raw["handCards"]), "Request hand differs from original frame", event)
    rows = {row["action_id"]: row for row in context["legal"]["actions"]}
    action = rows.get(final["action_id"])
    check(action is not None and action["wire_action"] == final["wire_action"], "Execution action is outside original legal set", history[-1])
    nodes = {}
    for kind in ("candidates_prepared", "model_call_admitted", "model_call_started", "model_call_finished", "main_registered"):
        entries = grouped[request_id][kind]
        check(len(entries) == 1, f"Main chain requires exactly one {kind}", event)
        nodes[kind] = entries[0]
        check(all(nodes[kind][key] == binding[key] for key in ENVELOPE), f"{kind} envelope binding mismatch", nodes[kind])
    snapshot = nodes["candidates_prepared"]["payload"]["snapshot"]
    snapshot_id = f"{request_id}:c1"
    candidate_ids = [c["action_id"] for c in snapshot["candidates"]]
    check(snapshot["binding"] == binding and snapshot["candidate_snapshot_id"] == snapshot_id, "Candidate snapshot binding mismatch")
    check(len(candidate_ids) == len(set(candidate_ids)) and 0 < len(candidate_ids) <= config["tools"]["candidate_hard_limit"]
          and set(candidate_ids) <= set(rows) and snapshot["rule_action_id"] in candidate_ids, "Candidate set is not a unique bounded legal subset")
    main = nodes["main_registered"]["payload"]["main"]
    result = nodes["model_call_finished"]["payload"]["result"]
    call_id = f"{request_id}:m1"
    check(main["binding"] == result["binding"] == binding and main["candidate_snapshot_id"] == result["candidate_snapshot_id"] == snapshot_id, "Main/result snapshot binding mismatch")
    check(main["call_id"] == result["call_id"] == nodes["model_call_admitted"]["payload"]["call_id"] == nodes["model_call_started"]["payload"]["call_id"] == call_id, "Main call identity mismatch")
    check(result["role"] == "main" and result["calls_started"] == 1 and result["error"] is None
          and result["decision"] == main["decision"] and nodes["model_call_finished"]["payload"]["late"] is False,
          "Main lacks valid successful result before suggestion deadline")
    check(results[call_id]["transport_requests"] == 1, "Main success requires exactly one actual provider transport request")
    check(main["decision"]["action_id"] == final["action_id"] and final["action_id"] in candidate_ids, "Submitted action differs from registered candidate decision")
    check(set(main["decision"]) == {"action_id", "reason"} and isinstance(main["decision"]["reason"], str) and 1 <= len(main["decision"]["reason"].strip()) <= 240, "Main decision schema invalid")
    ordered = [raw_event["seq"], event["seq"]] + [nodes[k]["seq"] for k in nodes]
    check(ordered == sorted(set(ordered)), "Main evidence ordering is invalid")
    counts = Counter(e["event_type"] for e in history)
    check(counts["submission_reserved"] == counts["write_started"] == counts["send_finished"] == 1, "Main submission requires one reserve/write/send completion")
    write = next(e for e in history if e["event_type"] == "write_started")
    check(submission_id == f"{request_id}:s1" and nodes["main_registered"]["seq"] < history[0]["seq"] <= write["seq"], "Main submission identity/order mismatch")
    stable = ("binding", "action_id", "wire_action", "source", "submission_id", "reserved_at")
    check(all(all(e["payload"]["execution"][key] == final[key] for key in stable) and all(e[key] == binding[key] for key in ENVELOPE) for e in history), "Execution identity/binding changed")
    check(not any(event["seq"] < e["seq"] < write["seq"] for e in grouped[request_id]["request_invalidated"]), "Write used an invalidated request")
    check(all(e["payload"].get("result", {}).get("status", "sent") == "sent" for e in history), "Submission send was not completed")
    check(0 <= (_instant(final["write_started_at"]) - _instant(context["received_at"])).total_seconds() * 1000 < event["payload"]["final_budget_ms"], "Write was outside configured local deadline")
    evidence = by_id.get(final["feedback_event_id"])
    check(evidence is not None and evidence["event_type"] == "platform_received", "Missing original confirmation evidence")
    accepted = evidence["payload"]["event"]
    notification = accepted["raw_message"]
    check(all(evidence[key] == binding[key] for key in ENVELOPE if key != "request_id") and accepted["session_id"] == binding["session_id"] and accepted["connection_epoch"] == binding["connection_epoch"], "Confirmation ownership/connection mismatch", evidence)
    check(write["seq"] < evidence["seq"] <= history[-1]["seq"] and accepted["rx_seq"] > frame["rx_seq"], "Confirmation precedes write/request", evidence)
    check(final["confirmed_at"] == accepted["received_at"] and _instant(final["reserved_at"]) <= _instant(final["write_started_at"]) <= _instant(accepted["received_at"]), "Confirmation timestamp mismatch", evidence)
    stage, seat = context["state"]["stage"], binding["seat"]
    check(stage in {"play", "tribute", "back"}, "Unknown action stage")
    check(notification.get("type") == "notify" and notification.get("stage") == stage, "Confirmation is not matching original notify", evidence)
    if stage == "play":
        check(notification.get("curPos") == seat and _same_action(final["wire_action"], notification["curAction"]), "Raw PLAY acceptance does not match action/seat", evidence)
    elif stage == "tribute":
        check(any(row[0] == seat and row[2] == final["wire_action"][2][0] for row in notification["result"]), "Raw tribute acceptance mismatch", evidence)
    elif stage == "back":
        ctx = context["platform_context"]
        check(ctx == {"tribute_pos": raw["tributePos"], "tribute_card": raw["tribute"]}, "Back context differs from original request")
        check([seat, ctx["tribute_pos"], final["wire_action"][2][0]] in notification["result"], "Raw back sender/recipient/card mismatch", evidence)
    return {"request_id": request_id, "submission_id": submission_id, "call_id": call_id,
            "candidate_snapshot_id": snapshot_id, "stage": stage, "action_id": final["action_id"],
            "wire_action": final["wire_action"], "legal_count": len(rows), "candidate_ids": candidate_ids,
            "raw_request_event_id": raw_event["event_id"], "request_event_id": event["event_id"],
            **{f"{kind}_event_id": e["event_id"] for kind, e in nodes.items()},
            "write_event_id": write["event_id"], "confirmation_event_id": evidence["event_id"],
            "final_execution_event_id": history[-1]["event_id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify_run(args.run_dir, provenance=args.provenance)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.resolve() in {(args.run_dir / name).resolve() for name in ("config.json", "events.jsonl", "summary.json", "summary.pending")}:
            parser.error("Report must not overwrite core run evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
