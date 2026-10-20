#!/usr/bin/env python3
"""Read a complete V0 run once and independently audit its logged wire evidence.

This is an acceptance check, not an evaluator. It does not contact the platform,
load credentials, or count a local send-complete marker as server acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

PASS = ["PASS", "PASS", "PASS"]
BINDING = ("run_id", "match_id", "round_id", "session_id", "seat", "connection_epoch", "request_id", "state_version")


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


def verify_run(run_dir: Path) -> dict:
    """Return a JSON-safe report, retaining specific failures for interrupted runs."""
    reasons: list[str] = []
    report: dict[str, Any] = {"schema_version": "1.0", "run_dir": str(run_dir.resolve()),
                              "passed": False, "reasons": reasons}

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
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        reasons.append(f"Malformed or missing evidence: {type(exc).__name__}: {exc}")
    report["passed"] = not reasons
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
    check(config["run"]["policy"] == "V0" and all(s["policy"] == "V0" for s in config["run"]["seats"]),
          "All policies must be V0")
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
    model_counters = {"slots_used": 0, "calls_started": 0, "transport_requests": 0}

    for e in events:
        kind, payload, sid = e["event_type"], e["payload"], e["session_id"]
        types[kind] += 1
        check(e["run_id"] == run_id and e["event_id"] == f"{run_id}:e{e['seq']}", "Event run/ID mismatch", e)
        check(kind not in {"session_interrupted", "log_gap"}, f"Forbidden incomplete-run event {kind}", e)
        check(not kind.startswith("model_call_"), "V0 emitted a model call event", e)
        for name in model_counters:
            if name in payload:
                check(payload[name] == 0, f"V0 has nonzero {name}", e)
                model_counters[name] += payload[name] or 0
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
            check(state["facts_status"] == "trusted" and context["policy"] == "V0", "Request facts/policy invalid", e)
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
            check(proposal["source"] in {"unique", "rule", "fallback"}, "V0 proposal has model source", e)
        elif kind in {"submission_reserved", "write_started", "send_finished", "execution_updated"}:
            execution = payload["execution"]
            request = requests[execution["binding"]["request_id"]]
            context = request["context"]
            check(execution["binding"] == context["state"]["binding"], "Execution binding mismatch", e)
            check(all(e[k] == execution["binding"][k] for k in ("run_id", "match_id", "round_id", "session_id", "seat", "request_id")), "Execution envelope ownership mismatch", e)
            action = next((a for a in context["legal"]["actions"] if a["action_id"] == execution["action_id"]), None)
            check(action is not None and action["wire_action"] == execution["wire_action"], "Execution action is outside request legal set", e)
            check(execution["source"] in {"unique", "rule", "fallback"}, "V0 execution has model source", e)
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
    check(types["model_call_started"] == types["model_call_finished"] == 0, "V0 model activity is not zero")

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
        match_rows.append({"match_id": match, "sessions": {str(e["seat"]): e["session_id"] for e in members},
                           "rounds": len(match_rounds), "result_event_ids": [e["event_id"] for e in results[match]],
                           "closed_event_ids": [e["event_id"] for e in closed[match]],
                           "round_result_event_ids": [[e["event_id"] for e in rows] for _, rows in match_rounds],
                           "victory": raw_results[0]["victory"] if raw_results else None})
    report.update(counts={"events": len(events), "matches": len(match_ids), "sessions": len(sessions),
                          "rounds": len(rounds), "requests": len(requests), "submissions": len(executions),
                          "confirmed_chains": len(chains), "by_stage": dict(stages), "by_source": dict(sources),
                          "model_call_started": types["model_call_started"], **model_counters},
                  coverage={name: len(coverage[name]) for name in ("tribute", "back", "anti-tribute", "cross_round_sessions", "duplicate_action_requests")},
                  coverage_evidence=dict(coverage_evidence), matches=match_rows, chains=chains,
                  limits=["Only the recorded client run is audited; no network packet capture or strategy-strength claim.",
                          "Deadline check uses persisted UTC timestamps; runtime monotonic enforcement is separately unit-tested.",
                          "Current source equality is reported separately so immutable historical records remain auditable."])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify_run(args.run_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({key: report[key] for key in ("passed", "run_dir", "reasons", "counts", "coverage") if key in report}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
