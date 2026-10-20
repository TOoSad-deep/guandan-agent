"""Deterministic statistics over already loaded, closed recording evidence.

This module performs no I/O. Missing events are never reconstructed from config
budgets, and an incomplete log can establish a result without establishing the
total number of calls or exact latency distribution. UTC latency is observational
and is not a monotonic-clock or formal platform time-limit guarantee.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import math
from typing import Any

from guandan_agent.contracts import (
    EvaluationReport, ExecutionRecord, ModelCallResult, RequestContext,
)
from guandan_agent.recording import ClosedRunLog


def win_rate(outcomes: Sequence[str]) -> dict[str, int | float | None]:
    """Aggregate explicit normalized outcomes, including draws in the denominator.

    The current platform parser supplies only decisive match outcomes. A draw
    must be explicitly established by an upstream protocol before this helper
    can receive it; absence of a winner is never a draw.
    """
    if any(outcome not in {"win", "loss", "draw"} for outcome in outcomes):
        raise ValueError("Only explicit settled outcomes can enter the win-rate denominator")
    return _rate(outcomes.count("win"), len(outcomes))


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _delta(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    try:
        value = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _distribution(values: Sequence[float], missing: int = 0) -> dict[str, Any]:
    ordered = sorted(values)

    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {"n": len(ordered), "p50": quantile(.5), "p95": quantile(.95),
            "max": ordered[-1] if ordered else None, "missing_count": missing}


def _owned(value: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    return all(value.get(key) == event.get(key) for key in (
        "run_id", "match_id", "session_id", "seat", "round_id", "request_id"))


def _trusted_result(event: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Require the Session's accepted result and its same-session source frame."""
    payload = event["payload"]
    evidence = by_id.get(payload.get("evidence_event_id"))
    if evidence is None or evidence["event_type"] != "platform_received" or evidence["seq"] >= event["seq"]:
        return None
    if any(evidence[key] != event[key] for key in ("run_id", "match_id", "session_id", "seat", "round_id")):
        return None
    wire = evidence["payload"].get("event", {})
    result = payload.get("result")
    if wire.get("kind") != event["event_type"] or wire.get("payload") != result or not isinstance(result, dict):
        return None
    return result


def _cohort(record: ClosedRunLog) -> dict[str, Any]:
    config = record.config
    if config is None or config.config_id != record.config_id:
        raise ValueError("Evaluation requires the matching frozen source configuration")
    cfg = config.model_dump(mode="json")
    team = cfg["run"]["evaluation_team"]
    seats = cfg["run"]["seats"]
    return {
        "run_id": record.run_id, "config_id": record.config_id,
        "evaluation_team": team, "seats": seats,
        "teammates": [seat for seat in seats if seat["seat"] % 2 == team],
        "opponents": [seat for seat in seats if seat["seat"] % 2 != team],
        "uncontrolled_seats": [seat for seat in range(4) if seat not in {item["seat"] for item in seats}],
        "platform": cfg["platform"], "decision": cfg["decision"], "tools": cfg["tools"],
        "model": cfg["model"], "review": cfg["review"], "budget": cfg["batch"],
        "versions": cfg["versions"], "split": cfg["run"]["split"],
        "experiment_id": cfg["run"]["experiment_id"],
        # The official snapshot has no validated whole-match paired-deal event.
        "deal_group_id": None, "paired_verified": False,
    }


def _units(record: ClosedRunLog, rows: list[dict[str, Any]], unit: str) -> list[dict[str, Any]]:
    by_id = {event["event_id"]: event for event in rows}
    units: dict[tuple[str, int | None], dict[str, Any]] = {}
    results: dict[tuple[str, int | None], list[tuple[dict[str, Any], str]]] = defaultdict(list)
    interrupted: set[str] = set()
    invalid_result: set[tuple[str, int | None]] = set()
    assert record.config is not None
    team = record.config.run.evaluation_team
    for row in rows:
        match = row["match_id"]
        if match is None:
            continue
        round_id = row["round_id"] if unit == "round" else None
        wire = row["payload"].get("event", {}) if row["event_type"] == "platform_received" else {}
        if unit == "round" and wire.get("kind") == "round_started":
            # The receive envelope is logged before TableState increments round.
            round_id = (row["round_id"] or 0) + 1
        if unit == "round" and round_id is None:
            continue
        key = match, round_id
        units.setdefault(key, {"run_id": record.run_id, "match_id": match, "round_id": round_id,
                               "evaluation_team": team, "log_complete": record.health.log_complete})
        if row["event_type"] in {"session_interrupted", "run_finished"}:
            if row["event_type"] == "session_interrupted" or row["payload"].get("status") == "interrupted":
                interrupted.add(match)
        if row["event_type"] == f"{unit}_result":
            result = _trusted_result(row, by_id)
            if result is not None:
                results[key].append((result, row["event_id"]))
            else:
                invalid_result.add(key)
    output = []
    for key, sample in sorted(units.items()):
        settlements = results[key]
        distinct = {_json(result) for result, _ in settlements}
        sample["evidence_event_ids"] = sorted({event_id for _, event_id in settlements})
        sample["settlement_conflict"] = len(distinct) > 1
        sample["unverified_settlement"] = key in invalid_result
        result = settlements[0][0] if len(distinct) == 1 and key not in invalid_result else None
        outcome = None
        if result is not None and unit == "match":
            victory = result.get("victory")
            if type(victory) is int and victory in (0, 1):
                outcome = "win" if victory == team else "loss"
                sample["platform_result"] = result
        elif result is not None:
            order = result.get("order")
            if isinstance(order, list) and len(order) == 4 and set(order) == {0, 1, 2, 3} and all(type(seat) is int for seat in order):
                sample["finish_order"] = order
                sample["team_finish_order"] = [seat % 2 for seat in order]
                # No numeric small-round score is defined by this application.
                sample["platform_rank"] = result.get("curRank")
                sample["platform_rank_changes"] = result.get("rank_changes")
                outcome = "settled"
        if outcome is not None:
            sample["status"] = "settled"
        elif sample["settlement_conflict"] or sample["unverified_settlement"]:
            sample["status"] = "pending"
        else:
            sample["status"] = "interrupted" if key[0] in interrupted else "pending"
        sample["outcome"] = outcome
        output.append(sample)
    return output


def evaluate(records: Sequence[ClosedRunLog], *, report_id: str, created_at: str,
             unit: str = "match") -> EvaluationReport:
    """Compute an immutable report; caller owns loading and report persistence.

    Duplicate references to the same frozen run are harmless. Conflicting
    records with the same run ID, unclosed logs and evicted windows are rejected.
    All measurements are observed samples; global totals are null on log gaps.
    """
    if unit not in {"match", "round"}:
        raise ValueError("Evaluation unit must be match or round")
    if not records:
        raise ValueError("Evaluation requires at least one closed run")
    sources: dict[str, tuple[ClosedRunLog, list[dict[str, Any]]]] = {}
    for record in records:
        rows = [event.model_dump(mode="json") for event in record.offline_events()]
        if record.run_id in sources:
            previous, previous_rows = sources[record.run_id]
            if (previous.config_id != record.config_id or previous.summary != record.summary or previous_rows != rows):
                raise ValueError("Conflicting copies of one source run")
        else:
            sources[record.run_id] = record, rows
    ordered = [sources[key] for key in sorted(sources)]
    cohorts = [_cohort(record) for record, _ in ordered]
    complete = all(record.health.log_complete for record, _ in ordered)
    units = [sample for record, rows in ordered for sample in _units(record, rows, unit)]
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    executions: dict[tuple[str, str], dict[str, Any]] = {}
    calls: dict[tuple[str, str], dict[str, Any]] = {}
    reviews: dict[tuple[str, str], dict[str, Any]] = {}
    review_eligible: set[tuple[str, str]] = set()
    proposals: dict[tuple[str, str], dict[str, Any]] = {}
    invalidated: dict[tuple[str, str], str] = {}
    tool_results: dict[tuple[str, str], float | None] = {}
    integrity_errors = 0
    observed_call_slots = 0
    observed_call_starts = 0
    unknown_call_count = 0
    pending_calls: set[tuple[str, str]] = set()
    summary_unconfirmed: set[tuple[str, str]] = set()
    for record, rows in ordered:
        run_id = record.run_id
        admitted, started = set(), set()
        summaries = []
        for row in rows:
            kind, payload = row["event_type"], row["payload"]
            request_key = run_id, row["request_id"]
            if kind == "request_opened":
                try:
                    context = RequestContext.model_validate(payload["context"])
                    if not _owned(context.state.binding.model_dump(), row):
                        raise ValueError("ownership")
                    value = context.model_dump(mode="json")
                    if request_key in requests and requests[request_key] != value:
                        raise ValueError("conflict")
                    requests[request_key] = value
                except (KeyError, ValueError, TypeError):
                    integrity_errors += 1
            elif "execution" in payload:
                try:
                    execution = ExecutionRecord.model_validate(payload["execution"])
                    if not _owned(execution.binding.model_dump(), row):
                        raise ValueError("ownership")
                    executions[run_id, execution.submission_id] = execution.model_dump(mode="json")
                except (KeyError, ValueError, TypeError):
                    integrity_errors += 1
            if kind == "proposal_ready":
                proposals[request_key] = payload.get("proposal", {})
            elif kind == "review_finished":
                reviews[request_key] = payload.get("review", {})
                if payload.get("review", {}).get("kind") == "review":
                    review_eligible.add(request_key)
            elif kind == "main_registered" and payload.get("main", {}).get("trigger_codes"):
                # A risk-triggered main can lose its request before a final
                # proposal/review_finished exists. It still was an opportunity.
                if requests.get(request_key, {}).get("policy") == "V2":
                    review_eligible.add(request_key)
            elif kind == "request_invalidated":
                invalidated[request_key] = payload.get("cause_code", "")
            elif kind in {"candidates_prepared", "tools_failed"}:
                value = payload.get("elapsed_ms")
                tool_results[request_key] = value if _number(value) else None
            elif kind in {"model_call_admitted", "model_call_started"}:
                call_id = payload.get("call_id")
                if isinstance(call_id, str):
                    call = calls.setdefault((run_id, call_id), {"role": payload.get("role"),
                        "request_key": request_key, "usage": {}, "transport_requests": None})
                    call["role"] = payload.get("role")
                    call[kind.removeprefix("model_call_")] = True
                    (admitted if kind == "model_call_admitted" else started).add(call_id)
            elif kind == "model_call_finished":
                try:
                    result = ModelCallResult.model_validate(payload["result"])
                    if not _owned(result.binding.model_dump(), row):
                        raise ValueError("ownership")
                    if result.call_id is not None:
                        call = calls.setdefault((run_id, result.call_id), {"request_key": request_key})
                        call.update(result.model_dump(mode="json"))
                        call["finished"] = True
                        if result.calls_started:
                            started.add(result.call_id)
                            call["started"] = True
                except (KeyError, ValueError, TypeError):
                    integrity_errors += 1
            elif kind == "model_call_usage_updated":
                call = calls.get((run_id, payload.get("call_id")))
                if call is not None:
                    for name, value in payload.get("usage", {}).items():
                        if call.setdefault("usage", {}).get(name) is None and value is not None:
                            call["usage"][name] = value
                    if call.get("transport_requests") is None:
                        call["transport_requests"] = payload.get("transport_requests")
            elif kind == "model_budget_summary":
                summaries.append(payload)
        pending_calls.update((run_id, key) for key in record.summary.pending_call_ids)
        summary_unconfirmed.update((run_id, key) for key in record.summary.unconfirmed_submission_ids)
        for call_id in record.summary.pending_call_ids:
            call = calls.setdefault((run_id, call_id), {"usage": {}, "transport_requests": None})
            # Pending means begin_call succeeded but a final result is absent.
            call["started"] = True
            started.add(call_id)
        slots = max([len(admitted)] + [row["slots_used"] for row in summaries if type(row.get("slots_used")) is int])
        starts = max([len(started)] + [row["calls_started"] for row in summaries if type(row.get("calls_started")) is int])
        observed_call_slots += max(slots, starts)
        observed_call_starts += starts
        unknown_call_count += starts - len(started)

    by_request: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (run_id, _), execution in executions.items():
        by_request[run_id, execution["binding"]["request_id"]].append(execution)
    nonunique = {key for key, value in requests.items() if len(value["legal"]["actions"]) > 1}
    accepted = {key: values[0] for key, values in by_request.items() if len(values) == 1}
    fallback = {key for key in nonunique if key in accepted and accepted[key]["source"] == "fallback"}
    unadopted = nonunique - set(accepted)
    write_latency, decision_latency = [], []
    no_write, missing_write, missing_decision = 0, 0, 0
    illegal, legality_missing, written, unconfirmed = set(), set(), set(), set(summary_unconfirmed)
    for key, context in requests.items():
        execution = accepted.get(key)
        if execution is None or execution["write_started_at"] is None:
            no_write += 1
        else:
            elapsed = _delta(context["received_at"], execution["write_started_at"])
            if elapsed is None:
                missing_write += 1
            else:
                write_latency.append(elapsed)
        elapsed = _delta(context["received_at"], execution["reserved_at"] if execution else None)
        if elapsed is None:
            missing_decision += 1
        else:
            decision_latency.append(elapsed)
    for key, execution in executions.items():
        if execution["status"] in {"writing", "sent", "unknown"}:
            unconfirmed.add(key)
        if execution["write_started_at"] is not None:
            written.add(key)
            context = requests.get((key[0], execution["binding"]["request_id"]))
            if context is None:
                legality_missing.add(key)
            else:
                action = next((action for action in context["legal"]["actions"] if action["action_id"] == execution["action_id"]), None)
                if action is None or action["wire_action"] != execution["wire_action"]:
                    illegal.add(key)

    started_calls = [call for call in calls.values() if call.get("started")]
    main_content = [call for call in started_calls if call.get("role") == "main" and call.get("finished")
                    and (call.get("decision") is not None or (call.get("error") or {}).get("code") in {"MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN"})]
    invalid_main = [call for call in main_content if call.get("decision") is None]
    review_started = [call for call in started_calls if call.get("role") == "review"]
    repair_started = [call for call in started_calls if call.get("role") == "repair"]
    review_success, review_changed = 0, 0
    for key, review in reviews.items():
        call = calls.get((key[0], review.get("call_id")))
        if (key in accepted and review.get("kind") == "review" and review.get("status") in {"kept", "changed"}
                and call is not None and call.get("started") and call.get("role") == "review"
                and call.get("decision") is not None and call.get("error") is None
                and accepted[key]["source"] == "review"):
            review_success += 1
            review_changed += review["status"] == "changed"
    # A started review that loses its request before adoption is still a start.
    review_start_requests = {call["request_key"] for call in review_started if "request_key" in call}
    eligible_reviews = len(review_eligible | review_start_requests)
    per_request_calls = Counter(call.get("request_key") for call in calls.values() if call.get("admitted") or call.get("started"))
    over_budget = sum(count > {"V0": 0, "V1": 1, "V2": 2}[requests[key]["policy"]]
                      for key, count in per_request_calls.items() if key in requests)
    model_timeouts = sum((call.get("error") or {}).get("code") in {"MODEL_TIMEOUT", "TIMEOUT"} for call in started_calls)
    model_durations = [call["elapsed_ms"] for call in started_calls if _number(call.get("elapsed_ms"))]
    tool_durations = [value for value in tool_results.values() if value is not None]
    # Every new candidate-policy nonunique request enters preparation, even if
    # it is invalidated while the task is still running and never logs a result.
    tools_expected = {key for key in nonunique if sources[key[0]][0].config.versions.policy != "b02-cheap-v1"}

    def usage_metric(name: str, *, transport: bool = False) -> dict[str, Any]:
        values = [(call.get(name) if transport else call.get("usage", {}).get(name)) for call in started_calls]
        known = [value for value in values if _number(value)]
        missing = len(values) - len(known) + unknown_call_count
        return {"value": sum(known) if missing == 0 and complete else None,
                "known_subtotal": sum(known), "missing_call_count": missing}

    usage = {name: usage_metric(name) for name in ("input_tokens", "output_tokens", "total_tokens", "cost_amount")}
    currencies = sorted({call.get("usage", {}).get("currency") for call in started_calls if call.get("usage", {}).get("currency")})
    bases = sorted({call.get("usage", {}).get("cost_basis") for call in started_calls if call.get("usage", {}).get("cost_basis")})
    known_cost_by_currency: dict[str, float] = defaultdict(float)
    for call in started_calls:
        raw = call.get("usage", {})
        if _number(raw.get("cost_amount")) and raw.get("currency"):
            known_cost_by_currency[raw["currency"]] += raw["cost_amount"]
    missing_cost_metadata = sum(_number(call.get("usage", {}).get("cost_amount"))
                                and (not call.get("usage", {}).get("currency") or not call.get("usage", {}).get("cost_basis"))
                                for call in started_calls)
    if missing_cost_metadata:
        usage["cost_amount"]["value"] = None
    if len(currencies) > 1 or any(_number(call.get("usage", {}).get("cost_amount")) and not call.get("usage", {}).get("currency") for call in started_calls):
        usage["cost_amount"]["value"] = None
        usage["cost_amount"]["known_subtotal"] = None
    outcomes = [sample["outcome"] for sample in units if sample["outcome"] in {"win", "loss", "draw"}]
    counts = Counter(sample["status"] for sample in units)
    conflicts = sum(sample["settlement_conflict"] for sample in units)
    unverified = sum(sample["unverified_settlement"] for sample in units)
    conditions = [{key: value for key, value in cohort.items() if key not in {"run_id", "config_id", "experiment_id"}} for cohort in cohorts]
    for condition in conditions:
        for key in ("seats", "teammates", "opponents"):
            condition[key] = [{"seat": item["seat"], "policy": item["policy"]} for item in condition[key]]
    mixed = len({_json(value) for value in conditions}) > 1
    quality = {
        "fallback_rate": _rate(len(fallback), len(nonunique)),
        "nonunique_without_adoption_count": len(unadopted),
        "main_invalid_rate": _rate(len(invalid_main), len(main_content)),
        "main_content_basis": "valid decision or MODEL_SCHEMA/MODEL_ACTION_UNKNOWN; transport errors excluded",
        "review_start_rate": _rate(len(review_start_requests), eligible_reviews),
        "review_calls_started": len(review_started),
        "review_success_rate": _rate(review_success, len(review_started)),
        "review_change_rate": _rate(review_changed, review_success),
        "repair_count": len(repair_started),
        "repair_success_count": sum(key in accepted and accepted[key]["source"] == "repair" and value.get("kind") == "repair" and value.get("status") == "repaired" for key, value in reviews.items()),
        "unconfirmed_submission_count": len(unconfirmed),
        "over_budget_request_count": over_budget,
        "duplicate_submission_request_count": sum(len(value) > 1 for value in by_request.values()),
        "illegal_submission_rate": _rate(len(illegal), len(written - legality_missing)),
        "submission_legality_missing_count": len(legality_missing),
        "model_timeout_count": model_timeouts,
        "submission_deadline_unwritten_count": sum(cause == "FINAL_DEADLINE" and (key not in accepted or accepted[key]["write_started_at"] is None) for key, cause in invalidated.items()),
        "budget_degraded_request_count": sum(key in fallback and proposals.get(key, {}).get("cause_code") in {"BATCH_BUDGET", "CALL_BUDGET"} for key in nonunique),
        "recording_gap_run_count": sum(not record.health.log_complete for record, _ in ordered),
        "invalid_evidence_count": integrity_errors,
        "metrics_complete": complete and integrity_errors == 0,
        "denominator_basis": "trusted request_opened includes local fallback construction and candidate preparation; adopted ExecutionRecord source is authoritative",
    }
    return EvaluationReport(
        report_id=report_id, created_at=created_at, unit=unit,
        run_ids=tuple(record.run_id for record, _ in ordered),
        config_ids=tuple(record.config_id for record, _ in ordered),
        status="partial" if not complete or integrity_errors or conflicts or unverified or counts["pending"] or counts["interrupted"] else "completed",
        cohort={"sources": cohorts, "mixed_conditions": mixed,
                "aggregation": "descriptive_mixed_conditions" if mixed else "same_conditions",
                "paired_verified": False, "deal_group_id": None,
                "pairing_note": "No verified whole-match deal pairing exists in the current platform logs"},
        counts={"started": len(units), "settled": counts["settled"], "win": outcomes.count("win"),
                "loss": outcomes.count("loss"), "draw": outcomes.count("draw"),
                "interrupted": counts["interrupted"], "pending": counts["pending"],
                "log_incomplete_count": sum(not sample["log_complete"] for sample in units),
                "settlement_conflict_count": conflicts, "unverified_settlement_count": unverified,
                "sample_unit": unit, "units": units},
        performance={"win_rate": win_rate(outcomes), "mean_round_score": None,
                     "explicit_platform_losses": [{"run_id": sample["run_id"], "match_id": sample["match_id"],
                         "reason": sample.get("platform_result", {}).get("reason")} for sample in units if sample["outcome"] == "loss"],
                     "round_results": [sample for sample in units if unit == "round" and sample["status"] == "settled"],
                     "draw_evidence": "Current official protocol has victory=0/1 only; missing winners are pending"},
        latency={"unit": "ms", "quantile_method": "linear interpolation at (n-1)*q",
                 "request_to_first_write": _distribution(write_latency, missing_write),
                 "decision_to_acceptance": _distribution(decision_latency, missing_decision),
                 "model": _distribution(model_durations, observed_call_starts - len(model_durations)),
                 "tools": _distribution(tool_durations, len(tools_expected | set(tool_results)) - len(tool_durations)),
                 "requests_without_write": no_write, "metrics_complete": complete,
                 "clock_basis": "recorded UTC request receipt / first write; model and tool elapsed_ms when recorded",
                 "boundary": "Observed samples only; log gaps omit unknown events and small-sample quantiles are not time-limit guarantees"},
        model_usage={"online": {"slots_used": {"value": observed_call_slots if complete else None, "known_subtotal": observed_call_slots},
                        "calls_started": {"value": observed_call_starts if complete else None, "known_subtotal": observed_call_starts},
                        "transport_requests": usage_metric("transport_requests", transport=True), **usage,
                        "currency": currencies[0] if len(currencies) == 1 else None,
                        "cost_bases": bases, "known_cost_by_currency": dict(sorted(known_cost_by_currency.items())),
                        "cost_metadata_missing_call_count": missing_cost_metadata,
                        "inventory_complete": complete, "unidentified_started_calls": unknown_call_count,
                        "pending_call_count": len(pending_calls)},
                     "offline": {"included": False, "reason": "Offline review usage remains in its separate report/task record"}},
        quality=quality,
    )
