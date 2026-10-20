"""P21 accounting fixtures use protocol-shaped source events, never providers."""

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

from guandan_agent.config import RunConfig
from guandan_agent.contracts import LogEvent, RunSummary
from guandan_agent.offline.evaluate import evaluate, win_rate
from guandan_agent.recording import ClosedRunLog, HistoryUnavailable, RecordingClosedError


STAMP = "2026-10-19T00:00:00Z"


class Recording:
    """Small deterministic immutable source builder, not a game simulator."""

    def __init__(self, number=1, *, team=0, policy="V2", complete=True):
        self.run = str(UUID(int=number))
        self.match = str(UUID(int=1000 + number))
        self.sessions = [str(UUID(int=10000 + number * 4 + seat)) for seat in range(4)]
        self.config = RunConfig.model_validate({"run": {"policy": policy, "evaluation_team": team}})
        self.complete = complete
        self.events = []
        self.pending = []
        self.unconfirmed = []
        self.add("match_started", {})

    def add(self, kind, payload, *, seat=None, round_id=None, request=None):
        seq = len(self.events) + 1
        event = LogEvent.model_validate({
            "schema_version": "1.0", "run_id": self.run, "match_id": self.match,
            "session_id": self.sessions[seat] if seat is not None else None,
            "seat": seat, "round_id": round_id, "request_id": request,
            "event_type": kind, "payload": payload, "seq": seq,
            "event_id": f"{self.run}:e{seq}", "recorded_at": STAMP,
        })
        self.events.append(event)
        return event.event_id

    def settle(self, victory=0, *, seats=range(4), round_result=False):
        kind = "round_result" if round_result else "match_result"
        result = ({"order": [0, 1, 2, 3], "curRank": "3", "restCards": []} if round_result else
                  {"victory": victory, "victoryRank": ["A", "2"], "victoryNum": [1, 0, 0, 0],
                   "gameResult": [1, 0, 0, 0], "gameRound": 1})
        for seat in seats:
            event = self.add("platform_received", {"event": {"kind": kind, "payload": result}}, seat=seat, round_id=1)
            self.add(kind, {"result": result, "evidence_event_id": event}, seat=seat, round_id=1)

    def request(self, number, *, unique=False, seat=0):
        binding = {"run_id": self.run, "match_id": self.match, "round_id": 1,
                   "session_id": self.sessions[seat], "seat": seat, "connection_epoch": 1,
                   "request_id": f"{self.sessions[seat]}:q{number}", "state_version": number}
        actions = [{"action_id": "L0000", "source_indices": [0], "wire_action": ["Single", "8", ["S8"]]}]
        if not unique:
            actions.append({"action_id": "L0001", "source_indices": [1], "wire_action": ["Single", "9", ["S9"]]})
        context = {"state": {"binding": binding, "stage": "play", "self_hand": ["S8", "S9"],
                    "level_rank": "2", "remaining_counts": [2, 2, 2, 2], "team_ranks": ["2", "2"],
                    "turn_seat": seat, "leader_seat": None, "leading_action": None, "finished_order": [],
                    "history": [], "history_complete": True, "history_omitted": 0, "facts_status": "trusted"},
                   "legal": {"binding": binding, "raw_count": len(actions), "actions": actions},
                   "received_at": STAMP, "fallback_action_id": "L0000",
                   "policy": self.config.run.policy, "platform_context": None}
        self.req_event(binding, "request_opened", {"context": context})
        return binding

    def req_event(self, binding, kind, payload):
        return self.add(kind, payload, seat=binding["seat"], round_id=1, request=binding["request_id"])

    def execute(self, binding, source, *, status="confirmed", write=True, action="L0000"):
        value = {"submission_id": binding["request_id"] + ":s1", "binding": binding,
                 "action_id": action, "wire_action": ["Single", "8" if action == "L0000" else "9", ["S8" if action == "L0000" else "S9"]],
                 "source": source, "status": status, "reserved_at": "2026-10-19T00:00:00.050Z",
                 "write_started_at": "2026-10-19T00:00:00.100Z" if write else None,
                 "sent_at": "2026-10-19T00:00:00.110Z" if write else None,
                 "confirmed_at": "2026-10-19T00:00:00.150Z" if status == "confirmed" else None,
                 "feedback_event_id": None, "error": None}
        self.req_event(binding, "execution_updated", {"execution": value})
        return value

    def call(self, binding, *, role="main", ordinal=1, start=True, decision=True, error=None,
             usage=None, transport=1, finish=True):
        call_id = binding["request_id"] + f":m{ordinal}"
        body = {"call_id": call_id, "role": role, "ordinal": ordinal}
        self.req_event(binding, "model_call_admitted", body)
        if start:
            self.req_event(binding, "model_call_started", body)
        if finish:
            result = {"binding": binding, "candidate_snapshot_id": binding["request_id"] + ":c1",
                      "role": role, "call_id": call_id, "decision": {"action_id": "L0000", "reason": "fixture"} if decision else None,
                      "error": {"code": error, "component": "model", "safe_message": "fixture", "related_event_id": None} if error else None,
                      "elapsed_ms": 10.0, "queue_wait_ms": 2.0, "calls_started": int(start),
                      "transport_requests": transport, "usage": usage or {}}
            self.req_event(binding, "model_call_finished", {"result": result})
        return call_id

    def review(self, binding, *, kind="review", status="changed", call_id=None):
        self.req_event(binding, "review_finished", {"review": {"kind": kind, "status": status, "call_id": call_id}})

    def close(self):
        return ClosedRunLog(run_id=self.run, config_id=self.config.config_id,
                            events=tuple(self.events), config=self.config, evicted_through=0,
                            max_query_limit=200, summary=RunSummary(
                                schema_version="1.0", run_id=self.run, config_id=self.config.config_id,
                                closed=True, log_complete=self.complete, last_persisted_seq=len(self.events),
                                pending_call_ids=tuple(self.pending), unconfirmed_submission_ids=tuple(self.unconfirmed), error=None))


def report(records, **kwargs):
    return evaluate(records, report_id="fixed-evaluation", created_at=STAMP, **kwargs).model_dump(mode="json")


def t09_records():
    rows = []
    for number in range(1, 11):
        source = Recording(number, complete=number != 1)
        if number <= 8:
            source.settle(number % 2)
        elif number == 9:
            source.add("run_finished", {"status": "interrupted"})
        rows.append(source.close())
    return rows


def test_t09_counts_deduplicate_four_seats_and_gap_overlaps_settled():
    records = t09_records()
    result = report(records)
    assert {key: result["counts"][key] for key in ("started", "settled", "win", "loss", "interrupted", "pending", "log_incomplete_count")} == {
        "started": 10, "settled": 8, "win": 4, "loss": 4, "interrupted": 1, "pending": 1, "log_incomplete_count": 1}
    assert result["performance"]["win_rate"] == {"numerator": 4, "denominator": 8, "value": .5}
    assert result == report(list(reversed(records)) + [records[0]])
    assert result["status"] == "partial"
    assert result["model_usage"]["online"]["total_tokens"]["value"] is None


def test_draw_requires_explicit_normalized_outcome_not_missing_winner():
    assert win_rate(["win", "loss", "draw"]) == {"numerator": 1, "denominator": 3, "value": 1 / 3}
    assert win_rate([])["value"] is None
    with pytest.raises(ValueError):
        win_rate(["unknown"])
    source = Recording()
    source.settle(None)
    result = report([source.close()])
    assert result["counts"]["draw"] == 0 and result["counts"]["pending"] == 1


def test_settlement_conflict_or_broken_reference_stays_pending_after_interruption():
    source = Recording()
    source.settle(0, seats=[0])
    source.settle(1, seats=[1])
    source.add("run_finished", {"status": "interrupted"})
    result = report([source.close()])
    assert result["counts"]["settlement_conflict_count"] == 1
    assert result["counts"]["settled"] == 0 and result["counts"]["pending"] == 1
    source = Recording()
    source.add("match_result", {"result": {"victory": 0}, "evidence_event_id": "missing"}, seat=0, round_id=1)
    result = report([source.close()])
    assert result["counts"]["unverified_settlement_count"] == 1
    assert result["performance"]["win_rate"]["denominator"] == 0


def test_evaluation_team_and_mixed_conditions_preserve_source_configuration():
    a, b = Recording(team=0), Recording(2, team=1, policy="V0")
    a.settle(0)
    b.settle(0)
    result = report([a.close(), b.close()])
    assert result["counts"]["win"] == result["counts"]["loss"] == 1
    assert result["cohort"]["mixed_conditions"]
    assert [x["evaluation_team"] for x in result["cohort"]["sources"]] == [0, 1]
    assert not result["cohort"]["paired_verified"]


def test_latency_unknowns_and_final_adopted_fallback_denominator_includes_tools():
    source = Recording()
    fallback = source.request(1)
    main = source.request(2)
    preparing = source.request(3)
    unique = source.request(4, unique=True)
    source.execute(fallback, "fallback")
    source.execute(main, "main", status="cancelled_before_write", write=False)
    source.execute(unique, "unique")
    source.req_event(main, "proposal_ready", {"proposal": {"source": "fallback"}})  # late, not adopted
    source.req_event(fallback, "candidates_prepared", {"elapsed_ms": 7.0})
    source.req_event(preparing, "request_invalidated", {"cause_code": "FINAL_DEADLINE"})
    result = report([source.close()])
    quality, latency = result["quality"], result["latency"]
    assert quality["fallback_rate"] == {"numerator": 1, "denominator": 3, "value": 1 / 3}
    assert quality["nonunique_without_adoption_count"] == 1
    assert quality["submission_deadline_unwritten_count"] == 1
    assert latency["request_to_first_write"] == {"n": 2, "p50": 100.0, "p95": 100.0, "max": 100.0, "missing_count": 0}
    assert latency["requests_without_write"] == 2
    assert latency["tools"]["n"] == 1 and latency["tools"]["missing_count"] == 2


def test_usage_late_update_counted_once_pending_unknown_and_v0_zero():
    source = Recording()
    first, second = source.request(1), source.request(2)
    call = source.call(first, usage={"input_tokens": 10}, transport=None)
    for _ in range(2):
        source.req_event(first, "model_call_usage_updated", {"call_id": call, "transport_requests": 1,
            "usage": {"input_tokens": 999, "output_tokens": 2, "total_tokens": 12}})
    pending = source.call(second, finish=False)
    source.pending.append(pending)
    result = report([source.close()])["model_usage"]["online"]
    assert result["slots_used"]["value"] == result["calls_started"]["value"] == 2
    assert result["input_tokens"] == {"value": None, "known_subtotal": 10, "missing_call_count": 1}
    assert result["total_tokens"] == {"value": None, "known_subtotal": 12, "missing_call_count": 1}
    assert result["cost_amount"]["value"] is None and result["cost_amount"]["missing_call_count"] == 2
    assert result["transport_requests"]["known_subtotal"] == 1
    v0 = report([Recording(policy="V0").close()])["model_usage"]["online"]
    assert v0["total_tokens"]["value"] == v0["cost_amount"]["value"] == 0


def test_content_invalid_denominator_excludes_timeout_and_review_repair_starts_are_distinct():
    source = Recording()
    changed, repaired, timeout, skipped_main, skipped_repair = [source.request(n) for n in range(1, 6)]
    source.call(changed)
    review_id = source.call(changed, role="review", ordinal=2)
    source.review(changed, call_id=review_id)
    source.execute(changed, "review")
    source.call(repaired, decision=False, error="MODEL_SCHEMA")
    repair_id = source.call(repaired, role="repair", ordinal=2)
    source.review(repaired, kind="repair", status="repaired", call_id=repair_id)
    source.execute(repaired, "repair")
    source.call(timeout, decision=False, error="MODEL_TIMEOUT", transport=None)
    source.execute(timeout, "fallback")
    source.call(skipped_main)
    source.call(skipped_main, role="review", ordinal=2, start=False, finish=False)
    source.review(skipped_main, status="skipped")
    source.execute(skipped_main, "main")
    source.call(skipped_repair, decision=False, error="MODEL_ACTION_UNKNOWN")
    source.review(skipped_repair, kind="repair", status="skipped")
    source.execute(skipped_repair, "fallback")
    quality = report([source.close()])["quality"]
    assert quality["main_invalid_rate"] == {"numerator": 2, "denominator": 4, "value": .5}
    assert quality["review_start_rate"] == {"numerator": 1, "denominator": 2, "value": .5}
    assert quality["review_success_rate"]["value"] == quality["review_change_rate"]["value"] == 1
    assert quality["repair_count"] == quality["repair_success_count"] == 1
    assert quality["fallback_rate"]["numerator"] == 2
    assert quality["model_timeout_count"] == 1
    assert quality["over_budget_request_count"] == 0


def test_unknown_usage_is_not_reconstructed_from_prompt_completion_or_mixed_currencies():
    source = Recording()
    for num, currency in [(1, "USD"), (2, "CNY")]:
        source.call(source.request(num), usage={"input_tokens": 10, "output_tokens": 2,
                                               "cost_amount": 1.0, "currency": currency, "cost_basis": "provider"})
    usage = report([source.close()])["model_usage"]["online"]
    assert usage["total_tokens"]["value"] is None
    assert usage["total_tokens"]["known_subtotal"] == 0
    assert usage["cost_amount"]["value"] is None and usage["cost_amount"]["known_subtotal"] is None
    assert usage["known_cost_by_currency"] == {"CNY": 1.0, "USD": 1.0}


def test_invalidated_risky_main_still_counts_review_opportunity_and_unidentified_calls_stay_unknown():
    source = Recording()
    request = source.request(1)
    source.call(request)
    source.req_event(request, "main_registered", {"main": {"trigger_codes": ["BOMB"]}})
    source.req_event(request, "request_invalidated", {"cause_code": "STOPPED"})
    source.add("model_budget_summary", {"slots_used": 2, "calls_started": 2})
    result = report([source.close()])
    assert result["quality"]["review_start_rate"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert result["model_usage"]["online"]["unidentified_started_calls"] == 1
    assert result["model_usage"]["online"]["total_tokens"]["missing_call_count"] == 2


def test_cost_without_currency_or_basis_has_no_total():
    source = Recording()
    source.call(source.request(1), usage={"cost_amount": 1.0})
    usage = report([source.close()])["model_usage"]["online"]
    assert usage["cost_amount"]["value"] is None
    assert usage["cost_amount"]["known_subtotal"] is None
    assert usage["cost_metadata_missing_call_count"] == 1


def test_round_results_no_invented_numeric_score_or_victory():
    source = Recording()
    for seat in range(4):
        source.add("platform_received", {"event": {"kind": "round_started", "payload": {}}}, seat=seat)
    source.settle(round_result=True)
    result = report([source.close()], unit="round")
    assert result["counts"]["started"] == result["counts"]["settled"] == 1
    assert result["performance"]["win_rate"]["value"] is None
    assert result["performance"]["mean_round_score"] is None
    assert result["performance"]["round_results"][0]["team_finish_order"] == [0, 1, 0, 1]


def test_illegal_unconfirmed_and_three_call_violation_are_exposed_without_execution():
    source = Recording()
    request = source.request(1)
    for ordinal in (1, 2, 3):
        source.call(request, ordinal=ordinal)
    execution = source.execute(request, "fallback", status="unknown", action="L9999")
    source.unconfirmed.append(execution["submission_id"])
    quality = report([source.close()])["quality"]
    assert quality["over_budget_request_count"] == 1
    assert quality["unconfirmed_submission_count"] == 1
    assert quality["illegal_submission_rate"]["value"] == 1


def test_unclosed_evicted_conflicting_or_missing_config_sources_are_rejected():
    source = Recording().close()
    source.summary = source.summary.model_copy(update={"closed": False})
    with pytest.raises(RecordingClosedError):
        report([source])
    source = Recording().close()
    source._evicted_through = 1
    with pytest.raises(HistoryUnavailable):
        report([source])
    a, b = Recording().close(), Recording(complete=False).close()
    with pytest.raises(ValueError):
        report([a, b])
    source = Recording().close()
    source.config = None
    with pytest.raises(ValueError):
        report([source])


def test_existing_frozen_scripted_archive_is_read_once_and_counts_two_matches():
    source = asyncio.run(ClosedRunLog.load(Path(__file__).parents[1] / "fixtures/audit/valid_scripted_run"))
    result = report([source])
    assert result["counts"]["started"] == result["counts"]["settled"] == 2
    assert result["counts"]["win"] == 2
    assert result["latency"]["request_to_first_write"]["n"] == 6
    assert result["quality"]["illegal_submission_rate"]["numerator"] == 0
    assert result["model_usage"]["online"]["total_tokens"]["value"] == 0
