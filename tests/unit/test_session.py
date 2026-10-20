"""Synchronous authority and async boundaries with parsed frames and virtual time."""

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from guandan_agent.clock import ManualClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    AppendResult, ConnectionLost, DecisionInput, DecisionOutcome, DecisionProposal, ErrorInfo,
    ProposalReady, SeatConnection, SendFinished, SendResult, SubmissionDeadline,
)
from guandan_agent.platform import ScriptedPlatform, parse_platform_frame
from guandan_agent.runtime.scheduling import ManualScheduler
from guandan_agent.runtime.session import Session
from guandan_agent.strategy import DecisionPolicy

RUN = "00000000-0000-4000-8000-000000000051"
MATCH = "00000000-0000-4000-8000-000000000052"
SEAT_SESSION = "00000000-0000-4000-8000-000000000053"


class MemoryRecorder:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail
        self.health = SimpleNamespace(log_complete=not fail)

    def append(self, event):
        if self.fail:
            raise OSError("synthetic recording failure")
        self.events.append(event)
        seq = len(self.events)
        return AppendResult(accepted=True, seq=seq, event_id=f"{RUN}:e{seq}", error=None)


class ControlledPlatform:
    def __init__(self, *, wait_gate=False, wait_finish=False):
        self.before_gate = asyncio.Event()
        self.after_gate = asyncio.Event()
        if not wait_gate:
            self.before_gate.set()
        if not wait_finish:
            self.after_gate.set()
        self.written = []
        self.intents = []
        self.closed = False

    async def write(self, intent, gate):
        self.intents.append(intent)
        await self.before_gate.wait()
        decision = gate.begin_write(intent)
        if decision.status == "deny":
            return SendResult(submission_id=intent.submission_id, status="not_started", error=decision.error)
        self.written.append(intent)
        await self.after_gate.wait()
        return SendResult(submission_id=intent.submission_id, status="sent", error=None)

    async def close(self):
        self.closed = True


class DelayedPolicy:
    def __init__(self, *, ignore_cancel=False):
        self.ready = asyncio.Event()
        self.ignore_cancel = ignore_cancel
        self.inputs = []

    async def decide(self, input):
        self.inputs.append(input)
        while not self.ready.is_set():
            try:
                await self.ready.wait()
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
        return await DecisionPolicy().decide(input)


@dataclass
class Harness:
    session: Session
    clock: ManualClock
    scheduler: ManualScheduler
    recorder: MemoryRecorder
    platform: ControlledPlatform
    seq: int = 0

    def post(self, raw):
        self.seq += 1
        event = parse_platform_frame(
            raw, config=self.session.connection, room_id=1, rx_seq=self.seq,
            received_at=self.clock.utc_now(), received_mono=self.clock.monotonic(),
        )
        assert event.kind != "unknown", event.payload
        self.session.post(event)
        return event

    async def tick(self):
        # Advance cooperative tasks without sleeping on wall-clock time.
        for _ in range(4):
            await asyncio.sleep(0)


def begin(hand=("S2", "S2", "H4"), level="2"):
    return {"type": "notify", "stage": "beginning", "myPos": 0,
            "handCards": list(hand), "levelCard": level}


def request(hand=("S2", "S2", "H4"), *, stage="play", actions=None, counts=None,
            context=None, leader=None, tribute=(1, "SA"), level="2"):
    raw = {"type": "act", "stage": stage, "handCards": list(hand),
           "curRank": level, "selfRank": level, "oppoRank": level,
           "actionList": actions or ([["Single", hand[0][1], [hand[0]]]] if stage == "play"
                                      else [[stage, stage, [hand[0]]]])}
    if stage == "play":
        raw.update(publicInfo=[{"rest": n} for n in (counts or [len(hand), 27, 27, 27])],
                   curPos=-1 if context is None else context[0],
                   curAction=None if context is None else context[1],
                   greaterPos=-1 if leader is None else leader[0],
                   greaterAction=None if leader is None else leader[1])
    if stage == "back":
        raw.update(tributePos=tribute[0], tribute=tribute[1])
    return raw


def play(actor=0, action=None):
    return {"type": "notify", "stage": "play", "curPos": actor,
            "curAction": action or ["Single", "2", ["S2"]],
            "greaterPos": -1, "greaterAction": None}


def settled(level="2"):
    return {"type": "notify", "stage": "episodeOver", "curRank": level,
            "order": [0, 1, 2, 3], "restCards": [[3, ["HR", "SA"]]]}


def result():
    return {"type": "notify", "stage": "gameResult", "victory": 0,
            "victoryRank": ["A", "8"], "victoryNum": [2, 0, 0, 1],
            "gameResult": [1, 0, 1, 0], "gameRound": 1}


def closed():
    return {"type": "notify", "stage": "gameOver", "curTimes": 1,
            "settingTimes": 1, "gameRound": 1}


def harness(*, platform=None, policy=None, hand=("S2", "S2", "H4"), fail_log=False):
    clock = ManualClock(100.0)
    scheduler = ManualScheduler(clock)
    recorder = MemoryRecorder(fail_log)
    platform = platform or ControlledPlatform()
    connection = SeatConnection(
        url="ws://127.0.0.1:18181", mode="create", room_id=None, user_name="session-test",
        seat=0, session_id=SEAT_SESSION, connection_epoch=1, server_rounds=1,
    )
    session = Session(run_id=RUN, match_id=MATCH, connection=connection,
                      config=RunConfig(), platform=platform, recorder=recorder,
                      clock=clock, scheduler=scheduler, policy=policy)
    h = Harness(session, clock, scheduler, recorder, platform)
    h.post({"type": "CREATE_ROOM", "code": 200, "data": {"roomId": 1, "userNum": 0}})
    h.post(begin(hand))
    return h


def test_t01_exact_duplicate_shortcut_writes_original_once_and_waits_for_feedback():
    async def scenario():
        h = harness()
        action = ["Single", "2", ["S2"]]
        h.post(request(actions=[action, action]))
        assert h.session.active_context.legal.raw_count == 2
        assert len(h.session.active_context.legal.actions) == 1
        await h.tick()
        assert len(h.platform.written) == 1
        record = h.session.execution_records[0]
        assert record.source == "unique" and record.status == "sent"
        assert h.session.view().state.self_hand == ("S2", "S2", "H4")
        event = h.post(play())
        record = h.session.execution_records[0]
        assert record.status == "confirmed" and record.feedback_event_id is not None
        assert h.session.view().state.self_hand == ("S2", "H4")
        h.session.post(event)
        assert h.session.view().state.self_hand == ("S2", "H4")
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_t05_stop_applies_before_waiting_transport_reaches_gate():
    async def scenario():
        h = harness(platform=ControlledPlatform(wait_gate=True))
        h.post(request())
        await h.tick()
        intent = h.platform.intents[0]
        assert h.session.execution_records[0].status == "reserved"
        h.session.stop("user stop")
        assert h.session.active_context is None
        assert h.session.execution_records[0].status == "cancelled_before_write"
        assert h.session.begin_write(intent).status == "deny"
        h.platform.before_gate.set()
        await h.tick()
        assert not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())


def test_t05_public_progress_and_new_request_invalidate_old_binding_immediately():
    async def scenario():
        h = harness(platform=ControlledPlatform(wait_gate=True))
        h.post(request())
        await h.tick()
        old_intent = h.platform.intents[0]
        old_binding = old_intent.binding
        other = ["Single", "3", ["S3"]]
        h.post(play(1, other))
        h.post(request(counts=[3, 26, 27, 27], context=(1, other), leader=(1, other)))
        assert h.session.active_context.state.binding.request_id.endswith(":q2")
        assert h.session.begin_write(old_intent).status == "deny"
        h.session.post(SubmissionDeadline(binding=old_binding))
        assert h.session.active_context is not None
        h.platform.before_gate.set()
        await h.tick()
        assert len(h.platform.written) == 1
        assert h.platform.written[0].binding.request_id.endswith(":q2")
        await h.session.shutdown()
    asyncio.run(scenario())


def test_t04_suggestion_deadline_fallback_and_late_policy_cannot_send_twice():
    async def scenario():
        policy = DelayedPolicy()
        h = harness(policy=policy)
        h.post(request(actions=[["Single", "2", ["S2"]], ["Single", "4", ["H4"]]]))
        await h.tick()
        assert not h.session.execution_records
        h.scheduler.advance(29.0)
        await h.tick()
        assert len(h.platform.written) == 1
        assert h.session.execution_records[0].source == "fallback"
        policy.ready.set()
        await h.tick()
        h.scheduler.advance(1.0)
        assert h.session.active_context is None
        assert h.session.execution_records[0].status == "sent"
        h.post(play(action=h.platform.written[0].model_dump(mode="json")["wire_action"]))
        assert h.session.execution_records[0].status == "confirmed"
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_final_deadline_denies_reserved_transport_after_channel_wait():
    async def scenario():
        h = harness(platform=ControlledPlatform(wait_gate=True))
        h.post(request())
        await h.tick()
        h.scheduler.advance(30.0)
        h.platform.before_gate.set()
        await h.tick()
        assert h.session.execution_records[0].status == "cancelled_before_write"
        assert not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())


def test_confirmation_timeout_preserves_unknown_then_reliable_feedback_confirms_once():
    async def scenario():
        h = harness()
        h.post(request())
        await h.tick()
        h.scheduler.advance(5.0)
        assert h.session.execution_records[0].status == "unknown"
        h.post(play())
        assert h.session.execution_records[0].status == "confirmed"
        assert len(h.platform.written) == 1
        assert h.session.view().state.self_hand == ("S2", "H4")
        await h.session.shutdown()
    asyncio.run(scenario())


def test_confirmation_can_arrive_while_writing_and_sent_callback_never_regresses_it():
    async def scenario():
        h = harness(platform=ControlledPlatform(wait_finish=True))
        h.post(request())
        await h.tick()
        assert h.session.execution_records[0].status == "writing"
        h.post(play())
        assert h.session.execution_records[0].status == "confirmed"
        h.platform.after_gate.set()
        await h.tick()
        record = h.session.execution_records[0]
        assert record.status == "confirmed" and record.sent_at is not None
        h.scheduler.advance(30.0)
        assert h.session.execution_records[0].status == "confirmed"
        await h.session.shutdown()
    asyncio.run(scenario())


def test_t06_disconnect_while_writing_marks_unknown_late_same_connection_feedback_can_confirm():
    async def scenario():
        h = harness(platform=ControlledPlatform(wait_finish=True))
        h.post(request())
        await h.tick()
        h.session.post(ConnectionLost(run_id=RUN, session_id=SEAT_SESSION, reason="test EOF"))
        assert h.session.execution_records[0].status == "unknown"
        assert h.session.status == "interrupted"
        h.post(play())
        assert h.session.execution_records[0].status == "confirmed"
        await h.tick()
        assert h.session.execution_records[0].status == "confirmed"
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_t13_duplicate_proposal_forms_one_reservation_and_send_callback_is_idempotent():
    async def scenario():
        h = harness(policy=DelayedPolicy(), platform=ControlledPlatform(wait_gate=True))
        h.post(request())
        context = h.session.active_context
        outcome = await DecisionPolicy().decide(DecisionInput(context=context))
        event = ProposalReady(binding=context.state.binding, outcome=outcome)
        h.session.post(event)
        h.session.post(event)
        assert len(h.session.execution_records) == 1
        h.platform.before_gate.set()
        await h.tick()
        record = h.session.execution_records[0]
        version = h.session.view().view_version
        h.session.post(SendFinished(binding=record.binding,
            result=SendResult(submission_id=record.submission_id, status="sent", error=None)))
        assert h.session.view().view_version == version
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_malformed_policy_proposal_uses_prevalidated_fallback():
    class InvalidPolicy:
        async def decide(self, input):
            return DecisionOutcome(status="proposed", error=None, proposal=DecisionProposal(
                binding=input.context.state.binding, candidate_snapshot_id=None,
                action_id="L9999", source="rule", reason="invalid fixture", cause_code=None, review=None))

    async def scenario():
        h = harness(policy=InvalidPolicy())
        h.post(request())
        await h.tick()
        assert h.session.execution_records[0].source == "fallback"
        assert h.platform.written[0].action_id == "L0000"
        await h.session.shutdown()
    asyncio.run(scenario())


def test_recorder_failure_keeps_play_running_and_marks_evidence_incomplete():
    async def scenario():
        h = harness(fail_log=True)
        h.post(request())
        await h.tick()
        h.post(play())
        assert h.session.execution_records[0].status == "confirmed"
        assert h.session.status == "running"
        assert not h.session.view().log_complete
        await h.session.shutdown()
    asyncio.run(scenario())


def test_duplicate_request_without_fact_progress_interrupts_instead_of_refreshing_budget():
    async def scenario():
        h = harness(policy=DelayedPolicy())
        raw = request()
        h.post(raw)
        h.post(raw)
        assert h.session.status == "interrupted"
        assert h.session.view().state.facts_status == "degraded"
        assert h.session.active_context is None
        await h.tick()
        assert not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())


def test_sequence_gap_is_not_repaired_by_following_request():
    async def scenario():
        h = harness()
        h.seq += 1
        h.post(request())
        assert h.session.status == "interrupted"
        assert h.session.view().error.code == "PLATFORM_SEQUENCE_GAP"
        await h.tick()
        assert not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())


def test_wrong_connection_envelope_has_no_authority():
    async def scenario():
        h = harness()
        parsed = parse_platform_frame(request(), config=h.session.connection, room_id=1,
            rx_seq=h.seq + 1, received_at=h.clock.utc_now(), received_mono=h.clock.monotonic())
        old = h.session.view()
        h.session.post(parsed.model_copy(update={"connection_epoch": 2}))
        assert h.session.view() == old
        assert h.session.active_context is None
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("trusted", [False, True])
def test_only_trusted_match_result_plus_gameover_completes(trusted):
    async def scenario():
        h = harness()
        h.post(settled())
        if trusted:
            h.post(result())
        h.post(closed())
        assert h.session.finished.is_set()
        assert h.session.status == ("completed" if trusted else "interrupted")
        assert (h.session.match_result is not None) == trusted
        h.session.stop("after completion")
        await h.session.shutdown()
        assert h.session.status == ("completed" if trusted else "interrupted")
    asyncio.run(scenario())


def test_shutdown_does_not_wait_for_cancellation_resistant_policy_or_accept_its_late_result():
    async def scenario():
        policy = DelayedPolicy(ignore_cancel=True)
        h = harness(policy=policy)
        h.post(request())
        await h.tick()
        h.session.stop("cancel")
        await h.session.shutdown(timeout_s=0.001)
        assert h.platform.closed
        assert h.session.finished.is_set()
        policy.ready.set()
        await h.tick()
        assert not h.platform.written and not h.session.execution_records
    asyncio.run(scenario())


def test_same_card_in_later_round_cannot_confirm_old_unknown_submission():
    async def scenario():
        h = harness()
        h.post(request())
        await h.tick()
        h.scheduler.advance(5.0)
        old = h.session.execution_records[0]
        h.post(settled())
        h.post(begin(level="3"))
        h.post(play())
        assert h.session.execution_records[0].status == "unknown"
        assert h.session.execution_records[0].binding == old.binding
        assert h.session.view().round_id == 2
        await h.session.shutdown()
    asyncio.run(scenario())


def test_same_card_in_later_legitimate_request_is_not_content_deduplicated():
    async def scenario():
        h = harness()
        h.post(request())
        await h.tick()
        played = h.post(play())
        h.post(request(hand=("S2", "H4"), counts=[2, 27, 27, 27],
                       context=(0, ["Single", "2", ["S2"]])))
        await h.tick()
        h.post(play())
        assert [record.status for record in h.session.execution_records] == ["confirmed", "confirmed"]
        assert len(h.platform.written) == 2
        assert h.session.view().state.self_hand == ("H4",)
        h.session.post(played)  # old local envelope is harmless
        assert h.session.view().state.self_hand == ("H4",)
        await h.session.shutdown()
    asyncio.run(scenario())


def test_tribute_confirmation_does_not_exchange_hand_before_back_broadcast():
    async def scenario():
        h = harness(hand=("SA", "SA", "H4"))
        h.post(request(hand=("SA", "SA", "H4"), stage="tribute"))
        await h.tick()
        h.post({"type": "notify", "stage": "tribute", "result": [[0, 1, "SA"]]})
        assert h.session.execution_records[0].status == "confirmed"
        assert h.session.view().state.self_hand == ("SA", "SA", "H4")
        h.post({"type": "notify", "stage": "back", "result": [[1, 0, "C2"]], "count": 3})
        assert h.session.view().state.self_hand == ("SA", "H4", "C2")
        await h.session.shutdown()
    asyncio.run(scenario())


def test_back_confirmation_checks_original_tribute_pairing_and_target():
    async def scenario():
        h = harness(hand=("C2", "S2", "H4"))
        h.post({"type": "notify", "stage": "tribute", "result": [[1, 0, "SA"]]})
        h.post(request(hand=("C2", "S2", "H4"), stage="back", tribute=(1, "SA")))
        await h.tick()
        intent = h.platform.written[0]
        assert intent.platform_context.tribute_pos == 1
        h.post({"type": "notify", "stage": "back", "result": [[0, 1, "C2"]], "count": 3})
        assert h.session.execution_records[0].status == "confirmed"
        assert h.session.view().state.self_hand == ("S2", "H4", "SA")
        await h.session.shutdown()
    asyncio.run(scenario())


def test_real_run_connects_one_receiver_and_cleans_up_after_scripted_completion():
    async def scenario():
        h = harness()
        frames = [
            {"type": "CREATE_ROOM", "code": 200, "data": {"roomId": 1, "userNum": 0}},
            begin(), settled(), result(), closed(),
        ]
        platform = ScriptedPlatform(frames, clock=h.clock)
        session = Session(run_id=RUN, match_id=MATCH, connection=h.session.connection,
                          config=RunConfig(), platform=platform, recorder=MemoryRecorder(),
                          clock=h.clock, scheduler=h.scheduler)
        await session.run()
        assert session.status == "completed" and session.finished.is_set()
        assert session.match_result["victory"] == 0
        assert await session.wait_joined() == 1  # Fast completion still proves admission.
        assert len(platform.sent_frames) == 1  # Admission only; no fabricated action.
        await h.session.shutdown()
    asyncio.run(scenario())


def test_new_request_cannot_skip_unconfirmed_prior_pass_even_if_hand_matches():
    async def scenario():
        h = harness()
        first = ["Single", "3", ["S3"]]
        h.post(play(1, first))
        h.post(request(actions=[["PASS", "PASS", "PASS"]], counts=[3, 26, 27, 27],
                       context=(1, first), leader=(1, first)))
        await h.tick()
        h.scheduler.advance(5.0)
        assert h.session.execution_records[0].status == "unknown"
        second = ["Single", "4", ["D4"]]
        h.post(play(2, second))
        h.post(request(actions=[["PASS", "PASS", "PASS"]], counts=[3, 26, 26, 27],
                       context=(2, second), leader=(2, second)))
        assert h.session.status == "interrupted"
        assert h.session.view().error.code == "UNCONFIRMED_PREVIOUS_SUBMISSION"
        await h.tick()
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_logs_use_relative_budgets_and_canonical_execution_and_result_events():
    async def scenario():
        h = harness()
        h.post(request())
        await h.tick()
        h.post(play())
        h.post(settled())
        h.post(result())
        h.post(closed())
        events = h.recorder.events
        kinds = [event.event_type for event in events]
        for kind in ("session_started", "platform_received", "request_opened", "proposal_ready",
                     "submission_reserved", "write_started", "send_finished", "execution_updated",
                     "round_result", "match_result"):
            assert kind in kinds
        opened = next(event for event in events if event.event_type == "request_opened")
        assert opened.payload["suggestion_budget_ms"] == 29000.0
        assert opened.payload["final_budget_ms"] == 30000.0
        assert opened.payload["budget_source"] == "local_experiment"
        assert not any("_mono" in event.model_dump_json() for event in events)
        for event in events:
            if event.event_type in ("round_result", "match_result"):
                assert event.round_id == 1
                assert event.payload["evidence_event_id"].startswith(f"{RUN}:e")
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("feedback_before_send_failure", [False, True])
def test_actual_adapter_uncertain_write_interrupts_even_if_feedback_already_confirmed(feedback_before_send_failure):
    async def scenario():
        h = harness()
        frames = [
            {"type": "CREATE_ROOM", "code": 200, "data": {"roomId": 1, "userNum": 0}},
            begin(), request(),
        ]
        platform = ScriptedPlatform(frames, clock=h.clock)
        original_send = platform._script.send

        async def fail_after_write(raw):
            await original_send(raw)
            message = json.loads(raw)
            if message["type"] == "PLAY":
                if feedback_before_send_failure:
                    platform.feed(play(action=message["data"]["act"]))
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                raise OSError("synthetic failure after send")

        platform._script.send = fail_after_write
        session = Session(run_id=RUN, match_id=MATCH, connection=h.session.connection,
                          config=RunConfig(), platform=platform, recorder=MemoryRecorder(),
                          clock=h.clock, scheduler=h.scheduler)
        await asyncio.wait_for(session.run(), timeout=0.5)
        assert session.finished.is_set() and session.status == "interrupted"
        assert session.view().error.code == "SEND_UNCERTAIN"
        assert session.active_context is None
        assert [record.status for record in session.execution_records] == [
            "confirmed" if feedback_before_send_failure else "unknown"]
        assert len([frame for frame in platform.sent_frames if frame["type"] == "PLAY"]) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("code", ["PLATFORM_NOT_READY", "CONNECTION_BINDING_MISMATCH", "SEND_FAILED"])
def test_unrecoverable_before_write_failure_interrupts_current_request(code):
    class UnavailablePlatform(ControlledPlatform):
        async def write(self, intent, gate):
            return SendResult(submission_id=intent.submission_id, status="not_started",
                              error=ErrorInfo(code=code, component="platform",
                                              safe_message="fixture unavailable", related_event_id=None))

    async def scenario():
        h = harness(platform=UnavailablePlatform())
        h.post(request())
        await h.tick()
        assert h.session.finished.is_set() and h.session.status == "interrupted"
        assert h.session.execution_records[0].status == "cancelled_before_write"
        assert not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())


def test_late_not_started_result_for_invalidated_request_does_not_stop_new_request():
    async def scenario():
        h = harness(platform=ControlledPlatform(wait_gate=True))
        h.post(request())
        await h.tick()
        old = h.session.execution_records[0]
        other = ["Single", "3", ["S3"]]
        h.post(play(1, other))
        h.post(request(counts=[3, 26, 27, 27], context=(1, other), leader=(1, other)))
        h.session.post(SendFinished(binding=old.binding,
            result=SendResult(submission_id=old.submission_id, status="not_started",
                              error=ErrorInfo(code="PLATFORM_NOT_READY", component="platform",
                                              safe_message="old fixture unavailable", related_event_id=None))))
        assert h.session.status == "running"
        assert h.session.active_context.state.binding.request_id.endswith(":q2")
        h.platform.before_gate.set()
        await h.tick()
        assert len(h.platform.written) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_repeated_send_callback_is_deduplicated_by_submission_id_not_its_claimed_status():
    async def scenario():
        h = harness()
        h.post(request())
        await h.tick()
        record = h.session.execution_records[0]
        old_view = h.session.view()
        h.session.post(SendFinished(binding=record.binding,
            result=SendResult(submission_id=record.submission_id, status="unknown", error=None)))
        assert h.session.view() == old_view
        assert h.session.status == "running"
        assert len([event for event in h.recorder.events if event.event_type == "send_finished"]) == 1
        await h.session.shutdown()
    asyncio.run(scenario())


def test_async_recording_health_change_advances_only_projection_once():
    async def scenario():
        h = harness()
        before = h.session.view()
        h.recorder.health = SimpleNamespace(log_complete=False)
        after = h.session.view()
        assert not after.log_complete
        assert after.view_version == before.view_version + 1
        assert after.state_version == before.state_version
        assert h.session.view() == after
        assert h.session.status == "running"
        assert not h.session.execution_records and not h.platform.written
        await h.session.shutdown()
    asyncio.run(scenario())
