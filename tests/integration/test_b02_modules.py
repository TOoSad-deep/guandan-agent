"""Module handoffs using isolated historical frames, not a Session or real match."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from guandan_agent.clock import ManualClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import (
    DecisionInput, EventDraft, EventQueryScope, GateDecision, LegalActionSet,
    ParsedActions, RequestBinding, RequestContext, SeatConnection, SubmissionIntent,
    VisibleState,
)
from guandan_agent.hand import CHEAP_RULE_VERSION, cheap_choice
from guandan_agent.platform import ScriptedPlatform
from guandan_agent.recording import ClosedRunLog, EventStore
from guandan_agent.strategy import DecisionPolicy

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/b02_integration.json"
RUN = "00000000-0000-4000-8000-000000000021"
MATCH = "00000000-0000-4000-8000-000000000022"
SESSION = "00000000-0000-4000-8000-000000000023"
OTHER_SESSION = "00000000-0000-4000-8000-000000000024"


class OneWriteGate:
    """A deterministic test permission only; production Session arrives in B03."""

    def __init__(self):
        self.starts = 0

    def begin_write(self, intent):
        self.starts += 1
        return GateDecision(status="allow" if self.starts == 1 else "deny", error=None)


class NoModelCalls:
    def __getattr__(self, name):
        raise AssertionError("B02 must not access a model call port")


def test_integration_fixture_is_unchanged_historical_input():
    fixture = json.loads(FIXTURE.read_text())
    data = (ROOT / fixture["source_file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == fixture["source_sha256"]
    lines = data.decode().splitlines()
    for item in fixture["frames"]:
        assert json.loads(lines[item["line"] - 1])["message"] == item["frame"]


@pytest.mark.parametrize("frame_index,source,command", [(1, "rule", "PLAY"), (2, "unique", "TRIBUTE")])
def test_platform_rules_and_recording_handoff(tmp_path, frame_index, source, command):
    async def exercise():
        fixture = json.loads(FIXTURE.read_text())
        clock = ManualClock(monotonic_start=100)
        # This handoff remains the explicitly versioned B02 cheap baseline.
        config = RunConfig.model_validate({"versions": {
            "policy": CHEAP_RULE_VERSION, "candidate": "v1", "prompt": "v1"}})
        assert config.versions.policy == config.versions.fallback == CHEAP_RULE_VERSION
        transport = ScriptedPlatform(
            [fixture["frames"][0]["frame"], fixture["frames"][frame_index]["frame"]], clock=clock,
        )
        connection = SeatConnection(url=config.platform.url, mode="create", room_id=None,
                                    user_name="module-test", seat=0, session_id=SESSION,
                                    connection_epoch=1, server_rounds=1)
        await transport.connect(connection)
        events = transport.events()
        store = None
        try:
            assert (await anext(events)).kind == "joined"
            event = await anext(events)
            assert event.kind == "action_requested"
            # This manually assembled test input is not the future state reducer.
            data = event.model_dump(mode="json")["payload"]
            binding = RequestBinding(run_id=RUN, match_id=MATCH, round_id=1,
                                     session_id=SESSION, seat=0, connection_epoch=1,
                                     request_id=f"{SESSION}:q1", state_version=1)
            parsed = ParsedActions.model_validate(data["parsed_actions"])
            legal = LegalActionSet(binding=binding, **parsed.model_dump())
            is_play = data["stage"] == "play"
            leading = data.get("greaterAction") if data.get("greaterPos", -1) >= 0 else None
            state = VisibleState(
                binding=binding, stage=data["stage"], self_hand=data["handCards"],
                level_rank=data["curRank"],
                remaining_counts=([p["rest"] for p in data["publicInfo"]] if is_play
                                  else [len(data["handCards"]), None, None, None]),
                team_ranks=[data["selfRank"], data["oppoRank"]], turn_seat=0,
                leader_seat=data["greaterPos"] if leading else None, leading_action=leading,
                finished_order=[], history=[], history_complete=False, history_omitted=0,
                facts_status="trusted",
            )
            context = RequestContext(state=state, legal=legal, received_at=event.received_at,
                                     fallback_action_id=cheap_choice(state, legal), policy="V0",
                                     platform_context=data["platform_context"])
            outcome = await DecisionPolicy().decide(DecisionInput(context=context), NoModelCalls())
            assert outcome.proposal.source == source
            action = next(a for a in legal.actions if a.action_id == outcome.proposal.action_id)
            if source == "unique":
                assert legal.raw_count == 2 and len(legal.actions) == 1
                assert action.source_indices == (0, 1)
            intent = SubmissionIntent(submission_id="test-submit-1", binding=binding,
                                      action_id=action.action_id, wire_action=action.wire_action,
                                      stage=state.stage, platform_context=context.platform_context,
                                      final_deadline_mono=clock.monotonic() + 30)
            gate = OneWriteGate()
            sent = await transport.write(intent, gate)
            assert sent.status == "sent" and gate.starts == 1
            assert transport.sent_frames[-1]["type"] == command
            raw = fixture["frames"][frame_index]["frame"]["actionList"][action.source_indices[0]]
            assert transport.sent_frames[-1]["data"]["act"] == raw
            # Scripted transport acceptance is not a server confirmation.
            store = await EventStore.create(tmp_path / "module-log", RUN, config, clock=clock)
            draft = dict(schema_version="1.0", run_id=RUN, match_id=MATCH, session_id=SESSION,
                         seat=0, round_id=1, request_id=binding.request_id)
            assert store.append(EventDraft(**draft, event_type="proposal_ready",
                                           payload={"proposal": outcome.proposal.model_dump(mode="json"),
                                                    "usage": None})).accepted
            assert store.append(EventDraft(**draft, event_type="send_finished",
                                           payload=sent.model_dump(mode="json"))).accepted
            assert store.append(EventDraft(**{**draft, "session_id": OTHER_SESSION, "seat": 1,
                                              "request_id": f"{OTHER_SESSION}:q1"},
                                           event_type="request_opened", payload={"self_hand": ["SA"]})).accepted
            scope = EventQueryScope(run_id=RUN, session_id=SESSION, seat=0)
            page = store.query(scope)
            assert len(page.items) == 2
            assert all(item.seat == 0 and item.session_id == SESSION for item in page.items)
            assert store.query(EventQueryScope(run_id=RUN, session_id=SESSION, seat=1)).items == ()
            summary = await store.close(RUN, unconfirmed_submission_ids=(intent.submission_id,))
            assert summary.closed and summary.log_complete
            assert summary.unconfirmed_submission_ids == (intent.submission_id,)
            loaded = await ClosedRunLog.load(tmp_path / "module-log")
            assert loaded.query(scope).items == page.items
            assert loaded.query(scope).items[0].payload["usage"] is None
            all_events = loaded.offline_events()
            assert len([item for item in all_events if item.session_id is not None]) == 3
            assert all_events[-1].event_type == "run_closed"
        finally:
            if store is not None:
                await store.close(RUN)
            await events.aclose()
            await transport.close()

    asyncio.run(exercise())
