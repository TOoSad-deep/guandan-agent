"""Historical-frame parsing and offline transport boundaries (no live match)."""

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from guandan_agent.clock import ManualClock
from guandan_agent.contracts import GateDecision, RequestBinding, SeatConnection, SubmissionIntent
from guandan_agent.platform import (
    OpenGuanDanPlatform, PlatformConnectionError, ScriptedPlatform,
    admission_command, encode_submission, parse_actions, parse_platform_frame,
)

ROOT = Path(__file__).resolve().parents[2]
FRAMES = json.loads((ROOT / "tests/fixtures/platform/historical_frames.json").read_text())
BY_NAME = {case["name"]: case for case in FRAMES}
SESSION = "11111111-1111-4111-8111-111111111111"
RUN = "22222222-2222-4222-8222-222222222222"
MATCH = "33333333-3333-4333-8333-333333333333"


def config(*, seat=0, mode="create", **changes):
    return SeatConnection(**{
        "url": "ws://127.0.0.1:18181", "mode": mode,
        "room_id": None if mode == "create" else 1,
        "user_name": "test-local", "seat": seat, "session_id": SESSION,
        "connection_epoch": 1, "server_rounds": 1, **changes,
    })


def parse(raw, *, seat=0, mode="create", **changes):
    return parse_platform_frame(raw, config=config(seat=seat, mode=mode), rx_seq=1,
                                received_at="2026-10-19T00:00:00Z", received_mono=1.0, **changes)


def intent(*, stage="play", action=None, deadline=100.0, binding=None):
    if action is None:
        action = ["PASS", "PASS", "PASS"] if stage == "play" else [stage, stage, ["D2"]]
    return SubmissionIntent(
        submission_id="submission-1", binding=binding or RequestBinding(
            run_id=RUN, match_id=MATCH, round_id=1, session_id=SESSION, seat=0,
            connection_epoch=1, request_id=f"{SESSION}:q1", state_version=1,
        ), action_id="L0000", wire_action=action, stage=stage,
        platform_context={"tribute_pos": 2, "tribute_card": "SB"} if stage == "back" else None,
        final_deadline_mono=deadline,
    )


class Gate:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls = []

    def begin_write(self, submission):
        self.calls.append(submission)
        return GateDecision(status="allow" if self.allowed else "deny", error=None)


@pytest.mark.parametrize("case", FRAMES, ids=lambda case: case["name"])
def test_historical_frames_are_verbatim_and_parse(case):
    source = case["source"]
    line = (ROOT / source["path"]).read_bytes().splitlines(keepends=True)[source["line"] - 1]
    assert hashlib.sha256(line).hexdigest() == source["sha256"]
    assert json.loads(line)["message"] == case["raw"]
    mode = "join" if case["name"] in ("joined", "join_failed") else "create"
    event = parse(json.dumps(case["raw"]), seat=case["seat"], mode=mode)
    assert event.kind == case["expected_kind"]
    assert event.session_id == SESSION and event.rx_seq == 1
    assert event.connection_epoch == 1
    assert event.received_mono == 1.0
    assert event.model_dump(mode="json")["raw_message"] == case["raw"]


def test_raw_evidence_is_copied_and_frozen_separately_from_projection():
    raw = deepcopy(BY_NAME["play_request"]["raw"])
    expected = deepcopy(raw)
    event = parse(raw)
    raw["handCards"].clear()
    assert event.model_dump(mode="json")["raw_message"] == expected
    with pytest.raises(TypeError):
        event.raw_message["type"] = "replaced"
    assert "raw_message" not in event.payload


def test_action_request_keeps_actor_distinct_from_previous_player():
    raw = BY_NAME["play_request"]["raw"]
    event = parse(raw)
    assert event.kind == "action_requested"
    assert raw["curPos"] != 0
    assert event.payload["curPos"] == raw["curPos"]
    assert event.payload["greaterPos"] == raw["greaterPos"]
    assert "binding" not in event.payload["parsed_actions"]
    assert "state" not in event.payload
    assert event.payload["platform_context"] is None


def test_duplicate_back_actions_preserve_original_indices_context_and_order():
    event = parse(BY_NAME["back_request"]["raw"])
    parsed = event.payload["parsed_actions"]
    assert parsed["raw_count"] == 32
    assert len(parsed["actions"]) == 16
    assert parsed["actions"][0]["source_indices"] == (0, 1)
    assert parsed["actions"][1]["action_id"] == "L0002"
    assert event.payload["platform_context"] == {"tribute_pos": 2, "tribute_card": "SB"}
    assert event.payload["tribute"] == "SB"


def test_exact_dedup_does_not_sort_cards_or_change_declared_rank():
    actions = [["Pair", "9", ["S9", "H9"]], ["Pair", "9", ["H9", "S9"]],
               ["Pair", "8", ["S9", "H9"]], ["Pair", "9", ["S9", "H9"]]]
    result = parse_actions(actions, "play")
    assert len(result.actions) == 3
    assert result.actions[0].source_indices == (0, 3)
    assert [row.action_id for row in result.actions] == ["L0000", "L0001", "L0002"]
    duplicate_pass = parse_actions([["PASS", "PASS", "PASS"]] * 3, "play")
    assert duplicate_pass.raw_count == 3 and len(duplicate_pass.actions) == 1
    assert duplicate_pass.actions[0].wire_action == ("PASS", "PASS", "PASS")


@pytest.mark.parametrize("actions,stage", [
    ([], "play"), ([["PASS", "PASS", []]], "play"),
    ([["Single", "9", ["S9"]]], "back"), ([["back", "back", ["D2"]]], "play"),
    ([["tribute", "tribute", ["D2", "C2"]]], "tribute"),
    ([["Single", "9", ["INVALID"]]], "play"),
])
def test_invalid_actions_raise_without_guessing(actions, stage):
    with pytest.raises(ValueError):
        parse_actions(actions, stage)


@pytest.mark.parametrize("field,value", [
    ("actionList", []), ("actionList", [["Trips", "B", ["SB", "SB", "SB"]]]),
    ("publicInfo", [{"rest": True}] * 4), ("publicInfo", [{"rest": "27"}] * 4),
    ("handCards", "S2"), ("curPos", True), ("greaterPos", "0"),
    ("curRank", "R"), ("indexRange", True), ("stage", "future-stage"),
])
def test_malformed_requests_become_safe_unknown(field, value):
    raw = deepcopy(BY_NAME["play_request"]["raw"])
    raw[field] = value
    event = parse(raw)
    assert event.kind == "unknown"
    assert "parsed_actions" not in event.payload
    assert event.payload["error"]["code"] == "INVALID_PLATFORM_FRAME"
    assert event.payload["error"]["safe_message"] == "Unrecognized or invalid platform frame"


@pytest.mark.parametrize("raw", ['{"type":', '{"type":"act","type":"notify"}',
                                 '{"type":"act","x":NaN}', '[]', b'\xff'])
def test_invalid_json_is_diagnostic_only(raw):
    event = parse(raw)
    assert event.kind == "unknown"
    json.dumps(event.model_dump(mode="json"), allow_nan=False)


def test_missing_fields_and_connection_mismatches_are_unknown():
    raw = deepcopy(BY_NAME["play_request"]["raw"])
    del raw["publicInfo"]
    assert parse(raw).kind == "unknown"
    assert parse(BY_NAME["beginning"]["raw"], seat=1).kind == "unknown"
    assert parse(BY_NAME["joined"]["raw"], seat=0, mode="join").kind == "unknown"
    assert parse(BY_NAME["joined"]["raw"], seat=1, mode="join", room_id=2).kind == "unknown"
    assert parse(BY_NAME["joined"]["raw"], seat=1).kind == "unknown"


def test_notify_leader_is_preserved_and_payload_is_frozen_copy():
    raw = deepcopy(BY_NAME["play_notice"]["raw"])
    event = parse(raw)
    assert event.payload["greaterPos"] == -1
    assert event.payload["greaterAction"] == ("PASS", "PASS", "PASS")
    raw["curAction"][2].clear()
    assert event.payload["curAction"][2]
    with pytest.raises(TypeError):
        event.payload["curPos"] = 3


def test_admission_commands_and_all_three_action_encodings():
    assert admission_command(config()) == {"type": "CREATE_ROOM", "data": {"userId": "test-local", "seatNum": 0, "round": 1}}
    assert admission_command(config(mode="join", seat=1)) == {"type": "JOIN_ROOM", "data": {"userId": "test-local", "seatNum": 1, "roomId": 1}}
    for stage, kind in [("play", "PLAY"), ("tribute", "TRIBUTE"), ("back", "PAYTRIBUTE")]:
        encoded = encode_submission(intent(stage=stage), room_id=1, seat=0)
        assert encoded["type"] == kind
        assert encoded["data"]["act"] == intent(stage=stage).model_dump(mode="json")["wire_action"]
        assert encoded["data"]["roomId"] == 1 and encoded["data"]["player"] == 0
        assert "action_id" not in encoded["data"]
        if stage == "back":
            assert encoded["data"]["tributePos"] == 2 and encoded["data"]["tribute"] == "SB"


def test_scripted_uses_same_parser_and_requires_joined_before_write():
    async def scenario():
        platform = ScriptedPlatform([BY_NAME["join_notice"]["raw"], BY_NAME["created"]["raw"]], clock=ManualClock())
        await platform.connect(config())
        gate = Gate()
        assert (await platform.write(intent(), gate)).status == "not_started"
        stream = platform.events()
        assert (await anext(stream)).kind == "auxiliary_notice"
        assert platform.room_id is None
        assert (await platform.write(intent(), gate)).status == "not_started"
        assert not gate.calls
        assert (await anext(stream)).kind == "joined"
        assert platform.room_id == 1
        assert (await platform.write(intent(), gate)).status == "sent"
        assert platform.sent_frames[-1] == encode_submission(intent(), room_id=1, seat=0)
        assert len(gate.calls) == 1
        platform.feed(BY_NAME["play_request"]["raw"])
        assert (await anext(stream)).kind == "action_requested"
        await platform.close()
        await stream.aclose()
    asyncio.run(scenario())


def test_one_receive_consumer_no_reconnect_and_safe_ordering():
    async def scenario():
        platform = ScriptedPlatform([BY_NAME["beginning"]["raw"], BY_NAME["created"]["raw"]])
        await platform.connect(config())
        stream = platform.events()
        assert (await anext(stream)).kind == "unknown"
        with pytest.raises(PlatformConnectionError, match="one consumer"):
            await anext(platform.events())
        assert (await anext(stream)).kind == "joined"
        with pytest.raises(PlatformConnectionError, match="reused"):
            await platform.connect(config())
        platform.disconnect()
        with pytest.raises(PlatformConnectionError, match="before match completion"):
            await anext(stream)
        assert (await platform.write(intent(), Gate())).status == "not_started"
        await platform.close()
    asyncio.run(scenario())


def test_end_match_closes_write_authority_without_inventing_winner():
    async def scenario():
        platform = ScriptedPlatform([BY_NAME["created"]["raw"], BY_NAME["game_over"]["raw"]])
        await platform.connect(config())
        events = [event async for event in platform.events()]
        assert [event.kind for event in events] == ["joined", "match_closed"]
        assert "victory" not in events[-1].payload
        assert (await platform.write(intent(), Gate())).status == "not_started"
        await platform.close()
    asyncio.run(scenario())


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait(json.dumps(BY_NAME["created"]["raw"]))
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = False
        self.block = False
        self.before_send = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.incoming.get()

    async def send(self, raw):
        message = json.loads(raw)
        if message["type"] != "CREATE_ROOM":
            if self.before_send:
                self.before_send()
            self.started.set()
            if self.block:
                await self.release.wait()
            if self.fail:
                raise OSError("SENSITIVE-RAW-TRANSPORT-EXCEPTION")
        self.sent.append(message)

    async def close(self):
        self.closed = True


async def connected_fake(socket, clock=None):
    async def connector(url):
        assert url == "ws://127.0.0.1:18181"
        return socket
    platform = OpenGuanDanPlatform(clock=clock or ManualClock(), connector=connector)
    await platform.connect(config())
    stream = platform.events()
    assert (await anext(stream)).kind == "joined"
    return platform, stream


@pytest.mark.parametrize("kind", ["gate_denied", "seat", "epoch", "session", "expired"])
def test_invalid_send_never_reaches_transport(kind):
    async def scenario():
        socket = FakeSocket()
        clock = ManualClock()
        platform, stream = await connected_fake(socket, clock)
        submission = intent()
        if kind in ("seat", "epoch", "session"):
            values = submission.binding.model_dump()
            if kind == "seat": values["seat"] = 1
            if kind == "epoch": values["connection_epoch"] = 2
            if kind == "session":
                values["session_id"] = RUN
                values["request_id"] = f"{RUN}:q1"
            submission = intent(binding=RequestBinding(**values))
        if kind == "expired": clock.advance(100)
        gate = Gate(allowed=kind != "gate_denied")
        result = await platform.write(submission, gate)
        assert result.status == "not_started"
        assert len(socket.sent) == 1
        assert len(gate.calls) == (1 if kind == "gate_denied" else 0)
        await stream.aclose()
        await platform.close()
    asyncio.run(scenario())


def test_gate_is_rechecked_after_send_channel_wait_and_immediately_before_send():
    async def scenario():
        socket = FakeSocket()
        socket.block = True
        platform, stream = await connected_fake(socket)
        first_gate, second_gate = Gate(), Gate()
        socket.before_send = lambda: (_ for _ in ()).throw(AssertionError()) if not first_gate.calls else None
        first = asyncio.create_task(platform.write(intent(), first_gate))
        await socket.started.wait()
        second = asyncio.create_task(platform.write(intent(), second_gate))
        await asyncio.sleep(0)
        # A Stop/new request is synchronously applied while this write waits.
        second_gate.allowed = False
        socket.release.set()
        assert (await first).status == "sent"
        assert (await second).status == "not_started"
        assert len(second_gate.calls) == 1
        assert len(socket.sent) == 2
        await stream.aclose()
        await platform.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["exception", "cancel", "timeout"])
def test_started_write_failure_is_unknown_and_never_resent(failure):
    async def scenario():
        socket = FakeSocket()
        socket.fail = failure == "exception"
        socket.block = failure in ("cancel", "timeout")
        platform, stream = await connected_fake(socket)
        submission = intent(deadline=0.01 if failure == "timeout" else 100.0)
        task = asyncio.create_task(platform.write(submission, Gate()))
        await socket.started.wait()
        if failure == "cancel": task.cancel()
        result = await task
        assert result.status == "unknown"
        assert "SENSITIVE" not in result.model_dump_json()
        assert (await platform.write(intent(), Gate())).status == "not_started"
        assert len(socket.sent) == 1
        await stream.aclose()
        await platform.close()
    asyncio.run(scenario())


def test_expired_or_cancelled_while_waiting_never_begins_write():
    async def scenario():
        socket = FakeSocket()
        socket.block = True
        clock = ManualClock()
        platform, stream = await connected_fake(socket, clock)
        first = asyncio.create_task(platform.write(intent(), Gate()))
        await socket.started.wait()
        expired_gate, cancelled_gate = Gate(), Gate()
        expired = asyncio.create_task(platform.write(intent(), expired_gate))
        cancelled = asyncio.create_task(platform.write(intent(), cancelled_gate))
        await asyncio.sleep(0)
        cancelled.cancel()
        assert (await cancelled).status == "not_started"
        clock.advance(100)
        socket.release.set()
        assert (await first).status == "sent"
        assert (await expired).status == "not_started"
        assert not expired_gate.calls and not cancelled_gate.calls
        await stream.aclose()
        await platform.close()
    asyncio.run(scenario())


def test_connect_failure_has_safe_error_and_no_retry():
    async def scenario():
        calls = 0
        async def broken(url):
            nonlocal calls
            calls += 1
            raise OSError("SENSITIVE URL OR CREDENTIAL")
        platform = OpenGuanDanPlatform(connector=broken)
        with pytest.raises(PlatformConnectionError) as caught:
            await platform.connect(config())
        assert "SENSITIVE" not in str(caught.value)
        with pytest.raises(PlatformConnectionError):
            await platform.connect(config())
        assert calls == 1
    asyncio.run(scenario())
