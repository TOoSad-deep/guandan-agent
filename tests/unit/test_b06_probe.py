"""P13 orchestration checks use real local manager, scripted room/provider only."""
import asyncio
import json

import pytest

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.hand import HandTools
from guandan_agent.model.scripted import ScriptedProvider
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler
from scripts import run_b06_probe as probe
from tests.support.scripted_room import ScriptedRooms


def config(limit=4, **overrides):
    value = {
        "run": {"split": "development", "seats": [
            {"seat": seat, "user_name": f"probe-{seat}", "policy": "V1" if seat == 0 else "V0"}
            for seat in range(4)]},
        "batch": {"max_model_calls": limit}, "offline": {"max_model_calls": 0},
        "tools": {"deep_eval_limit": 0},
    }
    value.update(overrides)
    return RunConfig.model_validate(value)


def event(seq, kind, payload, request="request-1"):
    return {"seq": seq, "event_id": f"run:e{seq}", "event_type": kind,
            "payload": payload, "request_id": request}


def finished(seq, *, success=True, request="request-1"):
    return event(seq, "model_call_finished", {"late": False, "result": {
        "call_id": "call-1", "decision": {"action_id": "L0000"} if success else None,
        "error": None if success else {"code": "MODEL_TRANSPORT"}}}, request)


def execution(seq, status):
    return event(seq, "execution_updated", {"execution": {
        "source": "main", "status": status, "submission_id": "submission-1",
        "feedback_event_id": "run:e3" if status == "confirmed" else None,
        "action_id": "L0000"}})


def test_default_only_checks_config_without_starting_or_loading_credentials(tmp_path, monkeypatch, capsys):
    path = tmp_path / "probe.toml"
    path.write_text('''[batch]
max_model_calls=4
[offline]
max_model_calls=0
[[run.seats]]
seat=0
user_name="p0"
policy="V1"
[[run.seats]]
seat=1
user_name="p1"
policy="V0"
[[run.seats]]
seat=2
user_name="p2"
policy="V0"
[[run.seats]]
seat=3
user_name="p3"
policy="V0"
''')
    def forbidden(**kwargs):
        raise AssertionError("config check cannot create manager")
    monkeypatch.setattr(probe.app, "create_manager", forbidden)
    assert probe.main([str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["starts_run"] is False and output["budget_limit"] == 4


@pytest.mark.parametrize("override", [
    {"batch": {"max_model_calls": 5}},
    {"batch": {"max_model_calls": 0}},
    {"offline": {"max_model_calls": 1}},
    {"platform": {"url": "wss://example.com/game"}},
    {"run": {"policy": "V1", "split": "final"}},
    {"run": {"policy": "V1", "matches": 2}},
])
def test_config_enforces_bounded_development_scope(override):
    with pytest.raises(ValueError):
        probe.validate_probe_config(config(**override))


def test_incremental_tail_retries_partial_lines_and_reads_each_event_once(tmp_path):
    path = tmp_path / "events.jsonl"
    tail = probe.IncrementalEvents(path)
    assert tail.read() == []
    first = json.dumps({"x": "中文"}).encode()
    path.write_bytes(first[:7])
    assert tail.read() == []
    with path.open("ab") as stream:
        stream.write(first[7:] + b"\n" + b'{"x":2}\n')
    assert tail.read() == [{"x": "中文"}, {"x": 2}]
    assert tail.read() == []
    tail.close()
    assert tail.stream.closed


def test_exhaustion_waits_for_last_valid_main_confirmation_or_timeout():
    seen = probe.ProbeObservation(1)
    seen.consume(event(1, "model_call_admitted", {"call_id": "call-1"}))
    seen.consume(finished(2))
    assert not seen.exhausted_without_possible_confirmation
    for seq, status in enumerate(("reserved", "writing", "sent"), start=3):
        seen.consume(execution(seq, status))
        assert not seen.exhausted_without_possible_confirmation
    seen.consume(event(6, "request_invalidated", {"cause_code": "PUBLIC_UPDATE"}))
    assert not seen.exhausted_without_possible_confirmation  # A written action can still confirm.
    seen.consume(execution(7, "unknown"))
    assert seen.exhausted_without_possible_confirmation
    seen.consume(execution(8, "confirmed"))
    assert seen.main_confirmation["event_id"] == "run:e8"
    assert not seen.exhausted_without_possible_confirmation


def test_failed_last_call_stops_without_attempting_repair():
    seen = probe.ProbeObservation(1)
    seen.consume(event(1, "model_call_admitted", {"call_id": "call-1"}))
    assert not seen.exhausted_without_possible_confirmation
    seen.consume(finished(2, success=False))
    assert seen.exhausted_without_possible_confirmation


def test_four_failed_calls_are_counted_once_and_exhaust_allocation():
    seen = probe.ProbeObservation(4)
    for i in range(4):
        call = f"call-{i}"
        admitted = event(i * 2 + 1, "model_call_admitted", {"call_id": call})
        seen.consume(admitted)
        seen.consume(admitted)  # Observer counters do not inflate on duplicate IDs.
        result = finished(i * 2 + 2, success=False)
        result["payload"]["result"]["call_id"] = call
        seen.consume(result)
        assert seen.exhausted_without_possible_confirmation == (i == 3)


def test_invalidated_success_without_submission_no_longer_blocks_budget_stop():
    seen = probe.ProbeObservation(1)
    seen.consume(event(1, "model_call_admitted", {"call_id": "call-1"}))
    seen.consume(finished(2))
    assert not seen.exhausted_without_possible_confirmation
    seen.consume(event(3, "request_invalidated", {"cause_code": "FINAL_DEADLINE"}))
    assert seen.exhausted_without_possible_confirmation
    assert seen.main_confirmation is None


class OneActionRoom(ScriptedRooms):
    def _request(self, room, seat):
        if seat == 0:
            super()._request(room, seat)


@pytest.mark.parametrize("response,limit,expected", [
    ('{"action_id":"L0001","reason":"合法候选"}', 4, "main_confirmed"),
    ('{"action_id":"L9999","reason":"非法候选"}', 1, "budget_exhausted"),
])
def test_probe_stops_real_manager_after_scripted_confirmation_or_exhaustion(tmp_path, response, limit, expected):
    async def scenario():
        clock = SystemClock()
        rooms = OneActionRoom(clock, pair_options=True)
        provider = ScriptedProvider([response], clock=clock)
        def factory(**kwargs):
            return RunManager(clock=clock, scheduler=AsyncioScheduler(clock), **kwargs,
                              platform_factory=rooms.factory,
                              tools_factory=lambda _: HandTools(clock=clock),
                              provider_factory=lambda _: provider)
        result = await probe.run_probe(config(limit), recording_root=tmp_path,
                                       manager_factory=factory, max_runtime_s=2)
        assert result["stop_reason"] == expected
        assert result["status"] == "interrupted" and result["log_complete"]
        assert len(provider.inputs) == 1 and provider.closed
        assert bool(result["observed_main_confirmation"]) == (expected == "main_confirmed")
        loaded = await ClosedRunLog.load(tmp_path / result["run_id"])
        assert loaded.summary.closed and loaded.summary.log_complete
        assert not any(e.event_type == "match_completed" for e in loaded.offline_events())
        report = json.loads((tmp_path / result["run_id"] / "reports/b06-probe.json").read_text())
        assert report["requires_independent_acceptance_audit"] is True
    asyncio.run(scenario())


def test_runtime_limit_stops_waiting_room_and_closes_resources(tmp_path):
    async def scenario():
        clock = SystemClock()
        rooms = ScriptedRooms(clock, auto_play=False)
        provider = ScriptedProvider([], clock=clock)
        def factory(**kwargs):
            return RunManager(clock=clock, scheduler=AsyncioScheduler(clock), **kwargs,
                              platform_factory=rooms.factory,
                              provider_factory=lambda _: provider)
        result = await probe.run_probe(config(), recording_root=tmp_path,
                                       manager_factory=factory, max_runtime_s=0.05)
        assert result["stop_reason"] == "runtime_limit" and result["status"] == "interrupted"
        assert result["observed_main_confirmation"] is None and provider.closed
        assert result["observed_calls_started"] == 0
    asyncio.run(scenario())
