"""Actual V1 graph and call authority with local scripted provider/room only."""

import asyncio
import json

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.hand import HandTools
from guandan_agent.model.scripted import ScriptedProvider
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler
from tests.support.scripted_room import ScriptedRooms


def test_v1_uses_single_run_budget_across_seats_and_matches(tmp_path):
    async def scenario():
        clock = SystemClock()
        rooms = ScriptedRooms(clock, pair_options=True)
        provider = ScriptedProvider(['{"action_id":"L0001","reason":"固定合法候选"}'] * 2, clock=clock)
        manager = RunManager(clock=clock, scheduler=AsyncioScheduler(clock), recording_root=tmp_path,
                             platform_factory=rooms.factory, provider_factory=lambda _: provider,
                             tools_factory=lambda _: HandTools(clock=clock))
        config = RunConfig.model_validate({"run": {"policy": "V1", "matches": 2},
                                          "batch": {"max_model_calls": 2}, "tools": {"deep_eval_limit": 0}})
        run = manager.start(config)
        try:
            final = await asyncio.wait_for(manager.wait(run.run_id), 4)
            assert final.status == "completed", final
            assert final.log_complete and provider.closed
            assert len(provider.inputs) == 2
            assert len(rooms.writes) == 6
            loaded = await ClosedRunLog.load(tmp_path / run.run_id)
            events = loaded.offline_events()
            starts = [e for e in events if e.event_type == "model_call_started"]
            assert len(starts) == 2 and len({e.session_id for e in starts}) == 2
            assert len([e for e in events if e.event_type == "main_registered"]) == 2
            chosen = [e.payload["execution"] for e in events if e.event_type == "submission_reserved"]
            assert [entry["source"] for entry in chosen] == ["main", "main", "fallback", "fallback", "fallback", "fallback"]
            assert [entry["action_id"] for entry in chosen] == ["L0001", "L0001", "L0000", "L0000", "L0000", "L0000"]
            assert sum(e.event_type == "candidates_prepared" for e in events) == 6
            budget = next(e.payload for e in events if e.event_type == "model_budget_summary")
            assert budget["slots_used"] == budget["calls_started"] == 2
            assert loaded.summary.unconfirmed_submission_ids == loaded.summary.pending_call_ids == ()
            for request in provider.inputs:
                payload = json.loads(request.messages[1].content)
                assert "self_hand" in payload["visible"]
                assert not {"binding", "legal", "raw_message", "config", "other_hands"} & payload.keys()
        finally:
            manager.stop(run.run_id)
            await manager.wait(run.run_id)
    asyncio.run(scenario())


def test_v1_invalid_model_id_falls_back_without_repair(tmp_path):
    async def scenario():
        clock = SystemClock()
        rooms = ScriptedRooms(clock, pair_options=True)
        provider = ScriptedProvider(['{"action_id":"L9999","reason":"无效候选"}'], clock=clock)
        manager = RunManager(clock=clock, scheduler=AsyncioScheduler(clock), recording_root=tmp_path,
                             platform_factory=rooms.factory, provider_factory=lambda _: provider,
                             tools_factory=lambda _: HandTools(clock=clock))
        config = RunConfig.model_validate({"run": {"policy": "V1"},
                                          "batch": {"max_model_calls": 1}, "tools": {"deep_eval_limit": 0}})
        run = manager.start(config)
        try:
            final = await asyncio.wait_for(manager.wait(run.run_id), 4)
            assert final.status == "completed"
            assert len(provider.inputs) == 1
            loaded = await ClosedRunLog.load(tmp_path / run.run_id)
            events = loaded.offline_events()
            assert not any(e.event_type == "main_registered" for e in events)
            chosen = [e.payload["execution"] for e in events if e.event_type == "submission_reserved"]
            assert len(chosen) == 3 and all(e["source"] == "fallback" for e in chosen)
            assert not loaded.summary.pending_call_ids
        finally:
            manager.stop(run.run_id)
            await manager.wait(run.run_id)
    asyncio.run(scenario())


def test_new_v0_uses_candidates_without_creating_a_provider(tmp_path):
    async def scenario():
        clock = SystemClock()
        rooms = ScriptedRooms(clock, pair_options=True)

        def forbidden(_):
            raise AssertionError("V0 must not construct a provider or read its credentials")

        manager = RunManager(clock=clock, scheduler=AsyncioScheduler(clock), recording_root=tmp_path,
                             platform_factory=rooms.factory, provider_factory=forbidden,
                             tools_factory=lambda _: HandTools(clock=clock))
        run = manager.start(RunConfig.model_validate({"tools": {"deep_eval_limit": 0}}))
        final = await asyncio.wait_for(manager.wait(run.run_id), 4)
        assert final.status == "completed"
        loaded = await ClosedRunLog.load(tmp_path / run.run_id)
        events = loaded.offline_events()
        assert sum(e.event_type == "candidates_prepared" for e in events) == 3
        assert not any(e.event_type == "model_call_started" for e in events)
        assert all(e.payload["execution"]["source"] == "rule"
                   for e in events if e.event_type == "submission_reserved")
    asyncio.run(scenario())


def test_explicit_v1_without_process_key_fails_before_platform_connections(tmp_path, monkeypatch):
    from guandan_agent import app

    def forbidden(**kwargs):
        raise AssertionError("A missing key must be detected before constructing a platform")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(app, "OpenGuanDanPlatform", forbidden)

    async def scenario():
        final = await app.run_config(RunConfig.model_validate({"run": {"policy": "V1"}}),
                                     recording_root=tmp_path, max_runtime_s=2)
        assert final.status == "interrupted"
        assert final.error.code == "MODEL_CREDENTIALS_MISSING"
        assert final.sessions == ()
    asyncio.run(scenario())
