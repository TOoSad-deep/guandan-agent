"""Real runtime modules against an explicitly synthetic, network-free server."""

import asyncio
import json

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler
from tests.support.scripted_room import ScriptedRooms


def test_four_seats_two_scripted_matches_close_with_confirmed_actions(tmp_path):
    async def exercise():
        clock = SystemClock()
        rooms = ScriptedRooms(clock)
        manager = RunManager(recording_root=tmp_path, platform_factory=rooms.factory,
                             clock=clock, scheduler=AsyncioScheduler(clock), startup_timeout_s=1)
        config = RunConfig.model_validate({"run": {"matches": 2}})
        opened = manager.start(config)
        try:
            finished = await asyncio.wait_for(manager.wait(opened.run_id), timeout=4)
            assert finished.status == "completed", finished
            assert finished.log_complete
            assert len(rooms.rooms) == 2 and len(rooms.platforms) == 8
            assert len({p.connection.session_id for p in rooms.platforms}) == 8
            assert len({intent.binding.match_id for _, _, intent in rooms.writes}) == 2
            assert len(rooms.writes) == 6
            assert len({intent.binding.request_id for _, _, intent in rooms.writes}) == 6
            for platform in rooms.platforms:
                seat = platform.connection.seat
                view = manager.session_view(opened.run_id, platform.connection.session_id)
                assert view.status == "completed"
                assert view.state.self_hand == (("SJ",) if seat == 3 else ())
                if seat < 3:
                    assert view.latest_execution.status == "confirmed"
                    assert view.latest_execution.feedback_event_id is not None
            loaded = await ClosedRunLog.load(tmp_path / opened.run_id)
            assert loaded.summary.closed and loaded.summary.log_complete
            assert loaded.summary.unconfirmed_submission_ids == ()
            events = loaded.offline_events()
            assert len([e for e in events if e.event_type == "request_opened"]) == 6
            assert len([e for e in events if e.event_type == "round_result"]) == 8
            assert len([e for e in events if e.event_type == "match_result"]) == 8
            assert len([e for e in events if e.event_type == "match_completed"]) == 2
            assert len([e for e in events if e.event_type == "run_closed"]) == 1
            started = next(e for e in events if e.event_type == "run_started")
            assert len(started.payload["environment"]["source_sha256"]) == 64
            assert started.payload["environment"]["seed"] is None
            assert not any(e.event_type == "model_call_started" for e in events)
            assert json.loads((tmp_path / opened.run_id / "config.json").read_text())["config_id"] == config.config_id
        finally:
            manager.stop(opened.run_id)
            await asyncio.wait_for(manager.wait(opened.run_id), timeout=3)

    asyncio.run(exercise())


def test_stop_closes_all_real_sessions_without_inventing_a_result(tmp_path):
    async def exercise():
        clock = SystemClock()
        rooms = ScriptedRooms(clock, auto_play=False)
        manager = RunManager(recording_root=tmp_path, platform_factory=rooms.factory,
                             clock=clock, scheduler=AsyncioScheduler(clock), startup_timeout_s=1)
        opened = manager.start(RunConfig())
        try:
            async def all_admitted():
                while len(rooms.platforms) < 4 or any(p.room_id is None for p in rooms.platforms):
                    await asyncio.sleep(0)
            await asyncio.wait_for(all_admitted(), timeout=1)
            stopped = manager.stop(opened.run_id)
            assert stopped.status in ("stopping", "interrupted")
            # stop is synchronous: no old seat authority survives its return.
            for platform in rooms.platforms:
                assert manager.session_view(opened.run_id, platform.connection.session_id).status == "interrupted"
            finished = await asyncio.wait_for(manager.wait(opened.run_id), timeout=3)
            assert finished.status == "interrupted"
            assert rooms.writes == []
            loaded = await ClosedRunLog.load(tmp_path / opened.run_id)
            assert not any(e.event_type == "match_result" for e in loaded.offline_events())
            assert manager.stop(opened.run_id) == finished
        finally:
            manager.stop(opened.run_id)
            await asyncio.wait_for(manager.wait(opened.run_id), timeout=3)

    asyncio.run(exercise())
