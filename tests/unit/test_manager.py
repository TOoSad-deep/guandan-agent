import asyncio
from types import SimpleNamespace

import pytest

from guandan_agent.clock import ManualClock
from guandan_agent.config import RunConfig
from guandan_agent.runtime.manager import RunManager, RunManagerError


class FakeRecorder:
    def __init__(self, *, complete=True, append_failure=False):
        self.health = SimpleNamespace(log_complete=complete)
        self.events = []
        self.closed = False
        self.close_args = None
        self.append_failure = append_failure

    def append(self, event):
        if self.append_failure:
            raise OSError("private exception contents")
        self.events.append(event)
        return SimpleNamespace(accepted=True)

    async def close(self, run_id, **kwargs):
        self.closed = True
        self.close_args = (run_id, kwargs)
        return SimpleNamespace(closed=True, log_complete=self.health.log_complete)


class FakeSession:
    def __init__(self, *, auto_join=True, auto_complete=False, **kwargs):
        self.kwargs = kwargs
        self.connection = kwargs["connection"]
        self.status = "starting"
        self.room_id = None
        self.match_result = None
        self.execution_records = ()
        self.joined = asyncio.Event()
        self.finished = asyncio.Event()
        self.authorized = True
        self.closed = False
        self.auto_join = auto_join
        self.auto_complete = auto_complete

    async def run(self):
        if self.auto_join:
            self.admit(self.connection.room_id if self.connection.room_id is not None else 77)
        if self.auto_complete:
            self.complete()
        await self.finished.wait()

    def admit(self, room_id=77):
        self.room_id = room_id
        self.status = "running"
        self.joined.set()

    async def wait_joined(self):
        await self.joined.wait()
        return self.room_id

    def complete(self, winner=0):
        self.match_result = {"victory": winner}
        self.status = "completed"
        self.finished.set()

    def stop(self, reason="stopped"):
        self.authorized = False
        if self.status != "completed":
            self.status = "interrupted"
        self.finished.set()

    async def shutdown(self, timeout_s):
        self.closed = True

    def view(self):
        return SimpleNamespace(session_id=self.connection.session_id,
                               seat=self.connection.seat, self_hand=("S2",))


def harness(tmp_path, *, session_options=None, recorder=None, recorder_factory=None,
            startup_timeout_s=0.2, shutdown_timeout_s=0.2):
    sessions = []
    platforms = []
    recorder = recorder or FakeRecorder()

    def platform_factory(connection):
        platforms.append(connection)
        return object()

    def session_factory(**kwargs):
        options = session_options(kwargs) if callable(session_options) else session_options or {}
        session = FakeSession(**kwargs, **options)
        sessions.append(session)
        return session

    async def make_recorder(*args, **kwargs):
        return recorder

    manager = RunManager(
        platform_factory=platform_factory, session_factory=session_factory,
        recorder_factory=recorder_factory or make_recorder, clock=ManualClock(),
        scheduler=object(), recording_root=tmp_path,
        startup_timeout_s=startup_timeout_s, shutdown_timeout_s=shutdown_timeout_s,
    )
    return manager, sessions, platforms, recorder


async def spin_until(condition):
    async with asyncio.timeout(1):
        while not condition():
            await asyncio.sleep(0)


def test_start_claim_is_synchronous_and_conflict_does_not_allocate(tmp_path):
    async def scenario():
        manager, sessions, platforms, _ = harness(tmp_path)
        initial = manager.start(RunConfig())
        assert initial.status == "starting"
        assert not sessions and not platforms
        with pytest.raises(RunManagerError) as caught:
            manager.start(RunConfig())
        assert caught.value.error.code == "RUN_CONFLICT"
        stopped = manager.stop(initial.run_id)
        assert stopped.status == "stopping"
        final = await asyncio.wait_for(manager.wait(initial.run_id), 1)
        assert final.status == "interrupted"
        assert not platforms
        assert manager.stop(initial.run_id) == final
        next_run = manager.start(RunConfig())
        assert next_run.run_id != initial.run_id
        manager.stop(next_run.run_id)
        await manager.wait(next_run.run_id)
    asyncio.run(scenario())


def test_create_waits_for_returned_room_before_joining_following_seats(tmp_path):
    async def scenario():
        manager, sessions, connections, recorder = harness(
            tmp_path, session_options=lambda kwargs: {"auto_join": kwargs["connection"].seat != 2})
        config = RunConfig.model_validate({"run": {"seats": [
            {"seat": 2, "user_name": "first"}, {"seat": 0, "user_name": "second"}]}})
        run = manager.start(config)
        await spin_until(lambda: len(sessions) == 1)
        assert connections[0].mode == "create" and connections[0].room_id is None
        await asyncio.sleep(0)
        assert len(sessions) == 1
        sessions[0].admit(room_id=912)
        await spin_until(lambda: len(sessions) == 2 and sessions[1].joined.is_set())
        assert connections[1].mode == "join" and connections[1].room_id == 912
        assert connections[0].session_id != connections[1].session_id
        assert all(item.kwargs["recorder"] is recorder for item in sessions)
        assert manager.view(run.run_id).sessions[0].seat == 2
        for item in sessions:
            item.complete()
        result = await manager.wait(run.run_id)
        assert result.status == "completed" and result.log_complete
        assert recorder.closed
        assert [event.event_type for event in recorder.events].count("match_completed") == 1
        # Canonical match_result belongs to the per-seat platform evidence;
        # manager must not add a fifth result with a different payload shape.
        assert "match_result" not in [event.event_type for event in recorder.events]
    asyncio.run(scenario())


def test_join_mode_uses_existing_room_for_every_controlled_seat(tmp_path):
    async def scenario():
        manager, _, connections, _ = harness(tmp_path, session_options={"auto_complete": True})
        config = RunConfig.model_validate({"platform": {"mode": "join", "room_id": 128}})
        run = manager.start(config)
        result = await manager.wait(run.run_id)
        assert result.status == "completed"
        assert len(connections) == 4
        assert all(item.mode == "join" and item.room_id == 128 for item in connections)
    asyncio.run(scenario())


def test_sequential_matches_get_new_match_and_session_ids(tmp_path):
    async def scenario():
        manager, sessions, connections, _ = harness(tmp_path, session_options={"auto_complete": True})
        initial = manager.start(RunConfig.model_validate({"run": {"matches": 2}}))
        final = await manager.wait(initial.run_id)
        assert final.status == "completed" and final.active_match_id is None
        assert len(final.sessions) == 8
        assert len({item.session_id for item in final.sessions}) == 8
        assert len({item.match_id for item in final.sessions}) == 2
        assert [item.mode for item in connections] == ["create", "join", "join", "join"] * 2
        assert all(item.closed for item in sessions)
    asyncio.run(scenario())


@pytest.mark.parametrize("policy", ["V1", "V2"])
def test_any_non_v0_seat_is_rejected_before_resources(tmp_path, policy):
    async def scenario():
        manager, sessions, platforms, recorder = harness(tmp_path)
        config = RunConfig.model_validate({"run": {"seats": [
            {"seat": 0, "user_name": "test", "policy": policy}]}})
        with pytest.raises(RunManagerError) as caught:
            manager.start(config)
        assert caught.value.error.code == "POLICY_UNAVAILABLE"
        assert not sessions and not platforms and not recorder.closed
    asyncio.run(scenario())


@pytest.mark.parametrize("changed", [
    {"platform": {"rules_version": "0" * 40}},
    {"platform": {"adapter_version": "unknown"}},
    {"versions": {"policy": "unknown"}},
    {"versions": {"fallback": "unknown"}},
])
def test_unknown_versions_are_rejected_before_resources(tmp_path, changed):
    async def scenario():
        manager, _, connections, _ = harness(tmp_path)
        with pytest.raises(RunManagerError) as caught:
            manager.start(RunConfig.model_validate(changed))
        assert caught.value.error.code == "VERSION_UNSUPPORTED"
        assert not connections
    asyncio.run(scenario())


def test_stop_revokes_every_existing_session_before_returning(tmp_path):
    async def scenario():
        manager, sessions, _, _ = harness(tmp_path)
        run = manager.start(RunConfig())
        await spin_until(lambda: len(sessions) == 4)
        manager.stop(run.run_id)
        assert all(not item.authorized for item in sessions)
        assert manager.stop(run.run_id).status == "stopping"
        result = await manager.wait(run.run_id)
        assert result.status == "interrupted"
        assert all(item.match_result is None and item.closed for item in sessions)
    asyncio.run(scenario())


def test_stop_during_recorder_creation_never_connects(tmp_path):
    async def scenario():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def stalled(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        manager, _, connections, _ = harness(tmp_path, recorder_factory=stalled)
        run = manager.start(RunConfig())
        await entered.wait()
        manager.stop(run.run_id)
        final = await asyncio.wait_for(manager.wait(run.run_id), 0.1)
        await cancelled.wait()
        assert final.status == "interrupted" and not connections
    asyncio.run(scenario())


def test_late_recorder_acquisition_after_stop_is_closed_without_connecting(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        recorder = FakeRecorder()

        async def late(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                return recorder

        manager, _, connections, _ = harness(tmp_path, recorder_factory=late)
        initial = manager.start(RunConfig())
        await entered.wait()
        manager.stop(initial.run_id)
        final = await asyncio.wait_for(manager.wait(initial.run_id), 0.1)
        assert final.status == "interrupted" and not connections
        release.set()
        await spin_until(lambda: recorder.closed)
        assert not connections and manager.view(initial.run_id).status == "interrupted"
    asyncio.run(scenario())


def test_failed_log_start_leaves_queryable_run_and_releases_claim(tmp_path):
    async def scenario():
        async def failing(*args, **kwargs):
            raise OSError("a secret-like value must not escape")
        manager, _, connections, _ = harness(tmp_path, recorder_factory=failing)
        initial = manager.start(RunConfig())
        result = await manager.wait(initial.run_id)
        assert result.status == "interrupted" and not result.log_complete
        assert "secret" not in result.model_dump_json()
        assert not connections
        assert manager.view(initial.run_id) == result
        next_run = manager.start(RunConfig())
        manager.stop(next_run.run_id)
        await manager.wait(next_run.run_id)
    asyncio.run(scenario())


def test_admission_timeout_stops_seat_and_closes_log(tmp_path):
    async def scenario():
        manager, sessions, connections, recorder = harness(
            tmp_path, session_options={"auto_join": False}, startup_timeout_s=0.01)
        initial = manager.start(RunConfig())
        result = await asyncio.wait_for(manager.wait(initial.run_id), 1)
        assert result.status == "interrupted" and result.error.code == "STARTUP_TIMEOUT"
        assert len(connections) == 1 and not sessions[0].authorized
        assert sessions[0].closed and recorder.closed
    asyncio.run(scenario())


def test_join_failure_does_not_attempt_remaining_seats(tmp_path):
    async def scenario():
        manager, sessions, connections, recorder = harness(
            tmp_path, session_options={"auto_join": False})
        initial = manager.start(RunConfig())
        await spin_until(lambda: len(sessions) == 1)
        sessions[0].stop("join_failed")
        final = await asyncio.wait_for(manager.wait(initial.run_id), 0.1)
        assert final.status == "interrupted" and len(connections) == 1 and recorder.closed
    asyncio.run(scenario())


def test_one_seat_interruption_stops_all_without_inventing_result(tmp_path):
    async def scenario():
        manager, sessions, _, recorder = harness(tmp_path)
        run = manager.start(RunConfig())
        await spin_until(lambda: len(sessions) == 4)
        sessions[1].stop("disconnected")
        result = await asyncio.wait_for(manager.wait(run.run_id), 1)
        assert result.status == "interrupted" and result.error.code == "SESSION_INTERRUPTED"
        assert all(not item.authorized for item in sessions)
        assert all(item.match_result is None for item in sessions)
        assert "match_completed" not in [event.event_type for event in recorder.events]
    asyncio.run(scenario())


def test_contradictory_seat_results_are_not_completed(tmp_path):
    async def scenario():
        manager, sessions, _, _ = harness(tmp_path)
        run = manager.start(RunConfig())
        await spin_until(lambda: len(sessions) == 4 and all(item.joined.is_set() for item in sessions))
        for index, item in enumerate(sessions):
            item.complete(winner=index % 2)
        result = await manager.wait(run.run_id)
        assert result.status == "interrupted" and result.error.code == "MATCH_RESULT_CONFLICT"
    asyncio.run(scenario())


@pytest.mark.parametrize("append_failure", [False, True])
def test_log_failure_keeps_completed_game_separate(tmp_path, append_failure):
    async def scenario():
        recorder = FakeRecorder(complete=False, append_failure=append_failure)
        manager, _, _, _ = harness(tmp_path, recorder=recorder,
                                   session_options={"auto_complete": True})
        run = manager.start(RunConfig())
        result = await manager.wait(run.run_id)
        assert result.status == "completed" and not result.log_complete
    asyncio.run(scenario())


def test_close_keeps_unconfirmed_submission_identifiers(tmp_path):
    async def scenario():
        manager, sessions, _, recorder = harness(tmp_path)
        initial = manager.start(RunConfig())
        await spin_until(lambda: len(sessions) == 4)
        sessions[0].execution_records = (
            SimpleNamespace(submission_id="sent", status="sent"),
            SimpleNamespace(submission_id="unknown", status="unknown"),
            SimpleNamespace(submission_id="confirmed", status="confirmed"),
        )
        manager.stop(initial.run_id)
        await manager.wait(initial.run_id)
        assert recorder.close_args[1]["unconfirmed_submission_ids"] == ("sent", "unknown")
        assert recorder.close_args[1]["pending_call_ids"] == ()
    asyncio.run(scenario())


def test_hanging_log_close_is_bounded_and_releases_active_claim(tmp_path):
    async def scenario():
        class SlowRecorder(FakeRecorder):
            async def close(self, *args, **kwargs):
                await asyncio.Event().wait()

        manager, _, _, _ = harness(tmp_path, recorder=SlowRecorder(),
                                   session_options={"auto_complete": True},
                                   shutdown_timeout_s=0.01)
        initial = manager.start(RunConfig())
        final = await asyncio.wait_for(manager.wait(initial.run_id), 0.5)
        assert final.status == "completed" and not final.log_complete
        next_run = manager.start(RunConfig())
        manager.stop(next_run.run_id)
        await manager.wait(next_run.run_id)
    asyncio.run(scenario())


def test_single_session_read_scope_and_unknown_ownership(tmp_path):
    async def scenario():
        manager, _, _, _ = harness(tmp_path, session_options={"auto_complete": True})
        run = manager.start(RunConfig())
        first = await manager.wait(run.run_id)
        session = first.sessions[0]
        selected = manager.session_view(first.run_id, session.session_id)
        assert selected.seat == 0 and selected.session_id == session.session_id
        next_run = manager.start(RunConfig())
        with pytest.raises(RunManagerError) as caught:
            manager.session_view(next_run.run_id, session.session_id)
        assert caught.value.error.code == "SESSION_NOT_FOUND"
        with pytest.raises(RunManagerError):
            manager.view("unknown")
        manager.stop(next_run.run_id)
        await manager.wait(next_run.run_id)
    asyncio.run(scenario())
