import asyncio
import json
from pathlib import Path
import threading

import pytest

from guandan_agent.clock import ManualClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import EventDraft, EventQueryScope
from guandan_agent.recording import (
    ClosedRunLog, EventStore, FileEventSink, HistoryUnavailable, RecordingClosedError,
)


RUN = "00000000-0000-4000-8000-000000000001"
SESSION0 = "00000000-0000-4000-8000-000000000002"
SESSION1 = "00000000-0000-4000-8000-000000000003"
SCOPE0 = EventQueryScope(run_id=RUN, session_id=SESSION0, seat=0)
SCOPE1 = EventQueryScope(run_id=RUN, session_id=SESSION1, seat=1)


def draft(*, seat=0, session=SESSION0, event_type="platform_received", payload=None):
    return EventDraft(
        schema_version="1.0", run_id=RUN, match_id=None, session_id=session, seat=seat,
        round_id=None, request_id=None, event_type=event_type,
        payload={"private_hand": ["S2"]} if payload is None else payload,
    )


class BlockingSink(FileEventSink):
    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.written = []
        self.close_count = 0
        self.summary_count = 0

    def write_event(self, content):
        if not self.written:
            self.entered.set()
            assert self.release.wait(3), "test must release the writer"
        self.written.append(json.loads(content))
        super().write_event(content)

    def close_events(self):
        self.close_count += 1
        super().close_events()
        self.finished.set()

    def write_summary(self, content):
        self.summary_count += 1
        super().write_summary(content)


async def create_blocked(tmp_path, *, queue_capacity=1):
    holder = []

    def factory(path):
        sink = BlockingSink(path)
        holder.append(sink)
        return sink

    store = await EventStore.create(
        tmp_path / "run", RUN, RunConfig(logs={"queue_capacity": queue_capacity}),
        clock=ManualClock(), sink_factory=factory,
    )
    return store, holder[0]


def test_round_trip_config_ids_pagination_and_seat_isolation(tmp_path):
    async def scenario():
        config = RunConfig()
        clock = ManualClock(utc_start="2026-10-19T02:00:00Z")
        store = await EventStore.create(tmp_path / "run", RUN, config, clock=clock)
        first = store.append(draft())
        second = store.append(draft(seat=1, session=SESSION1, payload={"private_hand": ["HA"]}))
        clock.advance(1)
        third = store.append(draft(payload={"private_hand": ["D2"]}))
        assert [first.seq, second.seq, third.seq] == [1, 2, 3]
        assert first.event_id == f"{RUN}:e1"
        page = store.query(SCOPE0, limit=1)
        assert page.next_after_seq == 1 and page.has_more
        assert page.items[0].recorded_at == "2026-10-19T02:00:00Z"
        next_page = store.query(SCOPE0, after_seq=page.next_after_seq, limit=1)
        assert next_page.next_after_seq == 3 and not next_page.has_more
        assert next_page.items[0].recorded_at == "2026-10-19T02:00:01Z"
        assert store.query(SCOPE1).items[0].payload["private_hand"] == ("HA",)
        assert store.query(EventQueryScope(run_id=RUN, session_id=SESSION0, seat=1)).items == ()
        assert store.query(SCOPE0, after_seq=3).next_after_seq is None
        summary = await store.close(RUN, pending_call_ids=("unknown-call",), unconfirmed_submission_ids=("unknown-write",))
        assert summary.closed and summary.log_complete
        assert summary.pending_call_ids == ("unknown-call",)
        assert summary.unconfirmed_submission_ids == ("unknown-write",)
        assert summary.last_persisted_seq == 4  # the recorder's final run_closed
        assert not (tmp_path / "run/summary.pending").exists()
        assert await store.close(RUN) == summary
        with pytest.raises(RecordingClosedError):
            store.append(draft())
        saved_config = json.loads((tmp_path / "run/config.json").read_text())
        assert saved_config["config_id"] == config.config_id
        assert saved_config["config"] == config.model_dump(mode="json")
        assert "DEEPSEEK_API_KEY" not in json.dumps(saved_config)
        loaded = await ClosedRunLog.load(tmp_path / "run")
        assert loaded.health == store.health
        assert [event.seq for event in loaded.query(SCOPE0).items] == [1, 3]
        assert len(loaded.offline_events()) == 4
        assert (tmp_path / "run/reports").is_dir()

    asyncio.run(scenario())


def test_bounded_queue_failure_keeps_assigned_id_and_persists_gap(tmp_path):
    async def scenario():
        store, sink = await create_blocked(tmp_path)
        first = store.append(draft())
        assert await asyncio.to_thread(sink.entered.wait, 1)
        second = store.append(draft())
        rejected = store.append(draft())
        assert first.accepted and second.accepted
        assert not rejected.accepted and rejected.seq == 3 and rejected.event_id == f"{RUN}:e3"
        assert rejected.error.code == "LOG_QUEUE_FULL"
        assert not store.health.log_complete
        assert [event.seq for event in store.query(SCOPE0).items] == [1, 2]
        sink.release.set()
        summary = await store.close(RUN)
        assert summary.closed and not summary.log_complete
        events = (await ClosedRunLog.load(tmp_path / "run")).offline_events()
        assert [event.seq for event in events] == [1, 2, 4, 5]
        gap = next(event for event in events if event.event_type == "log_gap")
        assert gap.payload["missing_seq_ranges"] == ((3, 3),)
        assert not any(event.seq == rejected.seq for event in events)
        assert sink.close_count == 1

    asyncio.run(scenario())


def test_close_timeout_does_not_block_loop_or_append_after_late_write(tmp_path):
    async def scenario():
        store, sink = await create_blocked(tmp_path)
        store.append(draft())
        assert await asyncio.to_thread(sink.entered.wait, 1)
        store.append(draft())  # queued but must never be written after timeout
        ticked = False

        async def tick():
            nonlocal ticked
            await asyncio.sleep(0)
            ticked = True

        tick_task = asyncio.create_task(tick())
        summary = await store.close(RUN, timeout_s=0.02)
        await tick_task
        assert ticked
        assert not summary.closed and not summary.log_complete
        assert summary.error.code == "LOG_CLOSE_TIMEOUT"
        assert not (tmp_path / "run/summary.json").exists()
        with pytest.raises(RecordingClosedError):
            store.append(draft())
        sink.release.set()
        assert await asyncio.to_thread(sink.finished.wait, 1)
        assert [event["seq"] for event in sink.written] == [1]
        assert sink.summary_count == 0 and sink.close_count == 1
        assert await store.close(RUN) == summary
        assert not store.health.closed
        loaded = await ClosedRunLog.load(tmp_path / "run")
        assert not loaded.health.closed and not loaded.health.log_complete
        with pytest.raises(RecordingClosedError):
            loaded.offline_events()

    asyncio.run(scenario())


def test_write_failure_safe_error_no_retry_and_no_invented_result(tmp_path):
    class BrokenSink(FileEventSink):
        attempts = 0

        def write_event(self, content):
            self.attempts += 1
            raise OSError("DO NOT EXPOSE SECRET PROVIDER RESPONSE")

    async def scenario():
        holder = []

        def factory(path):
            sink = BrokenSink(path)
            holder.append(sink)
            return sink

        store = await EventStore.create(tmp_path / "run", RUN, RunConfig(), sink_factory=factory)
        store.append(draft())
        store.append(draft())
        summary = await store.close(RUN)
        assert holder[0].attempts == 1
        assert summary.closed and not summary.log_complete
        assert summary.error.code == "LOG_WRITE_FAILED"
        assert "SECRET" not in summary.model_dump_json()
        assert summary.last_persisted_seq == 0
        loaded = await ClosedRunLog.load(tmp_path / "run")
        assert loaded.offline_events() == ()
        assert not loaded.health.log_complete
        assert "winner" not in summary.model_dump_json()

    asyncio.run(scenario())


def test_live_and_loaded_queries_reject_expired_window(tmp_path):
    async def scenario():
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig(logs={"cache_events": 2}))
        for _ in range(3):
            store.append(draft())
        with pytest.raises(HistoryUnavailable) as failure:
            store.query(SCOPE0)
        assert failure.value.error.code == "HISTORY_UNAVAILABLE"
        assert failure.value.retained_from_seq == 2
        assert [event.seq for event in store.query(SCOPE0, after_seq=1).items] == [2, 3]
        await store.close(RUN)
        loaded = await ClosedRunLog.load(tmp_path / "run", cache_events=2)
        with pytest.raises(HistoryUnavailable):
            loaded.query(SCOPE0)
        with pytest.raises(HistoryUnavailable):
            loaded.offline_events()
        assert [event.seq for event in loaded.query(SCOPE0, after_seq=2).items] == [3]

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["truncated_tail", "sequence_gap", "missing_summary", "bad_summary", "wrong_run", "duplicate_seq"])
def test_loader_marks_missing_evidence_incomplete_without_rewriting(tmp_path, case):
    async def scenario():
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig())
        store.append(draft())
        store.append(draft())
        await store.close(RUN)
        events_path = tmp_path / "run/events.jsonl"
        summary_path = tmp_path / "run/summary.json"
        lines = events_path.read_bytes().splitlines(keepends=True)
        if case == "truncated_tail":
            events_path.write_bytes(b"".join(lines) + b'{"unfinished":')
        elif case == "sequence_gap":
            events_path.write_bytes(b"".join(lines[1:]))
        elif case == "missing_summary":
            summary_path.unlink()
        elif case == "bad_summary":
            summary_path.write_text('{"closed":true}')
        elif case == "wrong_run":
            data = json.loads(lines[1])
            data["run_id"] = SESSION1
            data["event_id"] = f"{SESSION1}:e2"
            lines[1] = json.dumps(data).encode() + b"\n"
            events_path.write_bytes(b"".join(lines))
        elif case == "duplicate_seq":
            events_path.write_bytes(lines[0] + b"".join(lines))
        before_events = events_path.read_bytes()
        before_summary = summary_path.read_bytes() if summary_path.exists() else None
        loaded = await ClosedRunLog.load(tmp_path / "run")
        assert not loaded.health.log_complete
        assert not loaded.query(SCOPE0).log_complete
        if case in {"missing_summary", "bad_summary"}:
            assert not loaded.health.closed
            with pytest.raises(RecordingClosedError):
                loaded.offline_events()
        assert events_path.read_bytes() == before_events
        assert (summary_path.read_bytes() if summary_path.exists() else None) == before_summary

    asyncio.run(scenario())


def test_history_loaded_once_and_null_usage_preserved(tmp_path):
    async def scenario():
        unknown = json.loads((Path(__file__).parents[1] / "fixtures/recording/unknown-usage.json").read_text())
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig())
        store.append(draft(event_type="model_call_finished", payload=unknown))
        await store.close(RUN)
        loaded = await ClosedRunLog.load(tmp_path / "run")
        (tmp_path / "run/events.jsonl").unlink()
        payload = loaded.query(SCOPE0).items[0].payload
        assert payload["transport_requests"] is None
        assert all(value is None for value in payload["usage"].values())
        assert payload["decision"] is None
        assert loaded.health.log_complete
        assert loaded.query(SCOPE0).items == loaded.query(SCOPE0).items

    asyncio.run(scenario())


def test_run_directory_cannot_overwrite_prior_record(tmp_path):
    async def scenario():
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig())
        store.append(draft())
        await store.close(RUN)
        before = (tmp_path / "run/events.jsonl").read_bytes()
        with pytest.raises(RuntimeError, match="could not be created"):
            await EventStore.create(tmp_path / "run", RUN, RunConfig())
        assert (tmp_path / "run/events.jsonl").read_bytes() == before

    asyncio.run(scenario())


@pytest.mark.parametrize("after_seq,limit", [(-1, 50), (True, 50), (0, 0), (0, 201), (0, True)])
def test_query_validates_cursor_and_limit(tmp_path, after_seq, limit):
    async def scenario():
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig())
        with pytest.raises(ValueError):
            store.query(SCOPE0, after_seq=after_seq, limit=limit)
        with pytest.raises(TypeError):
            store.query(None)
        with pytest.raises(ValueError):
            store.query(EventQueryScope(run_id=SESSION1, session_id=SESSION0, seat=0))
        await store.close(RUN)

    asyncio.run(scenario())


def test_summary_stage_timeout_never_publishes_closed_marker(tmp_path):
    class SlowSummary(FileEventSink):
        entered = threading.Event()
        release = threading.Event()
        returned = threading.Event()

        def write_summary(self, content):
            self.entered.set()
            assert self.release.wait(3)
            super().write_summary(content)
            self.returned.set()

    async def scenario():
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig(), sink_factory=SlowSummary)
        closing = asyncio.create_task(store.close(RUN, timeout_s=0.05))
        assert await asyncio.to_thread(SlowSummary.entered.wait, 1)
        summary = await closing
        assert not summary.closed
        SlowSummary.release.set()
        assert await asyncio.to_thread(SlowSummary.returned.wait, 1)
        assert not (tmp_path / "run/summary.json").exists()
        loaded = await ClosedRunLog.load(tmp_path / "run")
        assert not loaded.health.closed

    asyncio.run(scenario())


def test_late_publish_cannot_restore_closed_state_or_trusted_historical_source(tmp_path):
    class SlowPublish(FileEventSink):
        entered = threading.Event()
        release = threading.Event()
        returned = threading.Event()

        def publish_summary(self):
            self.entered.set()
            assert self.release.wait(3)
            super().publish_summary()
            self.returned.set()

    async def scenario():
        store = await EventStore.create(tmp_path / "run", RUN, RunConfig(), sink_factory=SlowPublish)
        store.append(draft())
        closing = asyncio.create_task(store.close(RUN, timeout_s=0.05))
        assert await asyncio.to_thread(SlowPublish.entered.wait, 1)
        summary = await closing
        assert not summary.closed and not summary.log_complete
        assert summary.error.code == "LOG_CLOSE_TIMEOUT"
        SlowPublish.release.set()
        assert await asyncio.to_thread(SlowPublish.returned.wait, 1)
        assert (tmp_path / "run/summary.json").exists()
        assert (tmp_path / "run/summary.pending").exists()
        assert not store.health.closed and not store.health.log_complete
        assert await store.close(RUN) == summary
        loaded = await ClosedRunLog.load(tmp_path / "run")
        assert not loaded.health.closed and not loaded.health.log_complete
        assert not loaded.query(SCOPE0).log_complete
        with pytest.raises(RecordingClosedError):
            loaded.offline_events()

    asyncio.run(scenario())
