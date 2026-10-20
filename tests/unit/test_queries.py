"""Read projections never replay decisions or enrich a seat with hidden cards."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.hand import HandTools
from guandan_agent.model.scripted import ScriptedProvider
from guandan_agent.recording.query import QueryError, QueryService
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler
from tests.support.scripted_room import ScriptedRooms

FIXTURE = Path(__file__).parents[1] / "fixtures/audit/valid_scripted_run"


def manager(root):
    clock = SystemClock()
    def forbidden(_):
        raise AssertionError("read projections must not construct transports")
    return RunManager(clock=clock, scheduler=AsyncioScheduler(clock),
                      recording_root=root, platform_factory=forbidden, provider_factory=forbidden)


def copy_archive(tmp_path):
    target = tmp_path / "archive"
    shutil.copytree(FIXTURE, target)
    return target


def rows(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def rewrite(path, events, *, renumber=False):
    if renumber:
        for seq, event in enumerate(events, 1):
            event["seq"] = seq
            event["event_id"] = f'{event["run_id"]}:e{seq}'
    (path / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    summary = json.loads((path / "summary.json").read_text())
    summary["last_persisted_seq"] = events[-1]["seq"]
    (path / "summary.json").write_text(json.dumps(summary))


def assert_error(status, call):
    with pytest.raises(QueryError) as caught:
        call()
    assert caught.value.status_code == status
    assert caught.value.safe_message
    return caught.value


def test_archive_read_is_one_load_and_private_facts_are_whitelisted(tmp_path, monkeypatch):
    async def scenario():
        source = copy_archive(tmp_path)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(source)
        assert run.status == "completed" and run.log_complete
        assert len(run.sessions) == 8
        first = run.sessions[0]
        page = query.decisions(run.run_id, first.session_id)
        detail = query.decision(run.run_id, first.session_id, page.items[0].request_id)
        assert detail.state.self_hand == ("S8",)
        assert detail.execution.status == "confirmed"
        assert detail.candidates is None and detail.calls == ()
        assert detail.proposal.source == "unique" and detail.review is None
        rendered = detail.model_dump_json()
        assert all(word not in rendered for word in ("restCards", "raw_message", "actionList", "S9", "ST", "SJ"))
        state = query.session_view(run.run_id, first.session_id)
        assert state.state.stage == "settled" and state.state.self_hand == ()
        assert state.active_request_id is None
        assert "restCards" not in state.model_dump_json()
        assert query.list_runs()[0].read_only
        assert query.closed_record(run.run_id).health.closed
        monkeypatch.setattr(Path, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GET disk read")))
        for _ in range(3):
            assert query.view(run.run_id) == run
            assert query.session_view(run.run_id, first.session_id) == state
            assert query.decisions(run.run_id, first.session_id) == page
            assert query.decision(run.run_id, first.session_id, page.items[0].request_id) == detail
        assert await query.register_archive(source) == run
        monkeypatch.undo()
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
        assert after == before
    asyncio.run(scenario())


def test_missing_and_foreign_identifiers_never_cross_sessions(tmp_path):
    async def scenario():
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(FIXTURE)
        first, other = run.sessions[:2]
        request = query.decisions(run.run_id, first.session_id).items[0].request_id
        assert_error(404, lambda: query.view(str(uuid4())))
        assert_error(404, lambda: query.session_view(run.run_id, str(uuid4())))
        assert_error(404, lambda: query.decision(run.run_id, other.session_id, request))
        assert_error(404, lambda: query.decision(run.run_id, first.session_id, first.session_id + ":q999"))
        assert_error(404, lambda: query.decisions(str(uuid4()), first.session_id))
        for after, limit in [(-1, 50), (True, 50), (0, 0), (0, 201), (0, True)]:
            assert_error(422, lambda: query.decisions(run.run_id, first.session_id, after_seq=after, limit=limit))
    asyncio.run(scenario())


def test_decision_cursor_is_request_opened_seq_not_last_execution_seq(tmp_path):
    async def scenario():
        source = copy_archive(tmp_path)
        events = rows(source)
        anchor = next(event for event in events if event["event_type"] == "request_opened")
        first_request = anchor["request_id"]
        extension = [json.loads(json.dumps(event).replace(first_request, first_request[:-1] + "2"))
                     for event in events if event["request_id"] == first_request]
        events[-1:-1] = extension
        rewrite(source, events, renumber=True)
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(source)
        sid = anchor["session_id"]
        page = query.decisions(run.run_id, sid, limit=1)
        assert len(page.items) == 1 and page.has_more
        assert page.next_after_seq == anchor["seq"]
        second = query.decisions(run.run_id, sid, after_seq=page.next_after_seq, limit=1)
        assert len(second.items) == 1 and not second.has_more
        assert second.items[0].request_id.endswith(":q2")
        assert second.next_after_seq > page.next_after_seq
        end = query.decisions(run.run_id, sid, after_seq=second.next_after_seq)
        assert end.items == () and end.next_after_seq is None
    asyncio.run(scenario())


def test_partial_record_exposes_known_data_and_blocks_unclosed_reports(tmp_path):
    async def scenario():
        source = copy_archive(tmp_path)
        (source / "summary.pending").write_text("not a close")
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(source)
        assert not run.log_complete
        first = run.sessions[0]
        page = query.decisions(run.run_id, first.session_id)
        assert not page.log_complete and not page.items[0].log_complete
        detail = query.decision(run.run_id, first.session_id, page.items[0].request_id)
        assert detail.state.self_hand == ("S8",) and not detail.log_complete
        assert not query.session_view(run.run_id, first.session_id).log_complete
        assert_error(409, lambda: query.closed_record(run.run_id))
    asyncio.run(scenario())


def test_gap_keeps_partial_flag_but_missing_context_is_410(tmp_path):
    async def scenario():
        source = copy_archive(tmp_path)
        events = rows(source)
        anchor = next(event for event in events if event["event_type"] == "request_opened")
        rewrite(source, [event for event in events if event is not anchor])
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(source)
        assert not run.log_complete
        assert_error(410, lambda: query.decision(run.run_id, anchor["session_id"], anchor["request_id"]))
        assert_error(410, lambda: query.decisions(run.run_id, anchor["session_id"]))
    asyncio.run(scenario())


def test_evicted_cache_cursor_returns_410_and_retained_request_can_be_read(tmp_path):
    async def scenario():
        query = QueryService(manager(tmp_path), cache_events=100)
        run = await query.register_archive(FIXTURE)
        latest = run.sessions[-4]
        assert_error(410, lambda: query.decisions(run.run_id, latest.session_id))
        source = query.closed_record(run.run_id)
        page = query.decisions(run.run_id, latest.session_id, after_seq=source.retained_from_seq - 1)
        assert page.items
        detail = query.decision(run.run_id, latest.session_id, page.items[0].request_id)
        assert detail.state.binding.session_id == latest.session_id
    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["state", "proposal", "execution"])
def test_nested_foreign_binding_is_not_a_query_result(tmp_path, field):
    async def scenario():
        source = copy_archive(tmp_path)
        events = rows(source)
        anchor = next(event for event in events if event["event_type"] == "request_opened")
        target = next(event for event in events if event["request_id"] == anchor["request_id"] and
                      (event["event_type"] == "request_opened" if field == "state" else field in event["payload"]))
        payload = target["payload"]["context"]["state"] if field == "state" else target["payload"][field]
        payload["binding"]["seat"] = 3
        rewrite(source, events)
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(source)
        assert_error(410, lambda: query.decision(run.run_id, anchor["session_id"], anchor["request_id"]))
    asyncio.run(scenario())


def test_live_and_completed_manager_queries_never_add_calls_or_submissions(tmp_path):
    async def scenario():
        clock = SystemClock()
        rooms = ScriptedRooms(clock, pair_options=True)
        provider = ScriptedProvider(['{"action_id":"L0001","reason":"选择候选"}'], clock=clock)
        live = RunManager(clock=clock, scheduler=AsyncioScheduler(clock), recording_root=tmp_path,
            platform_factory=rooms.factory, provider_factory=lambda _: provider,
            tools_factory=lambda _: HandTools(clock=clock))
        query = QueryService(live)
        run = live.start(RunConfig.model_validate({"run": {"policy": "V1"},
            "batch": {"max_model_calls": 1}, "tools": {"deep_eval_limit": 0}}))
        assert query.list_runs()[0].read_only is False
        assert_error(409, lambda: query.closed_record(run.run_id))
        final = await asyncio.wait_for(live.wait(run.run_id), 4)
        assert live.is_closed(run.run_id)
        assert live.archive_path(run.run_id) == tmp_path / run.run_id
        first = final.sessions[0]
        page = query.decisions(run.run_id, first.session_id)
        detail = query.decision(run.run_id, first.session_id, page.items[0].request_id)
        assert detail.candidates is not None and len(detail.calls) == 1
        assert detail.calls[0].usage.total_tokens is None
        assert detail.execution.status == "confirmed"
        before = (len(provider.inputs), len(rooms.writes), len(live.query_snapshot(run.run_id, first.session_id)[0]))
        for _ in range(3):
            query.view(run.run_id)
            query.session_view(run.run_id, first.session_id)
            query.decisions(run.run_id, first.session_id)
            query.decision(run.run_id, first.session_id, page.items[0].request_id)
        assert before == (len(provider.inputs), len(rooms.writes), len(live.query_snapshot(run.run_id, first.session_id)[0]))
        assert_error(409, lambda: query.closed_record(run.run_id))
        await query.register_archive(live.archive_path(run.run_id))
        assert query.closed_record(run.run_id).health.closed
        scoped = live.query_events(run.run_id, first.session_id, limit=200)
        assert scoped.items and all(event.seat == first.seat for event in scoped.items)
        assert_error(404, lambda: query.decisions(run.run_id, str(uuid4())))
    asyncio.run(scenario())


@pytest.mark.parametrize("main,second,expected", [
    ('{"action_id":"L0002","reason":"主建议"}', '{"action_id":"L0001","reason":"复核建议"}', "changed"),
    ('{"action_id":"L0002","reason":"主建议"}', 'bad-json', "failed"),
    ('bad-json', '{"action_id":"L0001","reason":"修复建议"}', "repaired"),
])
def test_review_repair_and_unknown_usage_are_projected_from_recorded_results(tmp_path, main, second, expected):
    from guandan_agent.config import load_config
    from scripts import probe_b07_review

    async def scenario():
        provider = ScriptedProvider([main, second])
        config = load_config(Path(__file__).parents[2] / "configs/b07-v2-fixed.toml")
        result = await probe_b07_review.run_probe(config, mode="scripted", recording_root=tmp_path,
                                                 provider_factory=lambda _: provider)
        query = QueryService(manager(tmp_path))
        run = await query.register_archive(tmp_path / result["run_id"])
        first = run.sessions[0]
        page = query.decisions(run.run_id, first.session_id)
        detail = query.decision(run.run_id, first.session_id, page.items[0].request_id)
        assert detail.review.status == expected
        assert len(detail.calls) == 2
        assert all(call.usage.total_tokens is None for call in detail.calls)
        assert detail.proposal.review == detail.review
        assert detail.execution.status == "confirmed"
        before = len(provider.inputs)
        assert query.decision(run.run_id, first.session_id, page.items[0].request_id) == detail
        assert len(provider.inputs) == before
    asyncio.run(scenario())
