"""Generated report reads validate source ownership and evidence before linking."""

import asyncio
import json
from pathlib import Path
import shutil

import pytest

from guandan_agent.app import create_manager
from guandan_agent.config import RunConfig
from guandan_agent.contracts import EvaluationReport, ProviderReply, ReviewReport, Usage
from guandan_agent.recording.query import QueryError, QueryService
from guandan_agent.recording.reports import ReportStore


SOURCE = Path(__file__).resolve().parents[1] / "fixtures/audit/valid_scripted_run"


@pytest.fixture
def archive(tmp_path):
    target = tmp_path / "record"
    shutil.copytree(SOURCE, target)
    return target


def rewrite_events(path, mutation):
    target = path / "events.jsonl"
    events = [json.loads(line) for line in target.read_text().splitlines()]
    mutation(events)
    target.write_text("".join(json.dumps(e) + "\n" for e in events))


async def setup(path):
    queries = QueryService(create_manager(recording_root=path.parent / "unused"))
    view = await queries.register_archive(path)
    source = queries.closed_record(view.run_id)
    anchor = next(e for e in source.cached_events if e.event_type == "request_opened")
    return queries, ReportStore(queries), source, anchor


def report(source, anchor, *, evidence=None, status="completed", later_facts=(), report_id="fixed-review"):
    return ReviewReport.model_validate({
        "report_id": report_id, "run_id": source.run_id, "config_id": source.config_id,
        "status": status, "created_at": "2026-10-19T00:00:00Z",
        "items": ({"request_id": anchor.request_id, "session_id": anchor.session_id,
                   "evidence_event_ids": tuple(evidence if evidence is not None else [anchor.event_id]),
                   "at_time_facts": ("仅本人手牌可见",), "later_facts": later_facts,
                   "assessment": "<script>window.modelTextExecuted=true</script>",
                   "suggestion": "固定报告，不调用模型", "evidence_available": True},),
        "usage": Usage(), "errors": (),
    })


@pytest.mark.parametrize("status", ["completed", "partial", "failed"])
def test_report_status_and_safe_model_text_are_preserved(archive, status):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        original = report(source, anchor, status=status)
        shelf.register(source.run_id, original)
        stored = shelf.get(source.run_id, original.report_id)
        assert stored.status == status
        assert stored.items[0].evidence_available
        assert stored.items[0].evidence_error is None
        assert stored.items[0].facts_source == "model"
        assert stored.items[0].facts_verified is False
        assert stored.items[0].at_time_facts == original.items[0].at_time_facts
        assert stored.items[0].later_facts == original.items[0].later_facts
        assert stored.items[0].assessment == original.items[0].assessment
        assert shelf.list(source.run_id).items[0].status == status
    asyncio.run(scenario())


def test_empty_shelf_and_missing_report_are_explicit(archive):
    async def scenario():
        _, shelf, source, _ = await setup(archive)
        assert shelf.list(source.run_id).items == ()
        with pytest.raises(QueryError) as failure:
            shelf.get(source.run_id, "missing")
        assert failure.value.status_code == 404
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["none", "missing", "cross_session", "before_request", "other_private_request"])
def test_unusable_evidence_cannot_authorize_a_jump(archive, kind):
    if kind == "other_private_request":
        def mutate(events):
            anchor = next(e for e in events if e["event_type"] == "request_opened")
            later = next(e for e in events if e["session_id"] == anchor["session_id"]
                         and e["event_type"] == "platform_received" and e["seq"] > anchor["seq"])
            later.update(event_type="other_private_request", request_id=f"{anchor['session_id']}:q99", payload={})
        rewrite_events(archive, mutate)

    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        event = {
            "none": None,
            "missing": "missing-event",
            "cross_session": next(e.event_id for e in source.cached_events if e.session_id and e.session_id != anchor.session_id),
            "before_request": next(e.event_id for e in source.cached_events if e.session_id == anchor.session_id and e.seq < anchor.seq),
            "other_private_request": next((e.event_id for e in source.cached_events if e.event_type == "other_private_request"), None),
        }[kind]
        evidence = () if kind == "none" else (anchor.event_id, event)
        original = report(source, anchor, evidence=evidence, later_facts=("事后观察",))
        shelf.register(source.run_id, original)
        stored = shelf.get(source.run_id, original.report_id)
        assert stored.status == "partial"
        assert not stored.items[0].evidence_available
        assert stored.items[0].evidence_error.code == "EVIDENCE_UNAVAILABLE"
        assert original.status == "completed" and original.items[0].evidence_available
    asyncio.run(scenario())


@pytest.mark.parametrize("later_facts,anchor_reference,expected", [
    ((), True, False), (("事后观察",), False, False), (("事后观察",), True, True),
])
def test_later_public_fact_requires_label_and_direct_request_evidence(archive, later_facts, anchor_reference, expected):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        public = next(e for e in source.cached_events if e.session_id == anchor.session_id
                      and e.event_type == "platform_received" and e.seq > anchor.seq
                      and e.payload["event"]["kind"] == "play_observed")
        evidence = ((anchor.event_id,) if anchor_reference else ()) + (public.event_id,)
        value = report(source, anchor, evidence=evidence, later_facts=later_facts)
        shelf.register(source.run_id, value)
        assert shelf.get(source.run_id, value.report_id).items[0].evidence_available is expected
    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["foreign_session", "foreign_request", "missing_request", "incomplete_source"])
def test_report_item_ownership_and_incomplete_source_disable_jump(archive, mutation):
    if mutation == "incomplete_source":
        summary = json.loads((archive / "summary.json").read_text())
        summary["log_complete"] = False
        (archive / "summary.json").write_text(json.dumps(summary))

    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        value = report(source, anchor)
        item = value.items[0]
        other = next(e for e in source.cached_events if e.event_type == "request_opened" and e.session_id != anchor.session_id)
        updates = {"foreign_session": {"session_id": other.session_id},
                   "foreign_request": {"request_id": other.request_id},
                   "missing_request": {"request_id": f"{anchor.session_id}:q999"},
                   "incomplete_source": {}}[mutation]
        value = value.model_copy(update={"items": (item.model_copy(update=updates),)})
        shelf.register(source.run_id, value)
        checked = shelf.get(source.run_id, value.report_id)
        assert checked.status == "partial" and not checked.items[0].evidence_available
    asyncio.run(scenario())


@pytest.mark.parametrize("field,value", [("run_id", "00000000-0000-0000-0000-000000000001"), ("config_id", "wrong-config")])
def test_report_source_identity_must_match(archive, field, value):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        with pytest.raises(QueryError) as failure:
            shelf.register(source.run_id, report(source, anchor).model_copy(update={field: value}))
        assert failure.value.status_code == 422 and failure.value.code == "REPORT_OWNERSHIP"
        assert shelf.list(source.run_id).items == ()
    asyncio.run(scenario())


def test_malformed_recorded_event_does_not_crash_report_read(archive):
    def mutate(events):
        anchor = next(e for e in events if e["event_type"] == "request_opened")
        later = next(e for e in events if e["session_id"] == anchor["session_id"]
                     and e["event_type"] == "platform_received" and e["seq"] > anchor["seq"])
        later["payload"]["event"] = "malformed recorded event"
    rewrite_events(archive, mutate)

    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        malformed = next(e for e in source.cached_events if e.payload.get("event") == "malformed recorded event")
        value = report(source, anchor, evidence=(anchor.event_id, malformed.event_id), later_facts=("事后观察",))
        shelf.register(source.run_id, value)
        checked = shelf.get(source.run_id, value.report_id)
        assert checked.status == "partial" and not checked.items[0].evidence_available
    asyncio.run(scenario())


def test_directory_load_ignores_audits_and_reports_partial_loading(archive, tmp_path):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        folder = tmp_path / "reports"
        folder.mkdir()
        valid = report(source, anchor)
        (folder / "review.json").write_text(valid.model_dump_json())
        (folder / "audit.json").write_text('{"passed":true}')
        (folder / "broken.json").write_text("not json")
        (folder / "wrong.json").write_text(valid.model_copy(update={"report_id": "wrong", "config_id": "wrong"}).model_dump_json())
        (folder / "link.json").symlink_to(folder / "review.json")
        await shelf.load_directory(source.run_id, folder)
        page = shelf.list(source.run_id)
        assert [item.report_id for item in page.items] == [valid.report_id]
        assert len(page.errors) == 1 and page.errors[0].code == "REPORT_UNAVAILABLE"
        await shelf.load_directory(source.run_id, folder)
        assert shelf.list(source.run_id).errors == page.errors
        for name in ("broken.json", "wrong.json", "link.json"):
            (folder / name).unlink()
        await shelf.load_directory(source.run_id, folder)
        assert shelf.list(source.run_id).errors == ()
    asyncio.run(scenario())


def test_execution_sidecars_do_not_consume_report_capacity(archive):
    async def scenario():
        queries, _, source, anchor = await setup(archive)
        shelf = ReportStore(queries, max_reports_per_run=2)
        folder = archive / "reports"
        folder.mkdir()
        for index in range(2):
            value = report(source, anchor, report_id=f"review-{index}")
            (folder / f"review-{index}.json").write_text(value.model_dump_json())
            (folder / f"review-{index}.execution.json").write_text('{"kind":"offline_execution"}')
        page = await shelf.refresh(source.run_id)
        assert len(page.items) == 2
        assert page.errors == ()
    asyncio.run(scenario())


def test_evaluation_is_bounded_deduplicated_and_published_after_save(archive, monkeypatch):
    from guandan_agent.offline import output

    async def scenario():
        _, shelf, source, _ = await setup(archive)
        started, release = asyncio.Event(), asyncio.Event()
        original = output.save_report
        saved = []

        async def slow_save(path, value):
            saved.append(path)
            started.set()
            await release.wait()
            return await original(path, value)

        monkeypatch.setattr(output, "save_report", slow_save)
        first = asyncio.create_task(shelf.evaluate(source.run_id))
        await started.wait()
        assert shelf.list(source.run_id).items == ()
        with pytest.raises(QueryError) as failure:
            await shelf.evaluate(source.run_id)
        assert failure.value.code == "REPORT_BUSY"
        release.set()
        value = await first
        assert value.kind == "evaluation" and value.run_ids == (source.run_id,)
        assert shelf.get(source.run_id, value.report_id) == value
        assert await shelf.evaluate(source.run_id) == value
        assert saved == [archive.resolve() / "reports" / f"evaluation-{value.report_id}.json"]
        assert len(tuple((archive / "reports").glob("evaluation-*.json"))) == 1
    asyncio.run(scenario())


def test_evaluation_rejects_symlink_report_destination(archive, tmp_path):
    async def scenario():
        _, shelf, source, _ = await setup(archive)
        outside = tmp_path / "outside"
        outside.mkdir()
        (archive / "reports").symlink_to(outside, target_is_directory=True)
        with pytest.raises(QueryError) as failure:
            await shelf.evaluate(source.run_id)
        assert failure.value.code == "REPORT_DIRECTORY_UNAVAILABLE"
        assert tuple(outside.iterdir()) == ()
    asyncio.run(scenario())


def test_evaluation_rejects_evicted_archive_without_writing_partial_totals(archive):
    async def scenario():
        queries = QueryService(create_manager(recording_root=archive.parent / "unused"), cache_events=1)
        view = await queries.register_archive(archive)
        shelf = ReportStore(queries)
        with pytest.raises(QueryError) as failure:
            await shelf.evaluate(view.run_id)
        assert failure.value.code == "HISTORY_UNAVAILABLE" and failure.value.status_code == 410
        assert not (archive / "reports").exists()
    asyncio.run(scenario())


def test_evaluation_pairs_must_reference_loaded_sources(archive):
    async def scenario():
        _, shelf, source, _ = await setup(archive)
        value = EvaluationReport.model_validate({
            "report_id": "evaluation", "run_ids": (source.run_id,), "config_ids": (source.config_id,),
            "status": "completed", "created_at": "2026-10-19T00:00:00Z", "unit": "match",
            **{key: {} for key in ("cohort", "counts", "performance", "latency", "model_usage", "quality")},
        })
        shelf.register(source.run_id, value)
        assert shelf.get(source.run_id, value.report_id) == value
        with pytest.raises(QueryError) as failure:
            shelf.register(source.run_id, value.model_copy(update={"config_ids": ("wrong",)}))
        assert failure.value.code == "REPORT_OWNERSHIP"
    asyncio.run(scenario())


async def program_report(source, anchor):
    """Exercise the real facts builder, with no provider or network access."""
    from guandan_agent.offline.review import generate_review

    class Calls:
        task_id = "program-review"

        async def call(self, messages):
            decisions = json.loads(messages[-1].content)["decisions"]
            output = {"items": [{
                "request_id": item["request_id"],
                "evidence_event_ids": [item["anchor_event_id"]],
                "assessment": "这是模型分析，不能视为已验证策略。",
                "suggestion": "可在相同条件下比较候选动作。",
            } for item in decisions]}
            return ProviderReply(call_id="fixed-test-call", content=json.dumps(output),
                                 usage=Usage(), transport_requests=0, elapsed_ms=0.0, error=None)

    return await generate_review(source, RunConfig(), Calls(), report_id="program-review",
                                 created_at="2026-10-19T00:00:00Z",
                                 session_id=anchor.session_id, request_ids=(anchor.request_id,))


def test_program_facts_are_recomputed_without_trusting_serialized_flag(archive):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        original = await program_report(source, anchor)
        assert original.status == "completed"
        item = original.items[0].model_copy(update={"facts_verified": False})
        original = original.model_copy(update={"items": (item,)})
        shelf.register(source.run_id, original)
        stored = shelf.get(source.run_id, original.report_id)
        assert stored.items[0].facts_source == "program"
        assert stored.items[0].facts_verified is True
        assert stored.items[0].evidence_available is True
        assert stored.items[0].at_time_facts == item.at_time_facts
        assert stored.items[0].later_facts == item.later_facts
        assert stored.items[0].assessment == item.assessment
        assert original.items[0].facts_verified is False
    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["text", "later_text", "required_reference", "version", "model_flag"])
def test_forged_program_facts_downgrade_report_without_rewriting_text(archive, mutation):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        original = await program_report(source, anchor)
        item = original.items[0]
        assert item.facts_verified is True
        if mutation == "text":
            item = item.model_copy(update={"at_time_facts": ("伪造：SB 是大王。",)})
        elif mutation == "later_text":
            item = item.model_copy(update={"later_facts": ("伪造：我方已经获胜。",)})
        elif mutation == "required_reference":
            assert len(item.evidence_event_ids) > 1
            item = item.model_copy(update={"evidence_event_ids": (anchor.event_id,)})
        elif mutation == "version":
            original = original.model_copy(update={"generation": {"prompt_version": "b09-offline-v1"}})
        else:
            item = item.model_copy(update={"facts_source": "model"})
        original = original.model_copy(update={"items": (item,)})
        shelf.register(source.run_id, original)
        stored = shelf.get(source.run_id, original.report_id)
        assert stored.status == "partial"
        assert stored.items[0].facts_verified is False
        assert stored.items[0].at_time_facts == item.at_time_facts
        assert stored.items[0].later_facts == item.later_facts
        assert any(error.code == "PROGRAM_FACTS_UNVERIFIED" for error in stored.errors)
        assert original.items[0].facts_verified is True
    asyncio.run(scenario())


def test_prompt_version_alone_does_not_relabel_legacy_report_or_modify_archive(archive, tmp_path):
    async def scenario():
        _, shelf, source, anchor = await setup(archive)
        original = report(source, anchor).model_copy(update={"generation": {"prompt_version": "b09-offline-v2"}})
        folder = tmp_path / "reports"
        folder.mkdir()
        path = folder / "legacy.json"
        raw = json.loads(original.model_dump_json())
        raw["items"][0].pop("facts_source")
        raw["items"][0].pop("facts_verified")
        encoded = json.dumps(raw, ensure_ascii=False).encode()
        path.write_bytes(encoded)
        await shelf.load_directory(source.run_id, folder)
        stored = shelf.get(source.run_id, original.report_id)
        assert stored.status == "completed"
        assert stored.items[0].facts_source == "model"
        assert stored.items[0].facts_verified is False
        assert stored.items[0].at_time_facts == original.items[0].at_time_facts
        assert stored.items[0].assessment == original.items[0].assessment
        assert path.read_bytes() == encoded
    asyncio.run(scenario())
