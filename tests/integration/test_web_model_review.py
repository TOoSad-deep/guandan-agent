"""Real offline runtime + fixed provider; web review never connects to a network."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from guandan_agent.config import RunConfig
from guandan_agent.recording.query import QueryService
from guandan_agent.recording.reports import ReportStore
from guandan_agent.web.api import create_web_app
from guandan_agent.web.review import WebReviews
from tests.integration.test_b09_offline import EvidenceProvider, copy_source, hashes
from tests.unit.test_offline_runtime import manager


ORIGIN = "http://127.0.0.1:8000"


def review_config(max_calls=2, prompt="b09-offline-v3"):
    return RunConfig.model_validate({
        "offline": {"max_model_calls": max_calls},
        "model": {"timeout_s": 1.0, "max_input_bytes": 24576},
        "versions": {"prompt": prompt},
    })


@asynccontextmanager
async def harness(tmp_path, *, provider=None, factory=None, max_calls=2, max_reports=100):
    source = copy_source(tmp_path)
    provider = provider or EvidenceProvider()
    runtime = manager(tmp_path / "unused", factory or (lambda _: provider))
    queries = QueryService(runtime)
    view = await queries.register_archive(source)
    reports = ReportStore(queries, max_reports_per_run=max_reports)
    reviews = WebReviews(runtime, queries, reports, review_config(max_calls), enabled=True)
    app = create_web_app(manager=runtime, queries=queries, reports=reports, allowed_configs={}, model_reviews=reviews)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as client:
        try:
            yield SimpleNamespace(source=source, manager=runtime, queries=queries, reports=reports,
                                  reviews=reviews, client=client, provider=provider, run_id=view.run_id,
                                  session_id=view.sessions[0].session_id,
                                  base=f"/api/v1/runs/{view.run_id}/reports")
        finally:
            await runtime.close_reviews()
            await reviews.close()


async def start(h):
    return await h.client.post(h.base + "/review", json={"session_id": h.session_id}, headers={"Origin": ORIGIN})


async def finished(h, task_id):
    async with asyncio.timeout(2):
        while True:
            response = await h.client.get(h.base + "/review-tasks/" + task_id)
            assert response.status_code == 200
            if response.json()["status"] != "running":
                return response.json()
            await asyncio.sleep(0.001)


def test_click_generates_one_v3_call_publishes_and_gets_never_start_work(tmp_path, monkeypatch):
    async def scenario():
        async with harness(tmp_path) as h:
            original = hashes(h.source)
            for _ in range(3):
                capability = (await h.client.get("/api/v1/configs")).json()["model_review"]
                assert capability["enabled"] and capability["remaining_calls"] == 2
                assert (await h.client.get(h.base)).json()["items"] == []
            assert h.provider.inputs == []
            response = await start(h)
            assert response.status_code == 202 and response.json()["status"] == "running"
            task_id = response.json()["task_id"]
            task = await finished(h, task_id)
            assert task["report_ready"] and task["status"] == "completed"
            assert task["report_id"] == task_id and task["calls_started"] == task["slots_used"] == 1
            assert len(h.provider.inputs) == 1
            payload = json.loads(h.provider.inputs[0].messages[-1].content)
            assert payload["prompt_version"] == "b09-offline-v3"
            assert 1 <= len(payload["decisions"]) <= 3
            report = h.reports.get(h.run_id, task_id)
            assert all(item.session_id == h.session_id and item.facts_verified for item in report.items)
            assert (h.source / "reports" / f"review-{task_id}.json").is_file()
            sidecar = json.loads((h.source / "reports" / f"review-{task_id}.execution.json").read_text())
            assert sidecar["offline_budget"]["slots_used"] == 1
            assert sidecar["task"]["calls_started"] == 1
            assert hashes(h.source) == original and h.manager.list_views() == ()
            with monkeypatch.context() as patch:
                def forbidden(*args, **kwargs):
                    raise AssertionError("GET cannot read files")
                patch.setattr(Path, "open", forbidden)
                patch.setattr(Path, "read_bytes", forbidden)
                patch.setattr(Path, "read_text", forbidden)
                for path in ("/api/v1/configs", h.base, h.base + "/" + task_id, h.base + "/review-tasks/" + task_id):
                    assert (await h.client.get(path)).status_code == 200
            assert len(h.provider.inputs) == 1
    asyncio.run(scenario())


def test_unpublished_task_blocks_second_click_and_cross_run_lookup(tmp_path, monkeypatch):
    from guandan_agent.web import review as module

    async def scenario():
        async with harness(tmp_path) as h:
            saving, release = asyncio.Event(), asyncio.Event()
            original = module.save_report

            async def slow_save(path, report):
                saving.set()
                await release.wait()
                return await original(path, report)

            monkeypatch.setattr(module, "save_report", slow_save)
            response = await start(h)
            task_id = response.json()["task_id"]
            await saving.wait()
            assert h.manager.review_view(task_id).report_ready
            state = h.reviews.state(h.run_id, task_id)
            assert state["status"] == "running" and not state["report_ready"] and state["report_id"] is None
            assert h.reviews.capabilities()["active_task"]["task_id"] == task_id
            duplicate = await start(h)
            assert duplicate.status_code == 409 and duplicate.json()["error"]["code"] == "REPORT_BUSY"
            foreign = await h.client.get(f"/api/v1/runs/{uuid4()}/reports/review-tasks/{task_id}")
            assert foreign.status_code == 404
            release.set()
            assert (await finished(h, task_id))["report_ready"]
            assert h.reviews.capabilities()["active_task"] is None
            assert len(h.provider.inputs) == 1
    asyncio.run(scenario())


def test_web_reviews_share_budget_and_exhaustion_rejects_before_new_task(tmp_path):
    async def scenario():
        async with harness(tmp_path, max_calls=2) as h:
            identifiers = []
            for _ in range(2):
                response = await start(h)
                assert response.status_code == 202
                identifiers.append(response.json()["task_id"])
                assert (await finished(h, identifiers[-1]))["calls_started"] == 1
            denied = await start(h)
            assert denied.status_code == 409 and denied.json()["error"]["code"] == "OFFLINE_BUDGET"
            assert h.reviews.capabilities()["remaining_calls"] == 0
            assert h.manager.offline_usage() == {"max_slots": 2, "slots_used": 2, "calls_started": 2}
            assert len(set(identifiers)) == len(h.provider.inputs) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["report", "execution", "register"])
def test_publication_failures_never_claim_report_ready(tmp_path, monkeypatch, failure):
    from guandan_agent.web import review as module

    async def scenario():
        async with harness(tmp_path) as h:
            async def fail(*args, **kwargs):
                raise OSError("sensitive-value")

            def fail_register(*args, **kwargs):
                raise ValueError("sensitive-value")

            if failure == "register":
                monkeypatch.setattr(h.reports, "register", fail_register)
            else:
                monkeypatch.setattr(module, "save_report" if failure == "report" else "save_review_execution", fail)
            response = await start(h)
            task = await finished(h, response.json()["task_id"])
            assert task["status"] == "failed" and not task["report_ready"] and task["report_id"] is None
            assert task["error"]["code"] == "REPORT_WRITE_FAILED"
            assert "sensitive-value" not in json.dumps(task)
            assert h.reviews.capabilities()["active_task"] is None
    asyncio.run(scenario())


def test_provider_initialization_failure_is_safe_and_persisted(tmp_path):
    async def scenario():
        def unavailable(_):
            raise RuntimeError("sensitive-value")
        async with harness(tmp_path, factory=unavailable) as h:
            response = await start(h)
            task = await finished(h, response.json()["task_id"])
            assert task["status"] == "failed" and task["report_ready"]
            assert task["calls_started"] == 0 and task["slots_used"] == 1
            assert task["error"]["code"] == "OFFLINE_PROVIDER_UNAVAILABLE"
            assert "sensitive-value" not in json.dumps(task)
            assert "模型配置" in task["error"]["safe_message"]
            assert (h.source / "reports" / f"review-{task['task_id']}.json").is_file()
    asyncio.run(scenario())


def test_manager_shutdown_is_persisted_before_web_publisher_closes(tmp_path):
    class PendingProvider(EvidenceProvider):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()

        async def complete(self, value):
            self.inputs.append(value)
            self.entered.set()
            await asyncio.Future()

    async def scenario():
        provider = PendingProvider()
        async with harness(tmp_path, provider=provider) as h:
            response = await start(h)
            task_id = response.json()["task_id"]
            await provider.entered.wait()
            await h.manager.close_reviews()
            await h.reviews.close()
            state = h.reviews.state(h.run_id, task_id)
            assert state["status"] == "partial" and state["report_ready"]
            assert state["error"]["code"] == "OFFLINE_CANCELLED" and state["calls_started"] == 1
            assert (h.source / "reports" / f"review-{task_id}.json").is_file()
            assert (h.source / "reports" / f"review-{task_id}.execution.json").is_file()
            assert not h.reviews.capabilities()["enabled"]
    asyncio.run(scenario())


def test_admission_rejects_foreign_session_symlink_and_full_shelf_without_provider(tmp_path):
    async def scenario():
        async with harness(tmp_path, max_reports=1) as h:
            foreign = await h.client.post(h.base + "/review", json={"session_id": str(uuid4())},
                                          headers={"Origin": ORIGIN})
            assert foreign.status_code == 404 and h.provider.inputs == []
            outside = tmp_path / "outside"
            outside.mkdir()
            (h.source / "reports").symlink_to(outside, target_is_directory=True)
            denied = await start(h)
            assert denied.status_code == 422 and h.provider.inputs == []
            (h.source / "reports").unlink()
            await h.reports.evaluate(h.run_id)
            denied = await start(h)
            assert denied.status_code == 409 and denied.json()["error"]["code"] == "REPORT_LIMIT"
            assert h.provider.inputs == [] and h.manager.offline_usage()["slots_used"] == 0
    asyncio.run(scenario())


def test_web_review_only_accepts_current_prompt(tmp_path):
    runtime = manager(tmp_path)
    queries = QueryService(runtime)
    with pytest.raises(ValueError, match="b09-offline-v3"):
        WebReviews(runtime, queries, ReportStore(queries), review_config(prompt="b09-offline-v2"), enabled=True)
