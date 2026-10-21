"""B08 composes real query/report stores over a fixed log, without transports."""

import asyncio
import json
import os
from pathlib import Path
import shutil

import httpx

from guandan_agent import app as composition
from guandan_agent.cli import main
from guandan_agent.config import RunConfig
from guandan_agent.recording.query import QueryError, QueryService
from tests.unit.test_reports import SOURCE, report


ORIGIN = "http://127.0.0.1:8000"


def prepare(tmp_path):
    run_id = json.loads((SOURCE / "summary.json").read_text())["run_id"]
    archive_root = tmp_path / "archives"
    archive = archive_root / run_id
    shutil.copytree(SOURCE, archive)
    shutil.copytree(SOURCE, archive_root / "ignored-non-uuid")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "local.example.toml").write_text("[batch]\nmax_model_calls = 0\n[tools]\ndeep_eval_limit = 0\n")
    return run_id, archive_root, configs


def forbidden(*args, **kwargs):
    raise AssertionError("Read-only web work must not construct transports or read additional files")


def forbid_credentials_and_providers(monkeypatch):
    original = composition.os.environ.get

    def guarded(key, default=None):
        if key == "DEEPSEEK_API_KEY":
            raise AssertionError("Read-only serving must not read provider credentials")
        return original(key, default)

    monkeypatch.setattr(composition.os.environ, "get", guarded)
    monkeypatch.setattr("guandan_agent.model.deepseek.DeepSeekProvider", forbidden)
    monkeypatch.setattr(composition, "OpenGuanDanPlatform", forbidden)


def test_closed_record_http_views_reports_and_repeated_gets_are_read_only(tmp_path, monkeypatch):
    run_id, archive_root, configs = prepare(tmp_path)
    forbid_credentials_and_providers(monkeypatch)
    application = composition.create_web_application(recording_root=tmp_path / "live", config_root=configs, archive_root=archive_root)

    async def scenario():
        async with application.router.lifespan_context(application):
            assert application.state.archive_errors == []
            manager, queries, reports = application.state.manager, application.state.queries, application.state.reports
            assert manager.list_views() == ()
            source = queries.closed_record(run_id)
            anchors = [event for event in source.cached_events if event.event_type == "request_opened"]
            for status in ("completed", "partial", "failed"):
                reports.register(run_id, report(source, anchors[0], status=status, report_id=status))
            bad = report(source, anchors[0], evidence=(anchors[1].event_id,), report_id="bad-evidence")
            reports.register(run_id, bad)
            # GET must only project the already loaded in-memory record/shelf.
            with monkeypatch.context() as patch:
                patch.setattr(Path, "open", forbidden)
                patch.setattr(Path, "read_text", forbidden)
                patch.setattr(Path, "read_bytes", forbidden)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url=ORIGIN) as client:
                    for _ in range(3):
                        runs = await client.get("/api/v1/runs")
                        assert runs.status_code == 200
                        assert [row["run_id"] for row in runs.json()["items"]] == [run_id]
                        assert runs.json()["items"][0]["read_only"]
                        pages = []
                        for anchor in anchors[:3]:
                            base = f"/api/v1/runs/{run_id}/sessions/{anchor.session_id}"
                            page = await client.get(f"{base}/decisions")
                            assert page.status_code == 200
                            pages.append(page.json())
                            detail = await client.get(f"{base}/decisions/{anchor.request_id}")
                            assert detail.status_code == 200
                            assert detail.json()["state"]["self_hand"] == list(anchor.payload["context"]["state"]["self_hand"])
                            assert detail.json()["state"]["binding"]["seat"] == anchor.seat
                            assert "raw_message" not in detail.text and "restCards" not in detail.text
                            state = await client.get(f"{base}/state")
                            assert state.status_code == 200 and state.json()["session_id"] == anchor.session_id
                            assert len(page.json()["items"]) == 1
                        assert len({p["items"][0]["request_id"] for p in pages}) == 3
                        foreign = await client.get(f"/api/v1/runs/{run_id}/sessions/{anchors[0].session_id}/decisions/{anchors[1].request_id}")
                        assert foreign.status_code == 404
                        summaries = await client.get(f"/api/v1/runs/{run_id}/reports")
                        assert summaries.status_code == 200 and len(summaries.json()["items"]) == 4
                        for status in ("completed", "partial", "failed"):
                            value = await client.get(f"/api/v1/runs/{run_id}/reports/{status}")
                            assert value.status_code == 200 and value.json()["status"] == status
                            assert value.headers["content-type"] == "application/json"
                            assert "<script>" in value.json()["items"][0]["assessment"]
                        bad_value = await client.get(f"/api/v1/runs/{run_id}/reports/bad-evidence")
                        assert bad_value.json()["status"] == "partial"
                        assert not bad_value.json()["items"][0]["evidence_available"]
            assert manager.list_views() == ()
        assert not (tmp_path / "live").exists()
    asyncio.run(scenario())


def test_lifespan_preloads_once_and_stops_owned_pending_run(tmp_path, monkeypatch):
    run_id, archive_root, configs = prepare(tmp_path)
    forbid_credentials_and_providers(monkeypatch)
    application = composition.create_web_application(recording_root=tmp_path / "live", config_root=configs, archive_root=archive_root)

    async def scenario():
        manager = application.state.manager
        async with application.router.lifespan_context(application):
            queries = application.state.queries
            first = queries.closed_record(run_id)
            (archive_root / run_id / "summary.json").write_text("damaged after startup")
            assert queries.closed_record(run_id) is first
            started = manager.start(RunConfig.model_validate({"tools": {"deep_eval_limit": 0}, "batch": {"max_model_calls": 0}}))
            assert started.status == "starting"
        final = manager.view(started.run_id)
        assert final.status == "interrupted"
        assert final.error.code == "RUN_STOPPED"
        assert final.sessions == ()
    asyncio.run(scenario())


def test_new_report_refresh_and_statistics_generation_preserve_source_and_get_boundary(tmp_path, monkeypatch):
    run_id, archive_root, configs = prepare(tmp_path)
    archive = archive_root / run_id
    original = {name: (archive / name).read_bytes() for name in ("config.json", "events.jsonl", "summary.json")}
    forbid_credentials_and_providers(monkeypatch)
    application = composition.create_web_application(recording_root=tmp_path / "live", config_root=configs,
                                                      archive_root=archive_root)

    async def scenario():
        async with application.router.lifespan_context(application):
            source = application.state.queries.closed_record(run_id)
            anchor = next(event for event in source.cached_events if event.event_type == "request_opened")
            folder = archive / "reports"
            folder.mkdir()
            (folder / "new-review.json").write_text(report(source, anchor).model_dump_json())
            base = f"/api/v1/runs/{run_id}/reports"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url=ORIGIN) as client:
                assert (await client.get(base)).json()["items"] == []
                refreshed = await client.post(base + "/refresh", json={}, headers={"Origin": ORIGIN})
                assert refreshed.status_code == 200
                assert [item["report_id"] for item in refreshed.json()["items"]] == ["fixed-review"]
                generated = await client.post(base + "/evaluate", json={}, headers={"Origin": ORIGIN})
                assert generated.status_code == 200
                value = generated.json()
                assert value["kind"] == "evaluation" and value["run_ids"] == [run_id]
                assert value["counts"]["started"] == 2
                persisted = json.loads((folder / f"evaluation-{value['report_id']}.json").read_text())
                assert persisted == value
                repeated = await client.post(base + "/evaluate", json={}, headers={"Origin": ORIGIN})
                assert repeated.json() == value
                with monkeypatch.context() as patch:
                    patch.setattr(Path, "open", forbidden)
                    patch.setattr(Path, "read_text", forbidden)
                    patch.setattr(Path, "read_bytes", forbidden)
                    assert len((await client.get(base)).json()["items"]) == 2
                    assert (await client.get(base + "/" + value["report_id"])).json() == value
            assert application.state.manager.list_views() == ()
        assert original == {name: (archive / name).read_bytes() for name in original}
        assert not (tmp_path / "live").exists()
    asyncio.run(scenario())


def test_archive_startup_chooses_eight_most_recent_summaries_not_uuid_order(tmp_path, monkeypatch):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    expected = []
    for index in range(10):
        # The most recent run deliberately has the lexically smallest UUID.
        path = archive_root / f"00000000-0000-0000-0000-{9 - index:012d}"
        path.mkdir()
        summary = path / "summary.json"
        summary.write_text("{}")
        os.utime(summary, ns=(1_000_000_000 + index, 1_000_000_000 + index))
        expected.append(path)
    attempted = []

    async def record_attempt(self, path):
        attempted.append(path)
        raise QueryError("ARCHIVE_INVALID", 422, "Test snapshot is intentionally unavailable.")

    monkeypatch.setattr(QueryService, "register_archive", record_attempt)
    application = composition.create_web_application(recording_root=tmp_path / "live", allowed_config_names=(),
                                                      archive_root=archive_root)

    async def scenario():
        async with application.router.lifespan_context(application):
            assert attempted == expected[-8:]
            assert len(application.state.archive_errors) == 8
    asyncio.run(scenario())


def test_cli_serve_forces_one_worker_no_reload_and_no_run_start(tmp_path, monkeypatch):
    _, archive_root, configs = prepare(tmp_path)
    calls = []
    forbid_credentials_and_providers(monkeypatch)
    monkeypatch.setattr("uvicorn.run", lambda application, **kwargs: calls.append((application, kwargs)))
    status = main(["serve", "--config-root", str(configs), "--archive-root", str(archive_root), "--recording-root", str(tmp_path / "live"), "--port", "8000"])
    assert status == 0 and len(calls) == 1
    application, kwargs = calls[0]
    assert kwargs == {"host": "127.0.0.1", "port": 8000, "workers": 1, "reload": False, "access_log": False}
    assert application.state.manager.list_views() == ()
    assert not (tmp_path / "live").exists()
