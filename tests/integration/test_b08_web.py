"""B08 composes real query/report stores over a fixed log, without transports."""

import asyncio
import json
from pathlib import Path
import shutil

import httpx

from guandan_agent import app as composition
from guandan_agent.cli import main
from guandan_agent.config import RunConfig
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
