"""Closed-source report generation, publication and HTTP reading as one path."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil

import httpx
import pytest

from guandan_agent import app, cli
from guandan_agent.clock import SystemClock
from guandan_agent.config import load_config
from guandan_agent.contracts import ProviderReply, Usage
from guandan_agent.offline.output import save_report
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/fixtures/audit/valid_scripted_run"
CONFIG = ROOT / "configs/b09-review.toml"


class EvidenceProvider:
    def __init__(self):
        self.inputs = []
        self.closed = False

    async def complete(self, value):
        self.inputs.append(value)
        payload = json.loads(value.messages[1].content)
        decisions = payload["decisions"]
        items = [{"request_id": d["request_id"], "evidence_event_ids": [d["anchor_event_id"]],
                  "at_time_facts": ["仅使用该请求本人手牌和公开张数。"], "later_facts": [],
                  "assessment": "固定响应用于验证引用与数据交接。", "suggestion": "对照原动作检查其他合法选项。"}
                 for d in decisions]
        if payload.get("prompt_version") in {"b09-offline-v2", "b09-offline-v3"}:
            for item in items:
                item.pop("at_time_facts")
                item.pop("later_facts")
        return ProviderReply(call_id=value.call_id, content=json.dumps({"items": items}),
                             transport_requests=0, elapsed_ms=1.0,
                             usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30), error=None)

    async def aclose(self):
        self.closed = True


def copy_source(tmp_path):
    identifier = json.loads((SOURCE / "config.json").read_text())["run_id"]
    target = tmp_path / identifier
    shutil.copytree(SOURCE, target)
    return target


def hashes(source):
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("config.json", "events.jsonl", "summary.json")}


def test_review_check_has_no_tasks_credentials_or_output(tmp_path, monkeypatch):
    source = copy_source(tmp_path)
    def forbidden(**kwargs):
        raise AssertionError("Default review check must not construct a runtime")
    monkeypatch.setattr(app, "create_manager", forbidden)
    result = asyncio.run(app.review_archives(load_config(CONFIG), (source,)))
    assert not result["starts_task"] and result["max_model_calls"] == 2
    assert result["sources"][0]["closed"]
    assert not (source / "reports").exists()


@pytest.mark.parametrize("prompt_version", ["b09-offline-v1", "b09-offline-v2", "b09-offline-v3"])
def test_generated_report_is_readable_without_modifying_online_evidence(tmp_path, prompt_version):
    async def scenario():
        source = copy_source(tmp_path)
        original = hashes(source)
        clock = SystemClock()
        provider = EvidenceProvider()
        def forbidden(_):
            raise AssertionError("Offline review cannot create an online platform")
        manager = RunManager(clock=clock, scheduler=AsyncioScheduler(clock),
                             platform_factory=forbidden, provider_factory=lambda _: provider)
        config = load_config(CONFIG)
        config = config.model_copy(update={"versions": config.versions.model_copy(update={"prompt": prompt_version})})
        result = await app.review_archives(config, (source,), live=True, manager=manager)
        assert result["offline_budget"] == {"max_slots": 2, "slots_used": 1, "calls_started": 1}
        assert len(provider.inputs) == 1 and provider.closed
        assert manager.list_views() == () and hashes(source) == original
        row = result["reports"][0]
        report = row["report"]
        assert report["status"] == "completed" and report["usage"]["total_tokens"] == 30
        assert report["coverage"]["selection_scope"] == "selected_requests"
        assert report["generation"]["prompt_version"] == prompt_version
        assert Path(row["report_path"]).is_file()
        assert (source / "reports" / f"review-{report['report_id']}.execution.json").is_file()
        web = app.create_web_application(recording_root=tmp_path / "unused", allowed_config_names=(),
                                         archive_root=tmp_path)
        async with web.router.lifespan_context(web):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url="http://127.0.0.1:8000") as client:
                base = f"/api/v1/runs/{report['run_id']}"
                body = (await client.get(base + "/reports/" + report["report_id"])).json()
                assert body["items"][0]["evidence_available"] is True
                item = body["items"][0]
                assert item["facts_source"] == ("program" if prompt_version != "b09-offline-v1" else "model")
                assert item["facts_verified"] is (prompt_version != "b09-offline-v1")
                detail = await client.get(f"{base}/sessions/{item['session_id']}/decisions/{item['request_id']}")
                assert detail.status_code == 200
                assert detail.json()["state"]["binding"]["request_id"] == item["request_id"]
        assert len(provider.inputs) == 1 and hashes(source) == original
    asyncio.run(scenario())


def test_evaluation_publication_never_replaces_source_or_existing_reports(tmp_path):
    async def scenario():
        source = copy_source(tmp_path)
        original = hashes(source)
        result = await app.evaluate_archives((source,))
        assert result["model_calls"] == 0 and result["report"]["counts"]["settled"] == 2
        assert hashes(source) == original
        with pytest.raises(FileExistsError):
            await app.evaluate_archives((source,), output=Path(result["report_path"]))
        from guandan_agent.contracts import EvaluationReport
        report = EvaluationReport.model_validate(result["report"])
        with pytest.raises(ValueError):
            await save_report(source / "config.json", report)
        assert hashes(source) == original
    asyncio.run(scenario())


def test_review_cli_requires_explicit_live_flag(tmp_path, capsys):
    source = copy_source(tmp_path)
    assert cli.main(["review", str(CONFIG), str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["starts_task"] is False
    assert not (source / "reports").exists()


def test_unknown_offline_version_is_rejected_without_runtime(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    config = config.model_copy(update={"versions": config.versions.model_copy(update={"prompt": "unknown"})})
    def forbidden():
        raise AssertionError("Unsupported prompt cannot construct a runtime")
    monkeypatch.setattr(app, "create_manager", forbidden)
    with pytest.raises(ValueError, match="Unsupported offline prompt"):
        asyncio.run(app.review_archives(config, (tmp_path / "not-read",), live=True))
