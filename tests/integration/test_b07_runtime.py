"""V2 composition uses a synthetic one-request platform and offline providers."""
import asyncio
import json
from pathlib import Path

import pytest

from guandan_agent import app
from guandan_agent.config import load_config
from guandan_agent.model import ScriptedProvider
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.manager import RunManagerError
from scripts import probe_b07_review as probe


def fixed_config():
    return load_config(Path(__file__).resolve().parents[2] / "configs/b07-v2-fixed.toml")


@pytest.mark.parametrize("choice,status", [("L0002", "kept"), ("L0001", "changed")])
def test_fixed_v2_main_review_uses_one_snapshot_and_two_run_slots(tmp_path, choice, status):
    async def scenario():
        provider = ScriptedProvider([
            '{"action_id":"L0002","reason":"主建议"}',
            json.dumps({"action_id": choice, "reason": "复核建议"}),
        ])
        result = await probe.run_probe(fixed_config(), mode="scripted", recording_root=tmp_path,
                                       provider_factory=lambda _: provider)
        assert result["valid_main_review"] and not result["accepted_live_fixed_case"]
        assert result["slots_used"] == result["calls_started"] == len(provider.inputs) == 2
        assert result["review"]["status"] == status
        assert result["review"]["trigger_codes"] == ["OPPONENT_NEAR_FINISH"]
        assert result["fallback_action_id"] == "L0001"  # Main and fallback differ.
        assert result["transport_requests"] == 0 and result["usage"]["total_tokens"] is None
        assert result["synthetic_writes"] == 1 and not result["real_platform"]
        assert not result["pending_call_ids"] and not result["unconfirmed_submission_ids"]
        assert result["closed"] and result["log_complete"] and provider.closed
        first, second = [json.loads(i.messages[1].content) for i in provider.inputs]
        assert first["visible"] == second["visible"] and first["candidates"] == second["candidates"]
        assert second["main"] == {"action_id": "L0002", "reason": "主建议"}
        record = await ClosedRunLog.load(tmp_path / result["run_id"])
        events = record.offline_events()
        chosen = next(e.payload["execution"] for e in events if e.event_type == "submission_reserved")
        assert chosen["source"] == "review" and chosen["action_id"] == choice
        assert sum(e.event_type == "review_finished" for e in events) == 1
    asyncio.run(scenario())


def test_fixed_probe_failed_review_is_not_accepted_and_never_retries(tmp_path):
    async def scenario():
        provider = ScriptedProvider(['{"action_id":"L0002","reason":"main"}', "bad-json",
                                     '{"action_id":"L0002","reason":"must not run"}'])
        result = await probe.run_probe(fixed_config(), mode="scripted", recording_root=tmp_path,
                                       provider_factory=lambda _: provider)
        assert not result["valid_main_review"] and len(provider.inputs) == 2
        assert result["review"]["status"] == "failed" and result["synthetic_writes"] == 1
        record = await ClosedRunLog.load(tmp_path / result["run_id"])
        chosen = next(e.payload["execution"] for e in record.offline_events()
                      if e.event_type == "submission_reserved")
        assert chosen["source"] == "fallback" and chosen["action_id"] == "L0001"
    asyncio.run(scenario())


def test_default_probe_checks_configuration_without_credentials_or_run(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("A config check must not build providers or runtime")
    monkeypatch.setattr(probe, "RunManager", forbidden)
    assert probe.main([]) == 0
    assert json.loads(capsys.readouterr().out)["starts_run"] is False


def test_v2_rejects_legacy_prompt_before_connecting(tmp_path):
    async def scenario():
        config = fixed_config()
        config = config.model_copy(update={"versions": config.versions.model_copy(update={"prompt": "b05-main-v1"})})
        manager = app.create_manager(recording_root=tmp_path)
        with pytest.raises(RunManagerError) as exc:
            manager.start(config)
        assert exc.value.error.code == "VERSION_UNSUPPORTED"
        assert not list(tmp_path.iterdir())
    asyncio.run(scenario())


def test_missing_v2_key_fails_before_constructing_real_platform(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Missing credentials must be detected before connecting")
    monkeypatch.setattr(app, "OpenGuanDanPlatform", forbidden)
    async def scenario():
        final = await app.run_config(fixed_config(), recording_root=tmp_path, max_runtime_s=2)
        assert final.error.code == "MODEL_CREDENTIALS_MISSING" and not final.sessions
    asyncio.run(scenario())
