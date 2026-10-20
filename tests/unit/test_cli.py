import json
import asyncio
from pathlib import Path

import pytest

from guandan_agent.cli import main


ROOT = Path(__file__).resolve().parents[2]


def test_config_check_and_explicit_model_override(capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "ignored-implicit-model")
    assert main(["config-check", str(ROOT / "configs/local.example.toml"),
                 "--model", "explicit-model"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["config"]["model"]["name"] == "explicit-model"
    assert result["config"]["run"]["policy"] == "V0"
    assert len(result["config_id"]) == 64


def test_bad_config_never_echoes_input_values(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text('api_key = "test-secret-value"\n')
    assert main(["config-check", str(path)]) == 2
    output = capsys.readouterr()
    assert "test-secret-value" not in output.out + output.err
    assert "invalid_config" in output.err


def test_unreadable_or_invalid_toml_is_a_clean_failure(tmp_path, capsys):
    path = tmp_path / "missing.toml"
    assert main(["config-check", str(path)]) == 2
    path.write_text("[broken\n")
    assert main(["config-check", str(path)]) == 2
    assert "Traceback" not in capsys.readouterr().err


def _view(status="completed"):
    from guandan_agent.contracts import RunView
    return RunView(
        run_id="00000000-0000-4000-8000-000000000001", status=status,
        config_id="test-config", active_match_id=None, sessions=(), log_complete=True,
    )


def test_run_command_dispatches_explicit_options_without_environment(tmp_path, capsys, monkeypatch):
    from guandan_agent import app
    observed = []

    async def fake_run(config, **kwargs):
        observed.append((config, kwargs))
        return _view()

    monkeypatch.setattr(app, "run_config", fake_run)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-never-appear")
    assert main(["run", str(ROOT / "configs/local.example.toml"),
                 "--recording-root", str(tmp_path), "--startup-timeout-s", "1.5",
                 "--max-runtime-s", "5"]) == 0
    config, options = observed[0]
    assert all(item.policy == "V0" for item in config.run.seats)
    assert options == {"recording_root": tmp_path, "startup_timeout_s": 1.5,
                       "max_runtime_s": 5.0}
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "completed"
    assert "must-never-appear" not in output.out + output.err


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("interrupted", 1)])
def test_run_exit_code_tracks_trusted_completion(status, exit_code, capsys, monkeypatch):
    from guandan_agent import app

    async def fake_run(*args, **kwargs):
        return _view(status)

    monkeypatch.setattr(app, "run_config", fake_run)
    assert main(["run", str(ROOT / "configs/local.example.toml")]) == exit_code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_run_provider_or_transport_exception_is_not_echoed(capsys, monkeypatch):
    from guandan_agent import app

    async def failing(*args, **kwargs):
        raise RuntimeError("sensitive-transport-response")

    monkeypatch.setattr(app, "run_config", failing)
    assert main(["run", str(ROOT / "configs/local.example.toml")]) == 1
    output = capsys.readouterr()
    assert "sensitive-transport-response" not in output.out + output.err
    assert "RUN_FAILED" in output.err


def test_run_rejected_policy_is_a_safe_error(capsys, monkeypatch):
    from guandan_agent import app
    from guandan_agent.runtime.manager import RunManagerError

    async def failing(*args, **kwargs):
        raise RunManagerError("POLICY_UNAVAILABLE", "Only V0 is available.")

    monkeypatch.setattr(app, "run_config", failing)
    assert main(["run", str(ROOT / "configs/local.example.toml")]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "POLICY_UNAVAILABLE"


def test_run_keyboard_interrupt_has_clean_exit(capsys, monkeypatch):
    from guandan_agent import app

    async def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "run_config", interrupted)
    assert main(["run", str(ROOT / "configs/local.example.toml")]) == 130
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_run_duration_must_be_finite_positive(value):
    with pytest.raises(SystemExit) as caught:
        main(["run", str(ROOT / "configs/local.example.toml"), "--max-runtime-s", value])
    assert caught.value.code == 2


def test_app_cancellation_stops_before_propagating_and_waits_cleanup(monkeypatch):
    from guandan_agent import app
    from guandan_agent.config import RunConfig

    async def scenario():
        waiting = asyncio.Event()
        calls = []

        class FakeManager:
            def start(self, config):
                calls.append("start")
                return _view("starting")

            async def wait(self, run_id):
                calls.append("wait")
                if "stop" not in calls:
                    waiting.set()
                    await asyncio.Event().wait()
                calls.append("closed")
                return _view("interrupted")

            def stop(self, run_id):
                calls.append("stop")

        monkeypatch.setattr(app, "create_manager", lambda **kwargs: FakeManager())
        task = asyncio.create_task(app.run_config(RunConfig()))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == ["start", "wait", "stop", "wait", "closed"]
    asyncio.run(scenario())


def test_app_explicit_runtime_limit_returns_interrupted_after_stop(monkeypatch):
    from guandan_agent import app
    from guandan_agent.config import RunConfig

    async def scenario():
        stopped = False

        class FakeManager:
            def start(self, config):
                return _view("starting")

            async def wait(self, run_id):
                if not stopped:
                    await asyncio.Event().wait()
                return _view("interrupted")

            def stop(self, run_id):
                nonlocal stopped
                stopped = True

        monkeypatch.setattr(app, "create_manager", lambda **kwargs: FakeManager())
        result = await app.run_config(RunConfig(), max_runtime_s=0.01)
        assert stopped and result.status == "interrupted"
    asyncio.run(scenario())
