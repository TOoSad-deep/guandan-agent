"""Configuration checks are offline and deliberately do not load .env."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from guandan_agent.config import RunConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = PROJECT_ROOT / "configs" / "local.example.toml"


def test_example_loads_without_credentials(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = load_config(EXAMPLE)
    assert config == RunConfig()
    assert config.run.policy == "V0"
    assert [seat.policy for seat in config.run.seats] == ["V0"] * 4
    assert config.batch.max_model_calls == 20
    assert config.offline.max_model_calls == 2
    assert not config.review.enabled


def test_hash_is_stable_sorted_and_uses_effective_configuration(tmp_path):
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text('[run]\npolicy = "V0"\nmatches = 1\n[batch]\nmax_model_calls = 4\n')
    second.write_text('# reordered\n[batch]\nmax_model_calls = 4\n[run]\nmatches = 1\npolicy = "V0"\n')
    one, two = load_config(first), load_config(second)
    expected = json.dumps(one.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert one.config_id == two.config_id == hashlib.sha256(expected.encode()).hexdigest()
    assert one.config_id != RunConfig().config_id
    explicit = RunConfig.model_validate(RunConfig().model_dump())
    assert explicit.config_id == RunConfig().config_id


def test_environment_and_dotenv_do_not_change_effective_config(tmp_path, monkeypatch):
    before = load_config(EXAMPLE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder-not-a-credential")
    monkeypatch.setenv("DEEPSEEK_MODEL", "must-not-implicitly-override")
    monkeypatch.setenv("GUANDAN_PROBE_TIMEOUT_SECONDS", "999")
    (tmp_path / ".env").write_text("DEEPSEEK_MODEL=must-not-read-dotenv\n")
    monkeypatch.chdir(tmp_path)
    after = load_config(EXAMPLE)
    assert before.config_id == after.config_id
    assert "test-placeholder" not in after.canonical_json()
    assert "DEEPSEEK_API_KEY" not in after.canonical_json()


def test_config_is_deeply_immutable():
    config = RunConfig()
    with pytest.raises(ValidationError):
        config.run = config.run
    with pytest.raises(ValidationError):
        config.run.policy = "V2"
    with pytest.raises(ValidationError):
        config.run.seats[0].user_name = "replacement"
    with pytest.raises(TypeError):
        config.run.seats[0] = config.run.seats[1]


@pytest.mark.parametrize(
    "settings",
    [
        {"unexpected": 1},
        {"run": {"unexpected": 1}},
        {"model": {"api_key": "fake"}},
        {"env": {"DEEPSEEK_API_KEY": "fake"}},
        {"run": {"matches": "1"}},
        {"run": {"matches": True}},
        {"decision": {"total_s": "30"}},
        {"model": {"thinking": 0}},
        {"model": {"max_retries": False}},
        {"web": {"workers": True}},
        {"run": {"seats": [{"seat": "0", "user_name": "p0"}]}},
        {"run": {"seats": [{"seat": 0, "user_name": 42}]}},
        {"run": {"policy": "v0"}},
    ],
)
def test_unknown_fields_and_type_coercions_are_rejected(settings):
    with pytest.raises(ValidationError):
        RunConfig.model_validate(settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"run": {"seats": []}},
        {"run": {"seats": [{"seat": 4, "user_name": "p4"}]}},
        {"run": {"seats": [{"seat": 0, "user_name": "a"}, {"seat": 0, "user_name": "b"}]}},
        {"run": {"seats": [{"seat": 0, "user_name": "a"}, {"seat": 1, "user_name": "a"}]}},
        {"run": {"seats": [{"seat": 0, "user_name": " "}]}},
        {"run": {"evaluation_team": 2}},
        {"run": {"matches": 0}},
        {"platform": {"mode": "join"}},
        {"platform": {"mode": "create", "room_id": 1}},
        {"platform": {"mode": "join", "room_id": 1}, "run": {"matches": 2}},
        {"platform": {"server_rounds": 2}},
        {"platform": {"url": "https://localhost"}},
        {"platform": {"url": "ws://user:fake@localhost"}},
        {"platform": {"url": "ws://localhost:0"}},
        {"platform": {"url": "ws://localhost/#fragment"}},
        {"platform": {"rules_version": "short-sha"}},
        {"decision": {"total_s": 1, "submit_reserve_s": 1}},
        {"decision": {"submit_reserve_s": 0}},
        {"decision": {"confirm_timeout_s": -1}},
        {"decision": {"total_s": float("inf")}},
        {"tools": {"total_s": float("nan")}},
        {"tools": {"candidate_target": 25}},
        {"tools": {"deep_eval_limit": 25}},
        {"tools": {"engine_stop_grace_s": 0}},
        {"model": {"max_retries": 1}},
        {"model": {"thinking": True}},
        {"model": {"timeout_s": 0}},
        {"model": {"max_output_tokens": 0}},
        {"model": {"max_input_bytes": 0}},
        {"model": {"concurrency": 0}},
        {"review": {"enabled": True}},
        {"run": {"policy": "V2"}, "review": {"enabled": False}},
        {"review": {"split_cost_threshold": -1}},
        {"batch": {"max_model_calls": -1}},
        {"offline": {"max_model_calls": -1}},
        {"logs": {"queue_capacity": 0}},
        {"logs": {"max_query_limit": 201}},
        {"web": {"workers": 2}},
        {"web": {"reload": True}},
        {"web": {"port": 65536}},
    ],
)
def test_invalid_values_and_cross_field_constraints_are_rejected(settings):
    with pytest.raises(ValidationError):
        RunConfig.model_validate(settings)


def test_join_and_mixed_policies_resolve_without_credentials():
    config = RunConfig.model_validate(
        {
            "platform": {"mode": "join", "room_id": 17},
            "run": {
                "policy": "V0",
                "seats": [
                    {"seat": 0, "user_name": "p0", "policy": "V2"},
                    {"seat": 2, "user_name": "p2"},
                ],
            },
            "batch": {"max_model_calls": 0},
            "offline": {"max_model_calls": 0},
        }
    )
    assert config.review.enabled
    assert config.run.seats[1].policy == "V0"
    assert config.platform.room_id == 17


def test_explicit_overrides_win_without_mutating_input():
    overrides = {"run": {"policy": "V1"}, "model": {"name": "explicit-test-model"}}
    config = load_config(EXAMPLE, overrides=overrides)
    assert config.run.policy == "V1"
    assert {seat.policy for seat in config.run.seats} == {"V1"}
    assert config.model.name == "explicit-test-model"
    assert config.model.max_retries == 0
    assert overrides == {"run": {"policy": "V1"}, "model": {"name": "explicit-test-model"}}
