"""Strict, immutable, non-secret run configuration.

Loading a configuration never reads environment variables or a .env file and
never starts a platform or model connection. Runtime credential loading belongs
to the model entry point in a later implementation batch.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BeforeValidator, Field, StringConstraints, field_validator, model_validator

from guandan_agent.contracts import ContractModel, Policy, PositiveInt, Seat

NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def _freeze_sequence(value: Any) -> Any:
    # TOML arrays are lists. Preserve strict element validation and reject all
    # other container types rather than silently accepting strings/generators.
    if isinstance(value, list):
        return tuple(value)
    return value


class SeatConfig(ContractModel):
    seat: Seat
    user_name: NonEmptyString
    policy: Policy | None = None


SeatConfigs = Annotated[
    tuple[SeatConfig, ...], BeforeValidator(_freeze_sequence), Field(min_length=1, max_length=4)
]


def _default_seats() -> tuple[SeatConfig, ...]:
    return tuple(SeatConfig(seat=seat, user_name=f"local-p{seat}") for seat in range(4))


class RunSettings(ContractModel):
    policy: Policy = "V0"
    seats: SeatConfigs = Field(default_factory=_default_seats)
    matches: PositiveInt = 1
    evaluation_team: Annotated[int, Field(ge=0, le=1)] = 0
    split: Literal["development", "final"] = "development"
    experiment_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def resolve_seats(self) -> RunSettings:
        if len({item.seat for item in self.seats}) != len(self.seats):
            raise ValueError("run.seats must have distinct seat numbers")
        if len({item.user_name for item in self.seats}) != len(self.seats):
            raise ValueError("run.seats must have distinct user_name values")
        # The frozen value and its hash describe the effective policy of every
        # seat, so omitting an override and explicitly repeating it hash alike.
        resolved = tuple(
            item if item.policy is not None else item.model_copy(update={"policy": self.policy})
            for item in self.seats
        )
        object.__setattr__(self, "seats", resolved)
        return self


class PlatformConfig(ContractModel):
    url: str = "ws://127.0.0.1:18181"
    adapter_version: NonEmptyString = "openguandan-e92db7f-v1"
    rules_version: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] = (
        "e92db7fa9d915fb318b3be78703d5ccd4a09c887"
    )
    mode: Literal["create", "join"] = "create"
    room_id: PositiveInt | None = None
    server_rounds: Annotated[int, Field(ge=1, le=1)] = 1

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("platform.url must be a valid WebSocket URL") from exc
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError("platform.url must be a ws:// or wss:// URL with a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("platform.url cannot contain credentials")
        if parsed.fragment or any(character.isspace() for character in value):
            raise ValueError("platform.url cannot contain a fragment or whitespace")
        if port is not None and port < 1:
            raise ValueError("platform.url port must be positive")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> PlatformConfig:
        if self.mode == "join" and self.room_id is None:
            raise ValueError("platform.room_id is required for join mode")
        if self.mode == "create" and self.room_id is not None:
            raise ValueError("platform.room_id must be omitted for create mode")
        return self


class DecisionConfig(ContractModel):
    total_s: PositiveSeconds = 30.0
    submit_reserve_s: PositiveSeconds = 1.0
    confirm_timeout_s: PositiveSeconds = 5.0

    @model_validator(mode="after")
    def validate_reserve(self) -> DecisionConfig:
        if self.total_s <= self.submit_reserve_s:
            raise ValueError("decision.total_s must exceed submit_reserve_s")
        return self


class ToolConfig(ContractModel):
    total_s: PositiveSeconds = 0.5
    deep_eval_limit: NonNegativeInt = 8
    candidate_target: PositiveInt = 12
    candidate_hard_limit: PositiveInt = 24
    engine_stop_grace_s: PositiveSeconds = 0.5

    @model_validator(mode="after")
    def validate_limits(self) -> ToolConfig:
        if self.candidate_target > self.candidate_hard_limit:
            raise ValueError("tools.candidate_target cannot exceed candidate_hard_limit")
        if self.deep_eval_limit > self.candidate_hard_limit:
            raise ValueError("tools.deep_eval_limit cannot exceed candidate_hard_limit")
        return self


class ModelConfig(ContractModel):
    name: NonEmptyString = "deepseek-flash"
    timeout_s: PositiveSeconds = 20.0
    max_output_tokens: PositiveInt = 384
    max_input_bytes: PositiveInt = 24576
    max_retries: Annotated[int, Field(ge=0, le=0)] = 0
    thinking: bool = False
    concurrency: PositiveInt = 1

    @field_validator("thinking")
    @classmethod
    def thinking_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("model.thinking must be false for this application version")
        return value


class ReviewConfig(ContractModel):
    enabled: bool = False
    min_remaining_s: PositiveSeconds = 2.0
    split_cost_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @field_validator("split_cost_threshold")
    @classmethod
    def split_risk_not_calibrated(cls, value: float | None) -> None:
        if value is not None:
            raise ValueError("SPLIT_COST risk is unavailable until its scoring threshold is validated")
        return None


class BatchConfig(ContractModel):
    max_model_calls: NonNegativeInt = 20


class OfflineConfig(ContractModel):
    max_model_calls: NonNegativeInt = 2


class HistoryConfig(ContractModel):
    max_events: NonNegativeInt = 100


class LogConfig(ContractModel):
    queue_capacity: PositiveInt = 4096
    cache_events: PositiveInt = 2000
    max_query_limit: Annotated[int, Field(ge=1, le=200)] = 200


class UIConfig(ContractModel):
    poll_ms: PositiveInt = 1000


class WebConfig(ContractModel):
    host: NonEmptyString = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    workers: Annotated[int, Field(ge=1, le=1)] = 1
    reload: bool = False

    @field_validator("reload")
    @classmethod
    def reload_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("web.reload must be false to keep one runtime authority")
        return value


class VersionConfig(ContractModel):
    policy: NonEmptyString = "b05-rules-v1"
    candidate: NonEmptyString = "b05-candidates-v1"
    fallback: NonEmptyString = "b02-cheap-v1"
    prompt: NonEmptyString = "b09-facts-v3"


class RunConfig(ContractModel):
    run: RunSettings = Field(default_factory=RunSettings)
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    offline: OfflineConfig = Field(default_factory=OfflineConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    logs: LogConfig = Field(default_factory=LogConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    versions: VersionConfig = Field(default_factory=VersionConfig)

    @model_validator(mode="before")
    @classmethod
    def default_review_for_policies(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        # Inspect only well-shaped containers here; malformed values remain for
        # Pydantic's strict field validation instead of being coerced or hidden.
        run = value.get("run", {})
        review = value.get("review", {})
        if isinstance(run, RunSettings):
            has_v2 = any(seat.policy == "V2" for seat in run.seats)
        elif isinstance(run, dict):
            policy = run.get("policy", "V0")
            seats = run.get("seats")
            if seats is None:
                has_v2 = policy == "V2"
            elif isinstance(seats, (list, tuple)):
                has_v2 = any(
                    (seat.policy or policy) == "V2"
                    if isinstance(seat, SeatConfig)
                    else (seat.get("policy") or policy) == "V2"
                    for seat in seats
                    if isinstance(seat, (SeatConfig, dict))
                )
            else:
                return value
        else:
            return value
        if isinstance(review, dict) and "enabled" not in review:
            return {**value, "review": {**review, "enabled": has_v2}}
        return value

    @model_validator(mode="after")
    def validate_run(self) -> RunConfig:
        if self.platform.mode == "join" and self.run.matches != 1:
            raise ValueError("join mode requires run.matches = 1")
        has_v2 = any(seat.policy == "V2" for seat in self.run.seats)
        if self.review.enabled != has_v2:
            raise ValueError("review.enabled must match whether any effective seat policy is V2")
        return self

    def canonical_json(self) -> str:
        """Canonical, credential-free effective settings used for config_id."""
        return json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def config_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _merge_explicit(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_explicit(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path, *, overrides: Mapping[str, Any] | None = None
) -> RunConfig:
    """Load TOML and optional explicit CLI settings, validate, then freeze.

    Built-in defaults are supplied by RunConfig. Only explicitly supplied
    overrides take precedence over TOML; environment variables are ignored.
    File, TOML and Pydantic validation errors are left for the caller to report.
    """
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if overrides is not None:
        data = _merge_explicit(data, overrides)
    return RunConfig.model_validate(data)
