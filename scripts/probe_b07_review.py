#!/usr/bin/env python3
"""P16: fixed synthetic inputs, optionally two real model calls, no live game.

The default checks configuration only. Scripted platform feedback is explicitly
synthetic even when --mode live selects the real DeepSeek provider.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig, load_config
from guandan_agent.hand import HandTools
from guandan_agent.platform import ScriptedPlatform
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/b07/opponent_one_card.json"


def validate_config(config: RunConfig) -> None:
    if not (config.run.split == "development" and config.run.matches == 1
            and len(config.run.seats) == 1 and config.run.seats[0].seat == 0
            and config.run.seats[0].policy == "V2" and config.review.enabled
            and config.batch.max_model_calls == 2 and config.offline.max_model_calls == 0
            and config.model.concurrency == 1 and config.tools.deep_eval_limit == 0
            and config.versions.prompt == "b07-review-v1"):
        raise ValueError("P16 requires one synthetic V2 request with a two-call budget")


class FixedScenarioPlatform(ScriptedPlatform):
    """One artificial request and its local feedback, not a game engine."""

    def __init__(self, fixture: dict[str, Any], *, clock: SystemClock):
        super().__init__(clock=clock)
        self.fixture = fixture
        self.writes = 0

    async def connect(self, connection):
        await super().connect(connection)
        self.feed({"type": "CREATE_ROOM", "code": 200,
                   "data": {"roomId": 707, "userNum": 0}})
        self.feed(self.fixture["beginning"])
        self.feed(self.fixture["prior_play"])
        self.feed(self.fixture["request"])

    async def write(self, intent, gate):
        result = await super().write(intent, gate)
        if result.status == "sent":
            self.writes += 1
            request = self.fixture["request"]
            self.feed({"type": "notify", "stage": "play", "curPos": 0,
                       "curAction": intent.model_dump(mode="json")["wire_action"],
                       "greaterPos": request["greaterPos"],
                       "greaterAction": request["greaterAction"]})
        return result


async def run_probe(config: RunConfig, *, mode: str, recording_root: Path = Path("runs"),
                    provider_factory=None, max_runtime_s: float = 35.0) -> dict[str, Any]:
    validate_config(config)
    if mode not in {"scripted", "live"} or not 0 < max_runtime_s <= 60:
        raise ValueError("Invalid explicit probe mode or finite deadline")
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    clock = SystemClock()
    platform = FixedScenarioPlatform(fixture, clock=clock)

    def make_provider(current):
        if provider_factory is not None:
            if mode == "live":
                raise ValueError("Live mode cannot replace the real provider")
            return provider_factory(current)
        if mode == "scripted":
            from guandan_agent.model import ScriptedProvider
            return ScriptedProvider([
                '{"action_id":"L0002","reason":"固定脚本主建议"}',
                '{"action_id":"L0001","reason":"固定脚本复核改选"}',
            ], clock=clock)
        from guandan_agent.model.deepseek import DeepSeekProvider
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("Missing process model credential")
        return DeepSeekProvider(api_key=key, model=current.model.name, clock=clock)

    manager = RunManager(platform_factory=lambda _: platform, clock=clock,
                         scheduler=AsyncioScheduler(clock), recording_root=recording_root,
                         tools_factory=lambda _: HandTools(clock=clock),
                         provider_factory=make_provider)
    initial = manager.start(config)
    deadline = asyncio.get_running_loop().time() + max_runtime_s
    stop_reason = "runtime_limit"
    try:
        while asyncio.get_running_loop().time() < deadline:
            view = manager.view(initial.run_id)
            if view.status in {"completed", "interrupted"}:
                stop_reason = "run_ended"
                break
            if view.sessions:
                state = manager.session_view(initial.run_id, view.sessions[0].session_id)
                if state.latest_execution is not None and state.latest_execution.status == "confirmed":
                    stop_reason = "synthetic_action_confirmed"
                    break
            await asyncio.sleep(0.005)
    finally:
        manager.stop(initial.run_id)
        final = await manager.wait(initial.run_id)

    directory = recording_root / initial.run_id
    record = await ClosedRunLog.load(directory)
    events = record.offline_events()
    results = [e.payload["result"] for e in events if e.event_type == "model_call_finished"]
    reviews = [e for e in events if e.event_type == "review_finished"]
    budget = next(e.payload for e in events if e.event_type == "model_budget_summary")
    context = next((e.payload["context"] for e in events if e.event_type == "request_opened"), None)
    review = reviews[-1].payload["review"] if reviews else None
    valid = (len(results) == 2 and [r["role"] for r in results] == ["main", "review"]
             and all(r["decision"] is not None and r["error"] is None for r in results)
             and results[0]["binding"] == results[1]["binding"]
             and results[0]["candidate_snapshot_id"] == results[1]["candidate_snapshot_id"]
             and review is not None and review["status"] in {"kept", "changed"}
             and review["before"] == results[0]["decision"] and review["after"] == results[1]["decision"])
    total = lambda field: (sum(r[field] for r in results) if all(r[field] is not None for r in results) else None)
    token_total = lambda field: (sum(r["usage"][field] for r in results)
                                 if results and all(r["usage"][field] is not None for r in results) else None)
    transport = total("transport_requests")
    report = {
        "schema_version": "1.0", "batch": "B07", "task": "P16", "mode": mode,
        "fixture_kind": fixture["fixture_kind"], "fixture_path": str(FIXTURE.relative_to(Path(__file__).resolve().parents[1])),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "source_case_id": fixture["source_case_id"], "source_sha256": fixture["source_sha256"],
        "run_id": initial.run_id, "config_id": config.config_id, "status": final.status,
        "stop_reason": stop_reason, "provider": "deepseek" if mode == "live" else "scripted",
        "real_platform": False, "natural_review_confirmation_covered": False,
        "full_match": False, "winner": None, "split": "development",
        "valid_main_review": valid,
        "accepted_live_fixed_case": bool(valid and mode == "live" and transport == 2
                                         and record.summary.closed and record.summary.log_complete),
        "slots_used": budget["slots_used"], "calls_started": budget["calls_started"],
        "transport_requests": transport, "usage": {k: token_total(k) for k in ("input_tokens", "output_tokens", "total_tokens")},
        "cost_amount": None, "calls": [json.loads(json.dumps(dict(r), default=lambda x: dict(x))) for r in results],
        "review": None if review is None else json.loads(json.dumps(dict(review), default=lambda x: dict(x))),
        "review_event_id": reviews[-1].event_id if reviews else None,
        "fallback_action_id": context["fallback_action_id"] if context else None,
        "synthetic_writes": platform.writes,
        "pending_call_ids": record.summary.pending_call_ids,
        "unconfirmed_submission_ids": record.summary.unconfirmed_submission_ids,
        "closed": record.summary.closed, "log_complete": record.summary.log_complete,
        "runtime_source_sha256": next(e.payload["environment"]["source_sha256"] for e in events if e.event_type == "run_started"),
    }
    report_path = directory / "reports/b07-fixed-review.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {**report, "report_path": str(report_path.resolve())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/b07-v2-fixed.toml"))
    parser.add_argument("--mode", choices=("check", "scripted", "live"), default="check")
    parser.add_argument("--recording-root", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        validate_config(config)
        if args.mode == "check":
            print(json.dumps({"mode": "check", "config_id": config.config_id,
                              "max_model_calls": 2, "real_platform": False, "starts_run": False}))
            return 0
        result = asyncio.run(run_probe(config, mode=args.mode, recording_root=args.recording_root))
    except KeyboardInterrupt:
        print('{"error":"PROBE_INTERRUPTED"}')
        return 130
    except Exception:
        print('{"error":"PROBE_FAILED"}')
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid_main_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
