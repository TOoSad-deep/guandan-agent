#!/usr/bin/env python3
"""Bounded P13 development probe; checking a config is the default operation.

This observer reads each newly persisted log line once. It never claims that a
local stop observation replaces the independent raw-platform acceptance audit.
Credentials are supplied to the application process externally, never read here.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import ipaddress
import json
import math
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

from guandan_agent import app
from guandan_agent.config import RunConfig, load_config


def validate_probe_config(config: RunConfig) -> None:
    """P13 uses a finite local development allocation, never a final match."""
    host = urlsplit(config.platform.url).hostname
    try:
        local = host == "localhost" or ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        local = False
    if not (
        local and config.platform.mode == "create"
        and config.run.split == "development" and config.run.matches == 1
        and {seat.seat for seat in config.run.seats} == {0, 1, 2, 3}
        and [seat.policy for seat in config.run.seats].count("V1") == 1
        and all(seat.policy in {"V0", "V1"} for seat in config.run.seats)
        and 1 <= config.batch.max_model_calls <= 4
        and config.offline.max_model_calls == 0 and not config.review.enabled
    ):
        raise ValueError("B06 requires one local development match, four seats with one V1, 1-4 online calls, and no offline/review calls")


class IncrementalEvents:
    """Keep one file cursor and an incomplete-line buffer, bounded per read."""

    def __init__(self, path: Path):
        self.path = path
        self.stream: BinaryIO | None = None
        self.partial = b""
        self.last_read_size = 0

    def read(self) -> list[dict[str, Any]]:
        self.last_read_size = 0
        if self.stream is None:
            try:
                self.stream = self.path.open("rb", buffering=0)
            except FileNotFoundError:
                return []
        data = self.stream.read(65536)
        self.last_read_size = len(data)
        if not data:
            return []
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        if len(self.partial) > 4 * 1024 * 1024:
            raise ValueError("Evidence line exceeds probe observation bound")
        return [json.loads(line) for line in lines if line]

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


@dataclass
class ProbeObservation:
    limit: int
    admitted: set[str] = field(default_factory=set)
    started: set[str] = field(default_factory=set)
    finished: set[str] = field(default_factory=set)
    successful_requests: set[str] = field(default_factory=set)
    invalidated_requests: set[str] = field(default_factory=set)
    executions: dict[str, dict[str, Any]] = field(default_factory=dict)
    main_confirmation: dict[str, Any] | None = None
    last_seq: int = 0

    def consume(self, event: dict[str, Any]) -> None:
        self.last_seq = event["seq"]
        kind, payload, request = event["event_type"], event["payload"], event["request_id"]
        if kind == "model_call_admitted":
            self.admitted.add(payload["call_id"])
        elif kind == "model_call_started":
            self.started.add(payload["call_id"])
        elif kind == "model_call_finished":
            result = payload["result"]
            self.finished.add(result["call_id"])
            if result["decision"] is not None and result["error"] is None and not payload["late"]:
                self.successful_requests.add(request)
        elif kind == "request_invalidated":
            self.invalidated_requests.add(request)
        execution = payload.get("execution")
        if execution is not None:
            self.executions[request] = execution
            if execution["source"] == "main" and execution["status"] == "confirmed" and self.main_confirmation is None:
                self.main_confirmation = {
                    "event_id": event["event_id"], "request_id": request,
                    "submission_id": execution["submission_id"],
                    "feedback_event_id": execution["feedback_event_id"],
                    "action_id": execution["action_id"],
                }

    @property
    def exhausted_without_possible_confirmation(self) -> bool:
        if len(self.admitted) < self.limit or not self.admitted <= self.finished:
            return False
        for request in self.successful_requests:
            execution = self.executions.get(request)
            if execution is None:
                if request not in self.invalidated_requests:
                    return False  # The graph may still register/reserve this result.
            elif execution["source"] == "main" and execution["status"] in {"reserved", "writing", "sent"}:
                return False  # Do not cancel the last valid attempt before feedback.
        return self.main_confirmation is None


async def run_probe(config: RunConfig, *, recording_root: Path = Path("runs"),
                    max_runtime_s: float = 120.0, startup_timeout_s: float = 15.0,
                    manager_factory: Any = None) -> dict[str, Any]:
    validate_probe_config(config)
    if not math.isfinite(max_runtime_s) or not 0 < max_runtime_s <= 120:
        raise ValueError("max_runtime_s must be positive and at most 120")
    manager = (manager_factory or app.create_manager)(
        recording_root=recording_root, startup_timeout_s=startup_timeout_s)
    initial = manager.start(config)
    run_dir = recording_root / initial.run_id
    tail = IncrementalEvents(run_dir / "events.jsonl")
    seen = ProbeObservation(config.batch.max_model_calls)
    waiter = asyncio.create_task(manager.wait(initial.run_id))
    deadline = asyncio.get_running_loop().time() + max_runtime_s
    reason = "run_ended"
    try:
        while True:
            events = tail.read()
            for event in events:
                seen.consume(event)
                if seen.main_confirmation is not None:
                    reason = "main_confirmed"
                    break
            if reason == "main_confirmed":
                manager.stop(initial.run_id)
                break
            if seen.exhausted_without_possible_confirmation:
                reason = "budget_exhausted"
                manager.stop(initial.run_id)
                break
            if waiter.done():
                # The writer has closed; consume any remaining complete chunks.
                if tail.last_read_size:
                    continue
                break
            if asyncio.get_running_loop().time() >= deadline:
                reason = "runtime_limit"
                manager.stop(initial.run_id)
                break
            await asyncio.sleep(0.01)
        final = await waiter
    except BaseException:
        manager.stop(initial.run_id)
        await asyncio.shield(waiter)
        raise
    finally:
        tail.close()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    result = {
        "schema_version": "1.0", "run_id": initial.run_id, "config_id": config.config_id,
        "split": "development", "stop_reason": reason, "status": final.status,
        "observed_main_confirmation": seen.main_confirmation,
        "requires_independent_acceptance_audit": True,
        "budget_limit": config.batch.max_model_calls, "observed_slots_admitted": len(seen.admitted),
        "observed_calls_started": len(seen.started), "observed_calls_finished": len(seen.finished),
        "last_observed_seq": seen.last_seq, "log_complete": final.log_complete,
        "pending_call_ids": summary["pending_call_ids"] if summary else None,
        "unconfirmed_submission_ids": summary["unconfirmed_submission_ids"] if summary else None,
        "run_error_code": final.error.code if final.error else None,
        "run_dir": str(run_dir.resolve()),
    }
    # Report is separate from immutable runtime-owned config/events/summary.
    report_path = run_dir / "reports" / "b06-probe.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {**result, "report_path": str(report_path.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B06 bounded development probe; no run unless --live")
    parser.add_argument("config", type=Path, nargs="?", default=Path("configs/b06-v1.toml"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--recording-root", type=Path, default=Path("runs"))
    parser.add_argument("--max-runtime-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        validate_probe_config(config)
        if not math.isfinite(args.max_runtime_s) or not 0 < args.max_runtime_s <= 120:
            raise ValueError("Invalid runtime bound")
        if not args.live:
            print(json.dumps({"mode": "config_check", "config_id": config.config_id,
                              "budget_limit": config.batch.max_model_calls, "starts_run": False}))
            return 0
        result = asyncio.run(run_probe(config, recording_root=args.recording_root,
                                       max_runtime_s=args.max_runtime_s))
    except KeyboardInterrupt:
        print('{"error":"PROBE_INTERRUPTED"}')
        return 130
    except Exception:
        # Do not echo exceptions or input values that might contain secrets.
        print('{"error":"PROBE_FAILED"}')
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["observed_main_confirmation"] is not None and result["log_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
