"""Publish new report files without changing or replacing source evidence."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from guandan_agent.contracts import EvaluationReport, OfflineTaskView, ReviewReport


async def save_report(path: Path, report: ReviewReport | EvaluationReport) -> Path:
    if path.name in {"config.json", "events.jsonl", "summary.json", "summary.pending"}:
        raise ValueError("Reports cannot replace core run evidence")
    data = report.model_dump_json(indent=2).encode() + b"\n"
    if len(data) > 512 * 1024:
        raise ValueError("Report exceeds the configured read boundary")

    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        # A generated UUID normally ensures a fresh name; existing files are
        # never replaced even when the caller supplies --output manually.
        with path.open("xb") as stream:
            stream.write(data)
    await asyncio.to_thread(write)
    return path


async def save_review_execution(path: Path, *, view: OfflineTaskView, config,
                                budget: dict[str, int]) -> None:
    value = {"schema_version": "1.0", "kind": "offline_execution",
             "task": view.model_dump(mode="json"), "offline_config_id": config.config_id,
             "offline_config": config.model_dump(mode="json"), "offline_budget": budget}
    def write():
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    await asyncio.to_thread(write)
