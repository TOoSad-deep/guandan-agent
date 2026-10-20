#!/usr/bin/env python3
"""Recompute B10's per-version reports from complete closed evidence, offline."""
from __future__ import annotations
import asyncio
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from guandan_agent.offline.evaluate import evaluate
from guandan_agent.recording import ClosedRunLog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/b10"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def distribution(values):
    ordered = sorted(values)
    def percentile(q):
        if not ordered:
            return None
        i = (len(ordered) - 1) * q
        return ordered[math.floor(i)] + (ordered[math.ceil(i)] - ordered[math.floor(i)]) * (i - math.floor(i))
    return {"n": len(ordered), "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "max_ms": max(ordered) if ordered else None, "quantile_method": "linear interpolation at (n-1)*q"}


async def main():
    plan = json.loads((OUT / "frozen-plan.json").read_text())
    comparisons, records = [], []
    for version in ("V0", "V1", "V2"):
        result = json.loads((OUT / "live" / version / "result.json").read_text())
        run_id = result["run_id"]
        if run_id is None:
            comparisons.append({"version": version, "run_id": None, "status": result["status"],
                                "error_code": result["error_code"]})
            continue
        directory = ROOT / "runs" / run_id
        rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        record = await ClosedRunLog.load(directory, cache_events=len(rows) + 1)
        frozen = plan["configs"][version]
        if (record.config_id != frozen["config_id"] or record.config is None or
                record.config.model_dump(mode="json") != frozen["config"]):
            raise ValueError(f"{version} archive does not match the frozen experiment configuration")
        if any(previous.run_id == record.run_id for previous in records):
            raise ValueError("One archive cannot represent two versions")
        records.append(record)
        reports = {}
        for unit in ("match", "round"):
            args = dict(report_id=str(uuid5(NAMESPACE_URL, f"b10/{version}/{unit}/{run_id}")),
                        created_at=plan["frozen_at"], unit=unit)
            report = evaluate([record], **args).model_dump(mode="json")
            if report != evaluate([record], **args).model_dump(mode="json"):
                raise ValueError("Evaluation changed under identical closed input")
            save(OUT / "evaluation" / f"{version}-{unit}.json", report)
            reports[unit] = report
        requests = {r["request_id"]: r for r in rows if r["event_type"] == "request_opened"}
        final = {}
        proposals = {}
        call_results = {}
        starts = []
        for row in rows:
            p = row["payload"]
            if "execution" in p:
                final[row["request_id"]] = (row, p["execution"])
            if row["event_type"] == "proposal_ready":
                proposals[row["request_id"]] = p["proposal"]
            if row["event_type"] == "model_call_started":
                starts.append(row)
            if row["event_type"] in ("model_call_finished", "model_call_usage_updated"):
                if "result" in p:
                    call_results[p["result"]["call_id"]] = p["result"]
        own = {q: r for q, r in requests.items() if r["seat"] == 0}
        own_executions = {q: pair for q, pair in final.items() if pair[0]["seat"] == 0}
        own_nonunique = {q for q, r in own.items() if len(r["payload"]["context"]["legal"]["actions"]) > 1}
        own_sources = Counter(e["source"] for _, e in own_executions.values())
        own_write_ms = {q: (datetime.fromisoformat(e["write_started_at"]) -
                            datetime.fromisoformat(own[q]["payload"]["context"]["received_at"])).total_seconds() * 1000
                        for q, (_, e) in own_executions.items() if e["write_started_at"] is not None}
        model_requests = {r["request_id"] for r in starts}
        own_budget_fallback = sum(e["source"] == "fallback" and proposals.get(q, {}).get("cause_code") in {"BATCH_BUDGET", "CALL_BUDGET"}
                                  for q, (_, e) in own_executions.items())
        examples = {}
        for q, (event, execution) in own_executions.items():
            source = execution["source"]
            if execution["status"] == "confirmed" and source not in examples:
                examples[source] = {"run_id": run_id, "session_id": event["session_id"],
                    "request_id": q, "event_id": event["event_id"],
                    "feedback_event_id": execution["feedback_event_id"],
                    "action_id": execution["action_id"], "status": "confirmed"}
        match = reports["match"]
        comparisons.append({
            "version": version, "run_id": run_id, "config_id": record.config_id,
            "status": result["status"], "log_complete": record.health.log_complete,
            "match_counts": match["counts"], "rounds_settled": reports["round"]["counts"]["settled"],
            "whole_table": {"requests": len(requests), "submissions": len(final),
                "confirmed": sum(e["status"] == "confirmed" for _, e in final.values()),
                "sources": dict(Counter(e["source"] for _, e in final.values()))},
            "tested_seat0": {"requests": len(own), "nonunique_requests": len(own_nonunique),
                "sources": dict(own_sources), "budget_fallback": own_budget_fallback,
                "fallback_rate": {"numerator": own_sources["fallback"], "denominator": len(own_nonunique),
                    "value": own_sources["fallback"] / len(own_nonunique) if own_nonunique else None},
                "model_adopted": sum(own_sources[s] for s in ("main", "review", "repair")),
                "request_to_first_write": distribution(list(own_write_ms.values())),
                "requests_with_started_model_to_first_write": distribution([ms for q, ms in own_write_ms.items() if q in model_requests]),
                "path_examples": examples},
            "calls_by_role": dict(Counter(r["payload"]["role"] for r in starts)),
            "model_errors": dict(Counter(c["error"]["code"] for c in call_results.values() if c.get("error"))),
            "model_usage": match["model_usage"], "latency": match["latency"], "quality": match["quality"],
            "tool_failure_count": sum(r["event_type"] == "tools_failed" for r in rows),
            "tool_failure_codes": dict(Counter(r["payload"].get("error", {}).get("code", "UNKNOWN")
                                        for r in rows if r["event_type"] == "tools_failed")),
        })
    if records:
        mixed = evaluate(records, report_id=str(uuid5(NAMESPACE_URL, "b10/mixed")),
                         created_at=plan["frozen_at"]).model_dump(mode="json")
        save(OUT / "evaluation/mixed-match.json", mixed)
    save(OUT / "comparison.json", {"batch": "B10", "tested_seat": 0, "sample_size_per_version": 1,
        "paired_verified": False, "created_at_basis": "frozen plan timestamp for reproducible offline output",
        "evaluation_recomputed_identically": True, "versions": comparisons,
        "limits": ["Different random deals; no seat rotation; no causal or win-rate superiority conclusion",
                   "20-call caps describe budget-limited policies, not all-game model play",
                   "Model text factual correctness and strategic strength are not measured by legality",
                   "Teacher API permission, version and time limit remain unknown"]})
    print(json.dumps({"versions": len(comparisons), "reports": str(OUT / "comparison.json"), "model_calls": 0}))


if __name__ == "__main__":
    asyncio.run(main())
