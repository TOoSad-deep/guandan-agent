#!/usr/bin/env python3
"""Bounded factual-repair retest: 3 main, 2 dependent review, 10 offline calls.

Default mode checks the frozen plan only. Live mode uses an existing process
credential; it never loads .env, retries, resumes, or replaces skipped calls.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

from guandan_agent.contracts import ModelDecision, ModelInput, ProviderInput, ProviderMessage, ProviderReply
from guandan_agent.model.gateway import build_messages
from guandan_agent.model.provider import ProviderPort, safe_error, unknown_usage
if __package__ in {None, ""}:
    from run_model_validation25 import _json, _unique, _write, check_output
else:
    from scripts.run_model_validation25 import _json, _unique, _write, check_output

LIMIT = 15
MAX_INPUT_BYTES = 24576
ONLINE_PROMPT = "b09-facts-v2"
OFFLINE_PROMPT = "b09-offline-v2"


def _message_size(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _source(case: dict) -> tuple[str, str, str]:
    value = case.get("source")
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) or not value[key]
                                          for key in ("run_id", "session_id", "request_id")):
        raise ValueError("Case requires a source run, session and request")
    return tuple(value[key] for key in ("run_id", "session_id", "request_id"))


def _payload(messages: list[dict]) -> dict:
    if (not isinstance(messages, list) or len(messages) != 2
            or [message.get("role") for message in messages if isinstance(message, dict)] != ["system", "user"]):
        raise ValueError("Case requires frozen system and user messages")
    for message in messages:
        ProviderMessage.model_validate(message)
    if _message_size(messages) > MAX_INPUT_BYTES:
        raise ValueError("Frozen input exceeds byte ceiling")
    payload = json.loads(messages[1]["content"], object_pairs_hook=_unique)
    if not isinstance(payload, dict):
        raise ValueError("User message must be a JSON object")
    return payload


def _without_history(payload: dict) -> dict:
    """Different role lengths may trim different amounts of the same history."""
    return {key: value for key, value in payload.items()
            if key not in {"history", "history_complete", "history_omitted"}}


def validate_plan(plan: dict) -> None:
    if (type(plan) is not dict or type(plan.get("global_call_limit")) is not int
            or plan["global_call_limit"] != LIMIT or plan.get("model") != "deepseek-flash"
            or not isinstance(plan.get("cases"), list) or len(plan["cases"]) != LIMIT
            or any(not isinstance(case, dict) for case in plan["cases"])):
        raise ValueError("Plan requires exactly 15 cases and the fixed DeepSeek model")
    if Counter(case.get("group") for case in plan["cases"]) != {"main": 3, "review": 2, "offline": 10}:
        raise ValueError("Plan requires 3 main, 2 review and 10 offline cases")
    previous: dict[str, dict] = {}
    for case in plan["cases"]:
        identity, group = case.get("case_id", ""), case["group"]
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity) or identity in previous:
            raise ValueError("Case IDs must be unique safe filenames")
        source = _source(case)
        ceiling = 1600 if group == "offline" else 384
        if type(case.get("max_output_tokens")) is not int or case["max_output_tokens"] != ceiling:
            raise ValueError("Output limit must match the frozen case ceiling")
        if type(case.get("timeout_s")) not in {int, float} or case["timeout_s"] != 20:
            raise ValueError("Case timeout must be 20 seconds")
        version = OFFLINE_PROMPT if group == "offline" else ONLINE_PROMPT
        payload = _payload(case.get("messages"))
        if case.get("prompt_version") != version or payload.get("prompt_version") != version:
            raise ValueError("Case must use the factual-repair prompt version")
        expected = case.get("expected")
        field = "request_ids" if group == "offline" else "candidate_ids"
        ids = expected.get(field) if isinstance(expected, dict) else None
        if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("Expected IDs must be nonempty and unique")
        if group == "offline":
            decisions = payload.get("decisions")
            if (not isinstance(decisions, list) or any(not isinstance(item, dict) for item in decisions)
                    or [item.get("request_id") for item in decisions] != ids or source[2] not in ids
                    or payload.get("scope") != "selected_requests_only"):
                raise ValueError("Offline request projection must match expected source requests")
        else:
            candidates = payload.get("candidates")
            if (not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates)
                    or [item.get("action_id") for item in candidates] != ids
                    or payload.get("role", "main") != group or not isinstance(payload.get("semantic_facts"), dict)):
                raise ValueError("Online role and candidate projection must match the case")
            if group == "main" and any(key in payload for key in ("main", "trigger_codes", "validation_error")):
                raise ValueError("Main must not contain secondary-role context")
        dependency = case.get("dependency")
        if group != "review":
            if dependency is not None or "model_input" in case:
                raise ValueError("Only reviews may carry dependency input")
        else:
            if not isinstance(dependency, dict) or set(dependency) != {"main_case_id"}:
                raise ValueError("Review requires a main dependency, without a seed")
            prior = previous.get(dependency["main_case_id"])
            if prior is None or prior["group"] != "main" or _source(prior) != source:
                raise ValueError("Review must depend on an earlier main for the same source request")
            model_input = ModelInput.model_validate(case.get("model_input"))
            binding = model_input.binding
            if (model_input.role != "review" or (binding.run_id, binding.session_id, binding.request_id) != source
                    or [item.action_id for item in model_input.candidates] != ids
                    or prior["expected"]["candidate_ids"] != ids):
                raise ValueError("Review model input must match its source and main candidates")
            rebuilt, _, _ = build_messages(model_input, max_input_bytes=MAX_INPUT_BYTES, prompt_version=ONLINE_PROMPT)
            if [message.model_dump(mode="json") for message in rebuilt] != case["messages"]:
                raise ValueError("Review frozen messages must match its complete model input")
            main_payload = _payload(prior["messages"])
            if (main_payload["candidates"] != payload["candidates"]
                    or _without_history(main_payload["visible"]) != _without_history(payload["visible"])
                    or _without_history(main_payload["semantic_facts"]) != _without_history(payload["semantic_facts"])
                    or main_payload.get("platform_context") != payload.get("platform_context")):
                raise ValueError("Review and main must describe the same public state and candidates")
        previous[identity] = case


def _messages(case: dict, results: dict) -> tuple[list[dict], int, int | None] | None:
    if case["group"] == "review":
        decision = results.get(case["dependency"]["main_case_id"], {}).get("technical", {}).get("decision")
        if decision is None:
            return None
        value = ModelInput.model_validate(case["model_input"]).model_copy(update={"main": ModelDecision.model_validate(decision)})
        messages, size, omitted = build_messages(value, max_input_bytes=MAX_INPUT_BYTES, prompt_version=ONLINE_PROMPT)
        return [message.model_dump(mode="json") for message in messages], size, omitted
    messages = json.loads(_json(case["messages"]))
    payload = json.loads(messages[-1]["content"])
    return messages, _message_size(messages), payload.get("visible", {}).get("history_omitted")


async def run_plan(plan: dict, output: Path, *, mode: str, provider: ProviderPort) -> dict:
    validate_plan(plan)
    if mode not in {"live", "scripted"}:
        raise ValueError("An explicit provider mode is required")
    plan = json.loads(_json(plan))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "exclusive.lock").open("x", encoding="utf-8") as lock:
        lock.write("Never resume this directory or replace its lock.\n")
    _write(output / "plan.json", plan)
    results: dict[str, dict] = {}
    started = 0
    stop_reason = "interrupted"
    close_error = None

    def checkpoint() -> dict:
        replies = [row["reply"] for row in results.values() if row.get("reply") is not None]
        usage, known = {}, {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [row["usage"][field] for row in replies]
            known[field] = sum(value for value in values if value is not None)
            usage[field] = sum(values) if values and all(value is not None for value in values) and len(values) == started else None
        counts = [row["transport_requests"] for row in replies]
        summary = {"mode": mode, "evaluation_kind": "archived_input_replay", "real_platform": False,
                   "global_call_limit": LIMIT, "calls_started": started, "results_recorded": len(replies),
                   "cases_recorded": len(results), "skipped_cases": sum(row["event"] == "skipped" for row in results.values()),
                   "stop_reason": stop_reason, "scheduled_cases": LIMIT, "remaining_cases": LIMIT - len(results),
                   "real_provider": mode == "live", "usage": usage, "known_usage_subtotal": known,
                   "transport_requests": sum(counts) if counts and all(v is not None for v in counts) and len(counts) == started else None,
                   "known_transport_requests": sum(v for v in counts if v is not None), "cost_amount": None,
                   "provider_errors": sum(row["error"] is not None for row in replies),
                   "action_schema_valid": sum(row.get("technical", {}).get("schema_valid") is True for row in results.values()),
                   "candidate_members": sum(row.get("technical", {}).get("candidate_member") is True for row in results.values()),
                   "semantic_judgment": "not_assessed", "provider_close_error": close_error,
                   "plan_sha256": hashlib.sha256(_json(plan).encode()).hexdigest()}
        _write(output / "summary.json", summary)
        return summary

    with (output / "calls.jsonl").open("x", encoding="utf-8") as journal:
        def append(row: dict) -> None:
            journal.write(_json(row) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        try:
            checkpoint()
            for case in plan["cases"]:
                projection = _messages(case, results)
                if projection is None:
                    row = {"event": "skipped", "case_id": case["case_id"], "group": case["group"],
                           "reason": "review_dependency_invalid", "reply": None, "dependency": case["dependency"]}
                    results[case["case_id"]] = row
                    _write(output / f"{case['case_id']}.result.json", row)
                    append(row)
                    checkpoint()
                    print(_json({"case": case["case_id"], "calls_started": started, "error": "REVIEW_DEPENDENCY_INVALID"}), flush=True)
                    continue
                if started >= LIMIT:
                    raise RuntimeError("Fixed call budget exhausted")
                messages, input_bytes, history_omitted = projection
                request = {"call_id": case["case_id"], "model": plan["model"], "messages": messages,
                           "max_output_tokens": case["max_output_tokens"], "timeout_s": case["timeout_s"],
                           "input_bytes": input_bytes, "history_omitted": history_omitted}
                request_hash = hashlib.sha256(_json(request).encode()).hexdigest()
                _write(output / f"{case['case_id']}.input.json", request)
                started += 1
                append({"event": "begin", "ordinal": started, "case_id": case["case_id"], "request_sha256": request_hash})
                checkpoint()
                start = time.monotonic()
                try:
                    request_input = ProviderInput(call_id=case["case_id"],
                        messages=tuple(ProviderMessage.model_validate(m) for m in messages),
                        deadline_mono=start + case["timeout_s"], max_output_tokens=case["max_output_tokens"])
                    async with asyncio.timeout(case["timeout_s"]):
                        reply = await provider.complete(request_input)
                    if reply.call_id != case["case_id"]:
                        raise ValueError("Provider reply binding mismatch")
                except asyncio.CancelledError:
                    stop_reason = "cancelled"
                    raise
                except Exception:
                    reply = ProviderReply(call_id=case["case_id"], content=None, usage=unknown_usage(),
                        transport_requests=None, elapsed_ms=(time.monotonic() - start) * 1000,
                        error=safe_error("RETEST_CALL_FAILED", "Retest call failed; details are not exposed and no retry was attempted."))
                row = {"event": "result", "case_id": case["case_id"], "group": case["group"],
                       "ordinal": started, "request_sha256": request_hash,
                       "reply": reply.model_dump(mode="json"), "technical": check_output(case, reply)}
                results[case["case_id"]] = row
                _write(output / f"{case['case_id']}.result.json", row)
                append(row)
                checkpoint()
                print(_json({"case": case["case_id"], "calls_started": started,
                             "error": reply.error.code if reply.error else None,
                             "transport_requests": reply.transport_requests, "total_tokens": reply.usage.total_tokens}), flush=True)
                if reply.transport_requests is not None and reply.transport_requests > 1:
                    stop_reason = "transport_budget_violation"
                    break
                if reply.error and reply.error.safe_message in {
                    "Provider request failed with HTTP status 401.", "Provider request failed with HTTP status 402.",
                }:
                    stop_reason = "provider_auth_or_credit_failure"
                    break
            else:
                stop_reason = "schedule_completed"
        finally:
            try:
                async with asyncio.timeout(2):
                    await provider.aclose()
            except Exception:
                close_error = "PROVIDER_CLOSE_FAILED"
            finally:
                summary = checkpoint()
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/retest15/plan.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"), object_pairs_hook=_unique)
        validate_plan(plan)
        if not args.live:
            print(_json({"mode": "check", "cases": LIMIT, "starts_calls": False}))
            return 0
        if args.output is None or args.output.exists():
            raise ValueError("Live mode requires a new output directory")
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("Missing process model credential")
        from guandan_agent.model.deepseek import DeepSeekProvider
        provider = DeepSeekProvider(api_key=key, model=plan["model"])
        summary = asyncio.run(run_plan(plan, args.output, mode="live", provider=provider))
        return 0 if summary["stop_reason"] == "schedule_completed" else 1
    except KeyboardInterrupt:
        print('{"error":"RETEST_INTERRUPTED"}')
        return 130
    except Exception:
        print('{"error":"RETEST_SETUP_OR_STORAGE_FAILED","details":"No automatic retry or resume."}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
