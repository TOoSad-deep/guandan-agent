#!/usr/bin/env python3
"""One-shot B10 final matches. No retries, replenishment, or model calibration.

Production RunManager owns model budgets and all action authority. This wrapper
only verifies the frozen files, owns a local Java process, and journals starts.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from guandan_agent.app import run_config
from guandan_agent.config import load_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/b10"
JAVA = Path("/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home/bin/java")
JAR = ROOT / "runtime/platform/workspace/guandan-websocket/target/guandan-java-pure-1.0.0.jar"
LIMIT_S = 1200


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate(version: str):
    plan = json.loads((OUTPUT / "frozen-plan.json").read_text())
    for name, expected in plan["frozen_files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    config = load_config(ROOT / f"configs/b10-{version.lower()}.toml")
    assert config.config_id == plan["configs"][version]["config_id"]
    assert config.run.split == "final" and config.run.matches == 1
    assert [s.seat for s in config.run.seats] == [0, 1, 2, 3]
    assert [s.policy for s in config.run.seats] == [version, "V0", "V0", "V0"]
    assert config.batch.max_model_calls == (0 if version == "V0" else 20)
    assert config.offline.max_model_calls == 0 and config.model.max_retries == 0
    assert not config.model.thinking
    assert config.review.enabled == (version == "V2")
    assert config.platform.url == "ws://127.0.0.1:18181"
    assert plan["max_runtime_s_per_match"] == LIMIT_S
    return config


def listening() -> bool:
    with socket.socket() as connection:
        connection.settimeout(.15)
        return connection.connect_ex(("127.0.0.1", 18181)) == 0


def run(version: str) -> int:
    config = validate(version)
    if version != "V0" and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("Model credentials are missing")
    if listening():
        raise ValueError("Port 18181 is already owned by another process")
    # Exclusive creation permanently consumes this version's start, including
    # failures. A second invocation must not silently reset its twenty slots.
    case = OUTPUT / "live" / version
    case.mkdir(parents=True, exist_ok=False)
    save(case / "start.json", {"version": version, "config_id": config.config_id,
         "started_at": datetime.now(timezone.utc).isoformat(), "max_matches": 1,
         "max_model_calls": config.batch.max_model_calls, "max_runtime_s": LIMIT_S,
         "provenance": "local official snapshot, production OpenGuanDanPlatform and DeepSeekProvider"})
    server_env = {key: value for key, value in os.environ.items()
                  if not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    server = None
    result = {"version": version, "status": "not_started", "run_id": None,
              "error_code": None, "server_stopped": False}
    try:
        with (case / "server.log").open("wb") as log:
            server = subprocess.Popen([str(JAVA), "-Dserver.port=18181", "-jar", str(JAR)],
                                      cwd=case, stdout=log, stderr=subprocess.STDOUT, env=server_env)
            ready_deadline = time.monotonic() + 20
            while not listening():
                if server.poll() is not None or time.monotonic() >= ready_deadline:
                    raise RuntimeError("SERVER_NOT_READY")
                time.sleep(.1)
            result["server_pid"] = server.pid
            save(case / "server-ready.json", {"pid": server.pid, "listen": "127.0.0.1:18181",
                                                "jar_sha256": sha(JAR)})
            final = asyncio.run(run_config(config, recording_root=ROOT / "runs",
                                           startup_timeout_s=15, max_runtime_s=LIMIT_S))
            result.update(status=final.status, run_id=final.run_id,
                          view=final.model_dump(mode="json"))
    except KeyboardInterrupt:
        result["error_code"] = "INTERRUPTED"
    except Exception as exc:
        # No exception text, credential, response body or environment is logged.
        result["error_code"] = type(exc).__name__
    finally:
        if server is not None:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            result["server_exit_code"] = server.returncode
            result["server_stopped"] = server.poll() is not None
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(case / "result.json", result)
    print(json.dumps({key: result[key] for key in ("version", "status", "run_id", "error_code", "server_stopped")}))
    return 0 if result["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=("V0", "V1", "V2"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    validate(args.version)
    if not args.live:
        print(json.dumps({"version": args.version, "frozen_inputs_valid": True, "network_started": False}))
        return 0
    return run(args.version)


if __name__ == "__main__":
    sys.exit(main())
