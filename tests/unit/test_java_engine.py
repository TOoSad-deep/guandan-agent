"""Own-residual worker isolation and bounded process lifetime."""
import asyncio
import json
from pathlib import Path

import pytest

from guandan_agent.clock import SystemClock
from guandan_agent.hand import EngineError, JavaActionEngine


class Stdin:
    def __init__(self):
        self.payloads = []
        self.closed = False
    def write(self, data):
        self.payloads.append(json.loads(data))
    async def drain(self):
        pass
    def close(self):
        self.closed = True


class Process:
    def __init__(self):
        self.stdin = Stdin()
        self.stdout = asyncio.StreamReader()
        self.returncode = None
        self.terminates = self.kills = 0
        self.stopped = asyncio.Event()
    def terminate(self):
        self.terminates += 1
        # Model a worker that ignores TERM so bounded KILL is exercised.
    def kill(self):
        self.kills += 1
        self.returncode = -9
        self.stdout.feed_eof()
        self.stopped.set()
    async def wait(self):
        await self.stopped.wait()
        return self.returncode


def test_timeout_disables_run_and_cleans_worker_without_restart(monkeypatch):
    async def scenario():
        process, starts = Process(), []
        async def spawn(*args, **kwargs):
            starts.append((args, kwargs))
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        engine = JavaActionEngine("test.jar", cleanup_timeout_s=.01)
        with pytest.raises(EngineError) as error:
            await engine.analyze(("S9", "S9"), "2", SystemClock().monotonic() + .02)
        assert error.value.code == "TOOL_TIMEOUT" and engine.disabled
        await engine.aclose()
        assert process.terminates == process.kills == 1
        assert process.stdin.closed
        with pytest.raises(EngineError):
            await engine.analyze(("S3",), "2", SystemClock().monotonic() + 1)
        assert len(starts) == 1
        sent = process.stdin.payloads[0]
        assert sent == {"id": 1, "mode": "first", "handCards": ["S9", "S9"], "heartsNum": 0, "currentRank": "2"}
    asyncio.run(scenario())


def test_wrong_response_id_and_unowned_actions_are_rejected(monkeypatch):
    async def scenario(reply):
        process = Process()
        process.stdout.feed_data(json.dumps(reply).encode() + b"\n")
        async def spawn(*_args, **_kwargs):
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        engine = JavaActionEngine("test.jar", cleanup_timeout_s=.01)
        with pytest.raises(EngineError):
            await engine.analyze(("S9",), "2", SystemClock().monotonic() + 1)
        assert engine.disabled
        await engine.aclose()
    asyncio.run(scenario({"id": 99, "ok": True, "actions": [["Single", "9", ["S9"]]]}))
    asyncio.run(scenario({"id": 1, "ok": True, "actions": [["Pair", "9", ["S9", "S9"]]]}))


def test_queue_wait_counts_against_deadline(monkeypatch):
    async def scenario():
        process = Process()
        async def spawn(*_args, **_kwargs):
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        engine = JavaActionEngine("test.jar", cleanup_timeout_s=.01)
        first = asyncio.create_task(engine.analyze(("S9",), "2", SystemClock().monotonic() + 1))
        await asyncio.sleep(0)
        with pytest.raises(EngineError) as error:
            await engine.analyze(("S3",), "2", SystemClock().monotonic() + .02)
        assert error.value.code == "TOOL_TIMEOUT"
        await engine.aclose()
        with pytest.raises(EngineError):
            await first
        assert len(process.stdin.payloads) == 1
    asyncio.run(scenario())


def test_real_java_residual_hand_smoke():
    jar = Path("runtime/platform/workspace/guandan/target/guandan-java-action.jar")
    java = Path("/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home/bin/java")
    if not jar.is_file() or not java.is_file():
        pytest.skip("locally prepared official Java JAR / JDK 21 unavailable")
    async def scenario():
        engine = JavaActionEngine(jar, java_executable=str(java))
        try:
            actions = await engine.analyze(("S9", "S9", "D4"), "2", SystemClock().monotonic() + 5)
            assert ("Pair", "9", ("S9", "S9")) in actions
            assert all(action[0] != "PASS" for action in actions)
        finally:
            await engine.aclose()
    asyncio.run(scenario())
