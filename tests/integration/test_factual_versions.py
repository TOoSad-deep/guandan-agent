"""Current semantic prompt reaches both V2 roles without changing call limits."""
import asyncio
import json

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.hand import HandTools
from guandan_agent.model import ScriptedProvider
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.scheduling import AsyncioScheduler
from scripts.probe_b07_review import FIXTURE, FixedScenarioPlatform


def test_current_default_prompt_composes_with_v2_and_two_call_budget(tmp_path):
    async def scenario():
        clock = SystemClock()
        platform = FixedScenarioPlatform(json.loads(FIXTURE.read_bytes()), clock=clock)
        provider = ScriptedProvider([
            '{"action_id":"L0002","reason":"固定主建议"}',
            '{"action_id":"L0001","reason":"固定复核建议"}',
        ], clock=clock)
        config = RunConfig.model_validate({
            "run": {"seats": [{"seat": 0, "user_name": "semantic-test", "policy": "V2"}]},
            "batch": {"max_model_calls": 2}, "offline": {"max_model_calls": 0},
        })
        manager = RunManager(platform_factory=lambda _: platform, clock=clock,
                             scheduler=AsyncioScheduler(clock), recording_root=tmp_path,
                             tools_factory=lambda _: HandTools(clock=clock),
                             provider_factory=lambda _: provider)
        run = manager.start(config)
        try:
            async with asyncio.timeout(5):
                while True:
                    view = manager.view(run.run_id)
                    if view.sessions:
                        state = manager.session_view(run.run_id, view.sessions[0].session_id)
                        if state.latest_execution and state.latest_execution.status == "confirmed":
                            break
                    await asyncio.sleep(0.005)
        finally:
            manager.stop(run.run_id)
            await manager.wait(run.run_id)
        assert len(provider.inputs) == 2 and platform.writes == 1
        payloads = [json.loads(value.messages[1].content) for value in provider.inputs]
        assert all(value["prompt_version"] == "b09-facts-v3" for value in payloads)
        assert all("semantic_facts" in value for value in payloads)
        assert payloads[1]["role"] == "review"
        assert payloads[0]["candidates"] == payloads[1]["candidates"]
    asyncio.run(scenario())
