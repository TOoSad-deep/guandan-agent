"""Offline behavioral checks. No model credentials or network are used."""
import asyncio
import copy
import json
import unittest

from probe_model import DEFAULT_FIXTURES, LiveCaller, ModelProbe, ProbeConfig, RequestLease, ScriptedCaller, public_snapshot


def output(action_id):
    return json.dumps({"action_id": action_id, "reason": "测试候选选择。"}, ensure_ascii=False)


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cases = json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8"))["cases"]
        self.config = ProbeConfig(total_budget_s=1, call_timeout_s=0.2, reserve_s=0.01, min_review_budget_s=0.01)

    async def run_case(self, case, steps, config=None):
        lease = RequestLease(case["request_version"])
        caller = ScriptedCaller(steps, lease)
        result = await ModelProbe(caller, config or self.config, lease).run(case)
        self.assertLessEqual(caller.invocations, (config or self.config).max_calls)
        self.assertEqual(result["transport_requests"], 0)
        self.assertEqual(result["platform_submission"], "not_performed")
        return result

    async def test_normal_one_call(self):
        result = await self.run_case(self.cases[0], [{"content": output("single_9")}])
        self.assertEqual((result["status"], result["model_attempts"], result["review_used"]), ("selected", 1, False))

    async def test_combined_risks_review_once(self):
        result = await self.run_case(self.cases[1], [{"content": output("bomb_9")}, {"content": output("single_a")}])
        self.assertEqual(result["model_attempts"], 2)
        self.assertEqual(result["triggers"], ["opponent_near_finish", "use_bomb"])
        self.assertEqual(result["action_id"], "single_a")

    async def test_invalid_id_repaired_once(self):
        result = await self.run_case(self.cases[0], [{"content": output("not_a_candidate")}, {"content": output("single_9")}])
        self.assertEqual(result["model_attempts"], 2)
        self.assertEqual(result["calls"][0]["status"], "invalid_candidate_id")
        self.assertEqual(result["status"], "selected")

    async def test_second_invalid_output_falls_back_without_third(self):
        result = await self.run_case(self.cases[0], [{"content": "{}"}, {"content": "not json"}, {"content": output("single_9")}])
        self.assertEqual((result["status"], result["model_attempts"]), ("fallback", 2))

    async def test_v1_never_repairs(self):
        config = ProbeConfig(total_budget_s=1, call_timeout_s=.2, reserve_s=.01, min_review_budget_s=.01, max_calls=1)
        result = await self.run_case(self.cases[0], [{"content": output("bad")}], config)
        self.assertEqual((result["status"], result["model_attempts"]), ("fallback", 1))

    async def test_timeout_falls_back_and_late_result_cannot_replace(self):
        config = ProbeConfig(total_budget_s=.5, call_timeout_s=.01, reserve_s=.01, min_review_budget_s=.01)
        result = await self.run_case(self.cases[0], [{"delay_s": .1, "content": output("single_j")}], config)
        self.assertEqual(result["calls"][0]["status"], "timeout")
        self.assertEqual((result["status"], result["action_id"]), ("fallback", "single_9"))
        await asyncio.sleep(.02)
        self.assertEqual(result["action_id"], "single_9")

    async def test_expired_response_never_uses_old_fallback(self):
        result = await self.run_case(self.cases[0], [{"content": output("single_9"), "expire": True}])
        self.assertEqual(result["status"], "expired")
        self.assertIsNone(result["action_id"])

    async def test_failure_counts_and_does_not_log_exception_body(self):
        result = await self.run_case(self.cases[0], [{"raise": True}])
        self.assertEqual((result["status"], result["model_attempts"]), ("fallback", 1))
        self.assertNotIn("scripted failure", json.dumps(result))

    async def test_review_failure_uses_rule_fallback(self):
        result = await self.run_case(self.cases[1], [{"content": output("bomb_9")}, {"raise": True}])
        self.assertEqual((result["status"], result["model_attempts"]), ("fallback", 2))
        self.assertEqual(result["action_id"], "single_a")

    async def test_insufficient_review_time_keeps_valid_main(self):
        config = ProbeConfig(total_budget_s=.5, call_timeout_s=.1, reserve_s=.01, min_review_budget_s=1)
        result = await self.run_case(self.cases[1], [{"content": output("single_k")}], config)
        self.assertEqual((result["status"], result["model_attempts"]), ("selected", 1))
        self.assertFalse(result["review_used"])
        self.assertEqual(result["action_id"], "single_k")
        self.assertEqual(result["events"][0]["event"], "review_skipped")

    async def test_unique_input_candidate_skips_model(self):
        case = copy.deepcopy(self.cases[0])
        case["candidates"] = [case["candidates"][2]]
        result = await self.run_case(case, [])
        self.assertEqual((result["status"], result["model_attempts"], result["source"]), ("selected", 0, "unique_candidate"))

    async def test_empty_candidates_never_invents_action(self):
        case = copy.deepcopy(self.cases[0])
        case["candidates"] = []
        result = await self.run_case(case, [])
        self.assertEqual(result["status"], "empty_candidates")
        self.assertIsNone(result["action_id"])

    def test_hidden_fixture_information_is_not_projected(self):
        first, second = copy.deepcopy(self.cases[0]), copy.deepcopy(self.cases[0])
        first["observation"]["other_hands"] = [["HA"]]
        second["observation"]["other_hands"] = [["SK"]]
        self.assertEqual(public_snapshot(first), public_snapshot(second))

    async def test_deepseek_json_mode_request_with_mock_http_only(self):
        import httpx

        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "mock-completion", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output("single_9")}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        caller = LiveCaller("test-key-not-real", "deepseek-flash", 1, 128, False,
                            http_transport=httpx.MockTransport(handler))
        try:
            case = self.cases[0]
            result = await ModelProbe(caller, self.config, RequestLease(case["request_version"])).run(case)
            self.assertEqual(result["status"], "selected")
            self.assertEqual(result["transport_requests"], 1)
            self.assertEqual(result["calls"][0]["usage"]["total_tokens"], 15)
            self.assertEqual(bodies[0]["response_format"], {"type": "json_object"})
            self.assertEqual(bodies[0]["model"], "deepseek-flash")
            self.assertEqual(bodies[0]["thinking"], {"type": "disabled"})
            self.assertNotIn("tools", bodies[0])
            self.assertNotIn("tool_choice", bodies[0])
            self.assertEqual(caller.model.max_retries, 0)
            self.assertNotIn("test-key-not-real", json.dumps(result))
        finally:
            await caller.close()

    async def test_http_failure_has_one_attempt_no_hidden_retries(self):
        import httpx

        caller = LiveCaller("test-key-not-real", "deepseek-flash", 1, 128, False,
                            http_transport=httpx.MockTransport(lambda request: httpx.Response(500, json={"error": {"message": "do-not-log-provider-body"}})))
        try:
            case = self.cases[0]
            result = await ModelProbe(caller, self.config, RequestLease(case["request_version"])).run(case)
            self.assertEqual((result["status"], result["model_attempts"], result["transport_requests"]), ("fallback", 1, 1))
            self.assertNotIn("do-not-log-provider-body", json.dumps(result))
        finally:
            await caller.close()


if __name__ == "__main__":
    unittest.main()
