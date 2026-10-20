"""Offline authorization, cancellation and accounting without network access."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import ErrorInfo, ProviderMessage, ProviderReply, ReviewReport, RunSummary, Usage
from guandan_agent.model.provider import unknown_usage
from guandan_agent.recording import ClosedRunLog
from guandan_agent.runtime.manager import RunManager, RunManagerError


MESSAGES = (ProviderMessage(role="user", content='Return JSON {"items":[]}.'),)


def config(**overrides):
    values = {"offline": {"max_model_calls": 2},
              "model": {"timeout_s": 0.2, "concurrency": 1}}
    for key, value in overrides.items():
        values.setdefault(key, {}).update(value)
    return RunConfig.model_validate(values)


def source(*, closed=True, complete=True, evicted=0):
    source_config = RunConfig()
    run_id = str(uuid4())
    summary = RunSummary(
        schema_version="1.0", run_id=run_id, config_id=source_config.config_id,
        closed=closed, log_complete=complete, last_persisted_seq=0,
        pending_call_ids=(), unconfirmed_submission_ids=(), error=None,
    )
    return ClosedRunLog(run_id=run_id, config_id=source_config.config_id, events=(),
                        summary=summary, evicted_through=evicted, max_query_limit=200,
                        config=source_config)


def report(port, archive, *, status="completed", usage=None, errors=()):
    return ReviewReport(report_id=port.task_id, run_id=archive.run_id, config_id=archive.config_id,
                        status=status, created_at="2026-10-19T00:00:00Z", items=(),
                        usage=usage or unknown_usage(), errors=errors)


class Provider:
    def __init__(self, function=None):
        self.function = function
        self.inputs = []
        self.closed = False

    async def complete(self, input):
        self.inputs.append(input)
        if self.function is not None:
            return await self.function(input)
        return ProviderReply(call_id=input.call_id, content='{"items":[]}',
                             usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10),
                             transport_requests=1, elapsed_ms=1.0, error=None)

    async def aclose(self):
        self.closed = True


def manager(tmp_path, factory=None, shutdown=0.02):
    def no_platform(_):
        raise AssertionError("offline review must not construct a platform")
    return RunManager(platform_factory=no_platform, clock=SystemClock(), scheduler=object(),
                      provider_factory=factory, recording_root=tmp_path,
                      shutdown_timeout_s=shutdown)


def one_call(archive, replies=None):
    async def job(port):
        reply = await port.call(MESSAGES)
        if replies is not None:
            replies.append(reply)
        return report(port, archive, status="partial" if reply.error else "completed",
                      usage=reply.usage, errors=() if reply.error is None else (reply.error,))
    return job


async def spin(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


def test_provider_is_lazy_and_empty_job_does_not_consume_quota(tmp_path):
    async def scenario():
        created = []
        runtime = manager(tmp_path, lambda cfg: created.append(cfg))
        archive = source()
        async def job(port):
            return report(port, archive)
        view = runtime.start_review(config(), archive, job)
        assert view.status == "running" and not view.report_ready and not created
        result = await runtime.wait_review(view.task_id)
        assert result.status == "completed" and result.usage.total_tokens == 0
        assert runtime.review_view(view.task_id).report_ready
        assert runtime.offline_usage() == {"max_slots": 2, "slots_used": 0, "calls_started": 0}
        await runtime.close_reviews()
        assert not created
    asyncio.run(scenario())


def test_two_sources_share_permanent_offline_budget_and_no_online_identity(tmp_path):
    async def scenario():
        providers = []
        def factory(cfg):
            provider = Provider()
            providers.append(provider)
            return provider
        runtime = manager(tmp_path, factory)
        results = []
        for _ in range(3):
            archive = source()
            view = runtime.start_review(config(), archive, one_call(archive, results))
            result = await runtime.wait_review(view.task_id)
            assert result.run_id == archive.run_id and result.config_id == archive.config_id
        assert [result.error is None for result in results] == [True, True, False]
        assert results[-1].error.code == "OFFLINE_BUDGET"
        assert runtime.offline_usage() == {"max_slots": 2, "slots_used": 2, "calls_started": 2}
        assert len(providers) == 2
        assert all(set(provider.inputs[0].model_dump()) ==
                   {"call_id", "messages", "deadline_mono", "max_output_tokens"} for provider in providers)
        assert all(":o1" in provider.inputs[0].call_id for provider in providers)
        await runtime.close_reviews()
        assert all(provider.closed for provider in providers)
    asyncio.run(scenario())


@pytest.mark.parametrize("overrides", [
    {"offline": {"max_model_calls": 3}}, {"model": {"name": "different"}},
    {"model": {"max_output_tokens": 500}}, {"model": {"concurrency": 2}},
    {"versions": {"prompt": "changed-prompt"}},
])
def test_explicit_offline_batch_cannot_be_reset_or_reconfigured(tmp_path, overrides):
    async def scenario():
        runtime = manager(tmp_path)
        archive = source()
        async def job(port):
            return report(port, archive)
        first = runtime.start_review(config(), archive, job)
        await runtime.wait_review(first.task_id)
        with pytest.raises(RunManagerError) as caught:
            runtime.start_review(config(**overrides), archive, job)
        assert caught.value.error.code == "OFFLINE_CONFIG_CONFLICT"
        await runtime.close_reviews()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["open", "truncated", "identity"])
def test_invalid_source_never_calls_provider(tmp_path, mode):
    async def scenario():
        runtime = manager(tmp_path, lambda cfg: pytest.fail("provider constructed"))
        archive = source(closed=mode != "open", evicted=1 if mode == "truncated" else 0)
        if mode == "identity":
            archive.run_id = str(uuid4())
        with pytest.raises(RunManagerError) as caught:
            runtime.start_review(config(), archive, one_call(archive))
        assert caught.value.error.code in {"OFFLINE_SOURCE_OPEN", "OFFLINE_SOURCE_TRUNCATED", "OFFLINE_SOURCE_INVALID"}
        assert runtime.offline_usage()["slots_used"] == 0
    asyncio.run(scenario())


def test_pending_marker_is_rejected_even_when_summary_claims_closed(tmp_path):
    async def scenario():
        import shutil
        fixture = Path("tests/fixtures/audit/valid_scripted_run")
        target = tmp_path / "source"
        shutil.copytree(fixture, target)
        (target / "summary.pending").write_text("pending")
        archive = await ClosedRunLog.load(target)
        runtime = manager(tmp_path, lambda cfg: pytest.fail("provider constructed"))
        with pytest.raises(RunManagerError) as caught:
            runtime.start_review(config(), archive, one_call(archive))
        assert caught.value.error.code == "OFFLINE_SOURCE_OPEN"
        assert (target / "summary.pending").exists()
    asyncio.run(scenario())


def test_single_active_offline_job_and_stop_before_start(tmp_path):
    async def scenario():
        runtime = manager(tmp_path, lambda cfg: pytest.fail("provider constructed"))
        archive = source()
        view = runtime.start_review(config(), archive, one_call(archive))
        with pytest.raises(RunManagerError) as caught:
            runtime.start_review(config(), archive, one_call(archive))
        assert caught.value.error.code == "OFFLINE_CONFLICT"
        stopped = runtime.stop_review(view.task_id, "secret or user input must not enter report")
        assert stopped.status == "partial" and stopped.report_ready
        result = await runtime.wait_review(view.task_id)
        assert "secret" not in result.model_dump_json()
        assert runtime.offline_usage()["slots_used"] == 0
        assert runtime.stop_review(view.task_id) == stopped
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_active_online_rejects_review_and_accepted_new_online_preempts_without_wait(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow(input):
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return ProviderReply(call_id=input.call_id, content="{}", usage=Usage(total_tokens=20),
                                 transport_requests=1, elapsed_ms=100.0, error=None)
        provider = Provider(slow)
        runtime = manager(tmp_path, lambda cfg: provider)
        archive = source()
        view = runtime.start_review(config(), archive, one_call(archive))
        await entered.wait()
        online = runtime.start(config(batch={"max_model_calls": 0}))
        closed = runtime.review_view(view.task_id)
        assert closed.status == "partial" and closed.calls_started == 1
        assert closed.calls[0].usage.total_tokens is None
        with pytest.raises(RunManagerError) as caught:
            runtime.start_review(config(), archive, one_call(archive))
        assert caught.value.error.code == "OFFLINE_ONLINE_ACTIVE"
        runtime.stop(online.run_id)
        await runtime.wait(online.run_id)
        await asyncio.wait_for(runtime.close_reviews(), 0.1)
        result = await runtime.wait_review(view.task_id)
        assert result.status == "partial" and result.usage.total_tokens is None
        assert runtime.offline_usage() == {"max_slots": 2, "slots_used": 1, "calls_started": 1}
        release.set()
        await asyncio.sleep(0.01)
        assert runtime.review_view(view.task_id) == closed
        assert await runtime.wait_review(view.task_id) == result
    asyncio.run(scenario())


def test_provider_failure_is_safe_and_consumes_slot_without_retry(tmp_path):
    async def scenario():
        async def fail(input):
            raise RuntimeError("Authorization: private-provider-key")
        provider = Provider(fail)
        runtime = manager(tmp_path, lambda cfg: provider)
        archive = source()
        view = runtime.start_review(config(), archive, one_call(archive))
        result = await runtime.wait_review(view.task_id)
        assert result.status == "partial" and "private-provider-key" not in result.model_dump_json()
        assert result.usage.total_tokens is None
        assert runtime.review_view(view.task_id).calls[0].transport_requests is None
        assert len(provider.inputs) == 1 and runtime.offline_usage()["slots_used"] == 1
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_provider_initialization_failure_reserves_but_does_not_claim_transmission(tmp_path):
    async def scenario():
        def fail(cfg):
            raise ValueError("secret")
        runtime = manager(tmp_path, fail)
        archive = source()
        view = runtime.start_review(config(), archive, one_call(archive))
        result = await runtime.wait_review(view.task_id)
        assert result.errors[0].code == "OFFLINE_PROVIDER_UNAVAILABLE"
        record = runtime.review_view(view.task_id).calls[0]
        assert record.calls_started == 0 and record.transport_requests == 0
        assert runtime.offline_usage() == {"max_slots": 2, "slots_used": 1, "calls_started": 0}
        await runtime.close_reviews()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["budget", "oversized", "empty"])
def test_preflight_denials_do_not_construct_provider(tmp_path, mode):
    async def scenario():
        runtime = manager(tmp_path, lambda cfg: pytest.fail("provider constructed"))
        archive = source()
        settings = config(offline={"max_model_calls": 0}) if mode == "budget" else config()
        replies = []
        async def job(port):
            messages = () if mode == "empty" else MESSAGES
            if mode == "oversized":
                messages = (ProviderMessage(role="user", content="中" * 12000),)
            replies.append(await port.call(messages))
            return report(port, archive, status="partial", errors=(replies[0].error,))
        view = runtime.start_review(settings, archive, job)
        await runtime.wait_review(view.task_id)
        assert replies[0].error.code == {"budget": "OFFLINE_BUDGET", "oversized": "INPUT_LIMIT",
                                        "empty": "OFFLINE_INPUT_INVALID"}[mode]
        assert runtime.offline_usage()["slots_used"] == 0
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_wrong_provider_identity_retains_unknown_usage(tmp_path):
    async def scenario():
        async def wrong(input):
            return ProviderReply(call_id="another-call", content="{}", usage=Usage(total_tokens=1),
                                 transport_requests=1, elapsed_ms=0.0, error=None)
        runtime = manager(tmp_path, lambda cfg: Provider(wrong))
        archive = source()
        view = runtime.start_review(config(), archive, one_call(archive))
        result = await runtime.wait_review(view.task_id)
        assert result.errors[0].code == "OFFLINE_REPLY_IDENTITY"
        assert result.usage.total_tokens is None
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_wrong_generated_report_identity_is_rejected(tmp_path):
    async def scenario():
        runtime = manager(tmp_path)
        archive = source()
        async def job(port):
            return report(port, archive).model_copy(update={"config_id": "invented"})
        view = runtime.start_review(config(), archive, job)
        result = await runtime.wait_review(view.task_id)
        assert result.status == "failed" and result.errors[0].code == "OFFLINE_REPORT_IDENTITY"
        assert result.config_id == archive.config_id
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_timeout_returns_finitely_and_late_reply_cannot_rewrite_closed_report(tmp_path):
    async def scenario():
        release = asyncio.Event()
        async def slow(input):
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return ProviderReply(call_id=input.call_id, content="{}", usage=Usage(total_tokens=10),
                                 transport_requests=1, elapsed_ms=40.0, error=None)
        runtime = manager(tmp_path, lambda cfg: Provider(slow))
        archive = source()
        view = runtime.start_review(config(model={"timeout_s": 0.02}), archive, one_call(archive))
        result = await asyncio.wait_for(runtime.wait_review(view.task_id), 0.15)
        final = runtime.review_view(view.task_id)
        assert result.status == "partial" and result.usage.total_tokens is None
        assert final.calls[0].error.code == "MODEL_TIMEOUT"
        await asyncio.wait_for(runtime.close_reviews(), 0.1)
        release.set()
        await asyncio.sleep(0.01)
        assert await runtime.wait_review(view.task_id) == result
        assert runtime.review_view(view.task_id) == final
    asyncio.run(scenario())


def test_cancelled_transport_keeps_shared_permit_across_jobs(tmp_path):
    async def scenario():
        release = asyncio.Event()
        providers = []
        async def slow(input):
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return ProviderReply(call_id=input.call_id, content="{}", usage=unknown_usage(),
                                 transport_requests=1, elapsed_ms=60.0, error=None)
        def factory(cfg):
            provider = Provider(slow)
            providers.append(provider)
            return provider
        runtime = manager(tmp_path, factory)
        archive = source()
        settings = config(model={"timeout_s": 0.02})
        first = runtime.start_review(settings, archive, one_call(archive))
        await runtime.wait_review(first.task_id)
        second = runtime.start_review(settings, archive, one_call(archive))
        result = await asyncio.wait_for(runtime.wait_review(second.task_id), 0.15)
        assert result.errors[0].code == "OFFLINE_TIMEOUT"
        assert len(providers) == 1
        assert runtime.offline_usage()["slots_used"] == 1
        release.set()
        await asyncio.sleep(0.01)
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_late_usage_fills_only_unknown_fields_while_task_still_open(tmp_path):
    async def scenario():
        release, waiting = asyncio.Event(), asyncio.Event()
        async def slow(input):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return ProviderReply(call_id=input.call_id, content="{}", usage=Usage(total_tokens=7),
                                 transport_requests=1, elapsed_ms=30.0, error=None)
        runtime = manager(tmp_path, lambda cfg: Provider(slow))
        archive = source()
        async def job(port):
            reply = await port.call(MESSAGES)
            assert reply.error.code == "MODEL_TIMEOUT"
            waiting.set()
            await release.wait()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return report(port, archive, status="partial", errors=(reply.error,))
        view = runtime.start_review(config(model={"timeout_s": 0.04}), archive, job)
        await waiting.wait()
        release.set()
        result = await runtime.wait_review(view.task_id)
        assert result.status == "partial" and result.usage.total_tokens == 7
        assert runtime.review_view(view.task_id).calls[0].error.code == "MODEL_TIMEOUT"
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_hung_generator_has_finite_task_deadline(tmp_path):
    async def scenario():
        runtime = manager(tmp_path)
        archive = source()
        async def job(port):
            await asyncio.Event().wait()
        view = runtime.start_review(config(model={"timeout_s": 0.01}), archive, job)
        result = await asyncio.wait_for(runtime.wait_review(view.task_id), 0.1)
        assert result.status == "partial" and result.errors[0].code == "OFFLINE_TIMEOUT"
        assert runtime.offline_usage()["slots_used"] == 0
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_partial_usage_does_not_infer_total_or_price(tmp_path):
    async def scenario():
        async def partial(input):
            return ProviderReply(call_id=input.call_id, content="{}",
                                 usage=Usage(input_tokens=8, output_tokens=2),
                                 transport_requests=None, elapsed_ms=1.0, error=None)
        runtime = manager(tmp_path, lambda cfg: Provider(partial))
        archive = source()
        view = runtime.start_review(config(), archive, one_call(archive))
        result = await runtime.wait_review(view.task_id)
        assert result.usage.input_tokens == 8 and result.usage.output_tokens == 2
        assert result.usage.total_tokens is None and result.usage.cost_amount is None
        await runtime.close_reviews()
    asyncio.run(scenario())


def test_closed_incomplete_source_forces_partial_even_if_generator_omits_warning(tmp_path):
    async def scenario():
        runtime = manager(tmp_path)
        archive = source(complete=False)
        async def job(port):
            return report(port, archive)
        view = runtime.start_review(config(), archive, job)
        result = await runtime.wait_review(view.task_id)
        assert result.status == "partial" and result.errors[0].code == "OFFLINE_SOURCE_INCOMPLETE"
        assert runtime.offline_usage()["slots_used"] == 0
        await runtime.close_reviews()
    asyncio.run(scenario())
