"""Explicit composition root for local V0/V1/V2; importing has no I/O."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager
from uuid import UUID

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import RunView
from guandan_agent.hand import HandTools, JavaActionEngine
from guandan_agent.java_runtime import java_executable
from guandan_agent.platform.adapter import OpenGuanDanPlatform
from guandan_agent.runtime.manager import RunManager
from guandan_agent.runtime.manager import RunManagerError
from guandan_agent.runtime.scheduling import AsyncioScheduler


def create_manager(*, recording_root: str | Path = "runs",
                   startup_timeout_s: float = 15.0) -> RunManager:
    clock = SystemClock()

    def tools(config: RunConfig) -> HandTools:
        # Explicit source-checkout runtime artifact; never import the all-hands
        # Python simulator. Missing JARs degrade through the bounded tool path.
        project = Path(__file__).resolve().parents[2]
        engine = None
        if config.tools.deep_eval_limit:
            engine = JavaActionEngine(project / "runtime/platform/workspace/guandan/target/guandan-java-action.jar",
                                      java_executable=java_executable(),
                                      clock=clock, cleanup_timeout_s=config.tools.engine_stop_grace_s)
        return HandTools(engine, clock=clock)

    def provider(config: RunConfig):
        # This factory is invoked only by an explicitly started model run, before
        # any game connection. Import/config-check/V0 do not read credentials.
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key or not api_key.strip():
            raise RunManagerError("MODEL_CREDENTIALS_MISSING", "Model policies require a configured process API key.")
        from guandan_agent.model.deepseek import DeepSeekProvider
        return DeepSeekProvider(api_key=api_key, model=config.model.name, clock=clock)

    return RunManager(
        platform_factory=lambda _: OpenGuanDanPlatform(clock=clock),
        clock=clock, scheduler=AsyncioScheduler(clock), recording_root=recording_root,
        startup_timeout_s=startup_timeout_s,
        tools_factory=tools, provider_factory=provider,
    )


async def run_config(config: RunConfig, *, recording_root: str | Path = "runs",
                     startup_timeout_s: float = 15.0,
                     max_runtime_s: float | None = None) -> RunView:
    """Run only after the CLI's explicit run command; cancellation stops seats."""
    manager = create_manager(recording_root=recording_root,
                             startup_timeout_s=startup_timeout_s)
    initial = manager.start(config)
    try:
        if max_runtime_s is None:
            return await manager.wait(initial.run_id)
        async with asyncio.timeout(max_runtime_s):
            return await manager.wait(initial.run_id)
    except TimeoutError:
        manager.stop(initial.run_id)
        return await manager.wait(initial.run_id)
    except asyncio.CancelledError:
        manager.stop(initial.run_id)
        # asyncio.run translates Ctrl-C into cancellation first. Let the bounded
        # close finish before propagating it to the CLI's normal interrupt exit.
        await manager.wait(initial.run_id)
        raise


def create_web_application(*, recording_root: str | Path = "runs",
                           config_root: str | Path = "configs",
                           allowed_config_names: tuple[str, ...] = ("local.example.toml",),
                           archive_root: str | Path | None = None,
                           allow_report_generation: bool = True,
                           review_config_name: str | None = "web-review.toml",
                           host: str = "127.0.0.1", port: int = 8000):
    """Explicit local serving composition; no game or provider starts on import/GET."""
    from guandan_agent.config import load_config
    from guandan_agent.recording.query import QueryError, QueryService
    from guandan_agent.recording.reports import ReportStore
    from guandan_agent.web import create_web_app
    from guandan_agent.web.review import WebReviews

    config_directory = Path(config_root).resolve()
    allowed = {}
    for name in allowed_config_names:
        if not name or Path(name).name != name or "\\" in name or not name.endswith(".toml"):
            raise ValueError("Only explicit configuration base names are allowed")
        path = config_directory / name
        if path.is_symlink() or path.resolve().parent != config_directory:
            raise ValueError("Configuration must be inside its configured directory")
        allowed[name] = load_config(path)
    review_config = None
    if allow_report_generation and review_config_name is not None:
        name = review_config_name
        if not name or Path(name).name != name or "\\" in name or not name.endswith(".toml"):
            raise ValueError("Only an explicit review configuration base name is allowed")
        path = config_directory / name
        if path.is_symlink() or path.resolve().parent != config_directory:
            raise ValueError("Review configuration must be inside its configured directory")
        if path.is_file():
            review_config = load_config(path)
    manager = create_manager(recording_root=recording_root)
    queries = QueryService(manager)
    reports = ReportStore(queries)
    model_reviews = WebReviews(manager, queries, reports, review_config,
                               enabled=allow_report_generation)
    archive_directory = Path(archive_root if archive_root is not None else recording_root)

    @asynccontextmanager
    async def lifespan(web):
        imported = set()
        web.state.archive_errors = []

        async def register(path: Path):
            try:
                await queries.register_archive(path)
                imported.add(path.name)
                await reports.load_directory(path.name, path / "reports")
            except (QueryError, OSError, ValueError):
                web.state.archive_errors.append("ARCHIVE_UNAVAILABLE")

        # Only UUID run directories below the trusted server-side root. No HTTP
        # path can choose a file; at most eight recent runs are loaded once.
        paths = []
        if archive_directory.is_dir():
            for path in archive_directory.iterdir():
                try:
                    if path.is_symlink() or not path.is_dir() or str(UUID(path.name)) != path.name:
                        continue
                    summary = path / "summary.json"
                    modified = summary.stat().st_mtime_ns if summary.is_file() else path.stat().st_mtime_ns
                    paths.append((modified, path))
                except (ValueError, OSError):
                    continue
        for _, path in sorted(paths, key=lambda item: (item[0], item[1].name))[-8:]:
            await register(path)

        async def collect_finished():
            while True:
                for view in manager.list_views():
                    if view.run_id not in imported and manager.is_closed(view.run_id):
                        # Mark attempted before I/O so damaged logs are not retried forever.
                        imported.add(view.run_id)
                        await register(manager.archive_path(view.run_id))
                await asyncio.sleep(0.1)

        collector = asyncio.create_task(collect_finished())
        try:
            yield
        finally:
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)
            views = manager.list_views()
            for view in views:
                manager.stop(view.run_id)
            if views:
                await asyncio.gather(*(manager.wait(v.run_id) for v in views))
            await manager.close_reviews()
            await model_reviews.close()

    web = create_web_app(manager=manager, queries=queries, allowed_configs=allowed,
                         reports=reports, host=host, port=port, lifespan=lifespan,
                         allow_report_generation=allow_report_generation,
                         model_reviews=model_reviews)
    web.state.manager = manager
    web.state.queries = queries
    web.state.reports = reports
    web.state.model_reviews = model_reviews
    return web


async def review_archives(config: RunConfig, sources: tuple[Path, ...], *, live: bool = False,
                          manager: RunManager | None = None, session_id: str | None = None,
                          request_ids: tuple[str, ...] | None = None) -> dict:
    """Explicit offline command; check mode never constructs a provider or task."""
    from guandan_agent.recording import ClosedRunLog
    from guandan_agent.offline.review import generate_review, LEGACY_PROMPT_VERSION, FACTS_PROMPT_VERSION, PROMPT_VERSION
    from guandan_agent.offline.output import save_report, save_review_execution

    if not sources:
        raise ValueError("At least one source record is required")
    # Explicit archived offline configs retain their frozen prompt. Online
    # configs use the current offline projection, recorded separately below.
    if config.versions.prompt in {LEGACY_PROMPT_VERSION, FACTS_PROMPT_VERSION, PROMPT_VERSION}:
        prompt_version = config.versions.prompt
    elif config.versions.prompt in {"b05-main-v1", "b07-review-v1", "b09-facts-v2", "b09-facts-v3"}:
        prompt_version = PROMPT_VERSION
    else:
        raise ValueError("Unsupported offline prompt version")
    loaded = [await ClosedRunLog.load(path, cache_events=20000) for path in sources]
    if not live:
        return {"mode": "check", "offline_config_id": config.config_id,
                "max_model_calls": config.offline.max_model_calls, "starts_task": False,
                "prompt_version": prompt_version,
                "sources": [{"run_id": source.run_id, "config_id": source.config_id,
                             "closed": source.summary.closed, "log_complete": source.summary.log_complete,
                             "retained_from_seq": source.retained_from_seq} for source in loaded]}
    owner = manager or create_manager()
    clock = SystemClock()
    reports = []
    try:
        for path, source in zip(sources, loaded):
            async def job(calls, source=source):
                return await generate_review(source, config, calls, report_id=calls.task_id,
                                             created_at=clock.utc_now(), session_id=session_id,
                                             request_ids=request_ids, prompt_version=prompt_version)
            task = owner.start_review(config, source, job)
            interrupted = False
            try:
                report = await owner.wait_review(task.task_id)
            except asyncio.CancelledError:
                owner.stop_review(task.task_id, "Offline command was interrupted.")
                report = await owner.wait_review(task.task_id)
                interrupted = True
            output = path / "reports" / f"review-{report.report_id}.json"
            await save_report(output, report)
            view = owner.review_view(task.task_id)
            await save_review_execution(path / "reports" / f"review-{report.report_id}.execution.json",
                                        view=view, config=config, budget=owner.offline_usage())
            reports.append({"task": view.model_dump(mode="json"), "report": report.model_dump(mode="json"),
                            "report_path": str(output.resolve())})
            if interrupted:
                raise asyncio.CancelledError
    finally:
        await owner.close_reviews()
    return {"mode": "live", "offline_budget": owner.offline_usage(), "reports": reports,
            "online_runs_started": 0}


async def evaluate_archives(sources: tuple[Path, ...], *, output: Path | None = None,
                            unit: str = "match") -> dict:
    from uuid import uuid4
    from guandan_agent.recording import ClosedRunLog
    from guandan_agent.offline.evaluate import evaluate
    from guandan_agent.offline.output import save_report
    if not sources:
        raise ValueError("At least one source record is required")
    records = [await ClosedRunLog.load(path, cache_events=20000) for path in sources]
    report = evaluate(records, report_id=str(uuid4()), created_at=SystemClock().utc_now(), unit=unit)
    output = output or sources[0] / "reports" / f"evaluation-{report.report_id}.json"
    await save_report(output, report)
    return {"report_path": str(output.resolve()), "report": report.model_dump(mode="json"),
            "model_calls": 0}
