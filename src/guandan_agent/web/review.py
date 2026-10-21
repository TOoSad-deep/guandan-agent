"""Publish explicitly requested, bounded model reviews through one runtime owner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from guandan_agent.clock import SystemClock
from guandan_agent.config import RunConfig
from guandan_agent.contracts import ErrorInfo
from guandan_agent.offline.output import save_report, save_review_execution
from guandan_agent.offline.review import PROMPT_VERSION, generate_review
from guandan_agent.recording.query import QueryError, QueryService
from guandan_agent.recording.reports import ReportStore
from guandan_agent.runtime.manager import RunManager, RunManagerError


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="web_review", safe_message=message, related_event_id=None)


def _display_error(error: ErrorInfo | None) -> ErrorInfo | None:
    if error is None:
        return None
    messages = {
        "OFFLINE_PROVIDER_UNAVAILABLE": "模型不可用；请确认服务已加载密钥或检查模型配置。",
        "MODEL_CREDENTIALS_MISSING": "启动服务的环境未配置 DEEPSEEK_API_KEY，请配置后重启服务。",
        "POLICY_UNAVAILABLE": "本服务没有可用的模型提供方。",
        "OFFLINE_BUDGET": "本次服务的模型复盘额度已用完。",
        "OFFLINE_ONLINE_ACTIVE": "在线对局进行中，请结束后再生成模型复盘。",
        "OFFLINE_CONFLICT": "已有模型复盘正在生成，请等待完成。",
        "OFFLINE_CANCELLED": "复盘任务已停止，已保留能够保存的结果。",
        "MODEL_TIMEOUT": "模型调用超时，未自动重试。",
        "OFFLINE_TIMEOUT": "复盘生成超时，未自动重试。",
    }
    message = messages.get(error.code)
    return error if message is None else error.model_copy(update={"safe_message": message})


def disabled_capabilities() -> dict[str, Any]:
    return {
        "enabled": False, "model": None, "calls_per_report": 1, "max_decisions": 3,
        "max_calls": 0, "remaining_calls": 0, "active_task": None,
        "disabled_reason": "本服务尚未配置网页模型复盘。",
    }


@dataclass
class _Review:
    task_id: str
    source_run_id: str
    status: str = "running"
    report_id: str | None = None
    error: ErrorInfo | None = None
    publisher: asyncio.Task | None = None


class WebReviews:
    """Keep publication separate from runtime completion; GET never starts work."""

    def __init__(self, manager: RunManager, queries: QueryService, reports: ReportStore,
                 config: RunConfig | None, *, enabled: bool) -> None:
        self.manager, self.queries, self.reports = manager, queries, reports
        self.config = None if config is None else RunConfig.model_validate_json(config.model_dump_json())
        if self.config is not None and self.config.versions.prompt != PROMPT_VERSION:
            raise ValueError("Web model reviews require the current b09-offline-v3 prompt.")
        self.enabled = enabled
        self._reviews: dict[str, _Review] = {}
        self._active: str | None = None
        self._starting = False
        self._closing = False

    def capabilities(self) -> dict[str, Any]:
        result = disabled_capabilities()
        if self.config is not None:
            limit = self.config.offline.max_model_calls
            result.update(model=self.config.model.name, max_calls=limit,
                          remaining_calls=max(0, limit - self.manager.offline_usage()["slots_used"]))
        if not self.enabled:
            result["disabled_reason"] = "当前页面使用只读档案，未启用模型复盘。"
        elif self._closing:
            result["disabled_reason"] = "服务正在关闭，不能开始新的复盘。"
        elif self.config is not None and self.config.offline.max_model_calls > 0:
            result.update(enabled=True, disabled_reason=None)
        elif self.config is not None:
            result["disabled_reason"] = "服务端模型复盘额度为零。"
        if self._active is not None:
            review = self._reviews[self._active]
            result["active_task"] = self.state(review.source_run_id, review.task_id)
        return result

    def _directory(self, run_id: str) -> Path:
        archive = self.queries.archive_path(run_id)
        directory = archive / "reports"
        if not archive.is_dir() or archive.resolve() != archive or directory.is_symlink():
            raise QueryError("REPORT_DIRECTORY_UNAVAILABLE", 422, "已注册档案的报告目录不可用。")
        return directory

    async def start(self, run_id: str, session_id: str) -> dict[str, Any]:
        capability = self.capabilities()
        if not capability["enabled"]:
            raise QueryError("MODEL_REVIEW_DISABLED", 403, capability["disabled_reason"])
        if self._starting or self._active is not None:
            raise QueryError("REPORT_BUSY", 409, "已有模型复盘正在生成或保存，请等待完成。")
        if not capability["remaining_calls"]:
            raise QueryError("OFFLINE_BUDGET", 409, "本次服务的模型复盘额度已用完。")
        source = self.queries.closed_record(run_id)
        self.queries.session_view(run_id, session_id)
        directory = self._directory(run_id)
        self._starting = True
        try:
            page = await self.reports.refresh(run_id)
            if len(page.items) >= self.reports.max_reports:
                raise QueryError("REPORT_LIMIT", 409, "当前运行的报告数量已达上限。")

            def prepare_directory() -> None:
                directory.mkdir(exist_ok=True)
                if not os.access(directory, os.W_OK):
                    raise OSError("Report directory is not writable")

            try:
                await asyncio.to_thread(prepare_directory)
            except OSError:
                raise QueryError("REPORT_DIRECTORY_UNAVAILABLE", 422, "报告目录不可写，未调用模型。") from None
            if self._closing:
                raise QueryError("MODEL_REVIEW_DISABLED", 403, "服务正在关闭，不能开始新的复盘。")
            # Recheck after the asynchronous directory refresh, before runtime
            # admission. The manager remains the shared budget authority.
            if not self.capabilities()["remaining_calls"]:
                raise QueryError("OFFLINE_BUDGET", 409, "本次服务的模型复盘额度已用完。")
            assert self.config is not None

            async def job(calls):
                return await generate_review(source, self.config, calls, report_id=calls.task_id,
                                             created_at=SystemClock().utc_now(), session_id=session_id,
                                             prompt_version=PROMPT_VERSION)

            try:
                view = self.manager.start_review(self.config, source, job)
            except RunManagerError as exc:
                error = _display_error(exc.error)
                status = (410 if error.code == "OFFLINE_SOURCE_TRUNCATED" else
                          409 if error.code in {"OFFLINE_ONLINE_ACTIVE", "OFFLINE_CONFLICT",
                                               "OFFLINE_CONFIG_CONFLICT", "OFFLINE_SOURCE_OPEN"} else 422)
                raise QueryError(error.code, status, error.safe_message,
                                 retained_from_seq=source.retained_from_seq if status == 410 else None) from None
            review = _Review(view.task_id, run_id)
            self._reviews[review.task_id] = review
            self._active = review.task_id
            review.publisher = asyncio.create_task(self._publish(review), name=f"web-review-{review.task_id}")
            return self.state(run_id, review.task_id)
        finally:
            self._starting = False

    def state(self, run_id: str, task_id: str) -> dict[str, Any]:
        review = self._reviews.get(task_id)
        if review is None or review.source_run_id != run_id:
            raise QueryError("REVIEW_TASK_NOT_FOUND", 404, "该运行没有此复盘任务。")
        view = self.manager.review_view(task_id)
        error = _display_error(review.error or view.error)
        return {
            "task_id": task_id, "source_run_id": run_id, "status": review.status,
            "report_ready": review.report_id is not None, "report_id": review.report_id,
            "calls_started": view.calls_started, "slots_used": view.slots_used,
            "error": None if error is None else error.model_dump(mode="json"),
        }

    async def _publish(self, review: _Review) -> None:
        try:
            report = await self.manager.wait_review(review.task_id)
            directory = self._directory(review.source_run_id)
            view = self.manager.review_view(review.task_id)
            # Publish the invisible execution sidecar first. A refresh must not
            # discover a report whose execution record failed to save.
            await save_review_execution(directory / f"review-{report.report_id}.execution.json",
                                        view=view, config=self.config, budget=self.manager.offline_usage())
            await save_report(directory / f"review-{report.report_id}.json", report)
            self.reports.register(review.source_run_id, report)
            published = self.reports.get(review.source_run_id, report.report_id)
            review.status = published.status
            review.error = published.errors[0] if published.errors else None
            review.report_id = report.report_id
        except asyncio.CancelledError:
            review.status = "failed"
            review.error = _error("REPORT_WRITE_FAILED", "报告保存已中断，请刷新目录检查已保存的结果。")
        except Exception:
            review.status = "failed"
            review.error = _error("REPORT_WRITE_FAILED", "复盘结果未能完整保存并注册，请检查报告目录后重试。")
        finally:
            if self._active == review.task_id:
                self._active = None

    async def close(self) -> None:
        """After manager.close_reviews(), drain publishers without closing online work."""
        self._closing = True
        pending = [review.publisher for review in self._reviews.values()
                   if review.publisher is not None and not review.publisher.done()]
        if pending:
            await asyncio.gather(*pending)
