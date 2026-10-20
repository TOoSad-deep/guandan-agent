"""Small same-origin HTTP boundary over the single-loop runtime authority."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException

from guandan_agent.config import RunConfig
from guandan_agent.contracts import ErrorInfo
from guandan_agent.runtime.manager import RunManagerError
from guandan_agent.recording.query import QueryError


_STATIC = Path(__file__).with_name("static")
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
}


def _error(status: int, code: str, message: str, *, retained_from_seq: int | None = None) -> JSONResponse:
    error = ErrorInfo(code=code, component="web", safe_message=message,
                      related_event_id=None)
    body = {"error": error.model_dump(mode="json")}
    if status == 410 and type(retained_from_seq) is int and retained_from_seq > 0:
        body["retained_from_seq"] = retained_from_seq
    return JSONResponse(body, status_code=status,
                        headers=_HEADERS)


def _manager_status(code: str) -> int:
    if code in {"RUN_NOT_FOUND", "SESSION_NOT_FOUND"}:
        return 404
    if code in {"RUN_CONFLICT", "RUN_LOOP_CONFLICT"}:
        return 409
    if code in {"INVALID_CONFIG", "VERSION_UNSUPPORTED"}:
        return 422
    return 503


class _StartBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    config_name: Annotated[str, Field(min_length=1, max_length=128)]

    @field_validator("config_name")
    @classmethod
    def base_name(cls, value: str) -> str:
        if value in {".", ".."} or any(c in value for c in "/\\\x00"):
            raise ValueError("A configured file base name is required.")
        return value


class _StopBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _SafeRoute(APIRoute):
    def get_route_handler(self) -> Any:
        handler = super().get_route_handler()

        async def safe_handler(request: Request) -> Any:
            try:
                return await handler(request)
            except (QueryError, RunManagerError, RequestValidationError, HTTPException):
                raise
            except Exception:
                # Keep unexpected exception reprs out of HTTP and server traces.
                return _error(503, "APP_UNAVAILABLE", "The local application is unavailable.")

        return safe_handler


class _LocalBoundary:
    """Reject foreign hosts/origins before parsing controls or touching runtime."""

    def __init__(self, app: Any, *, port: int) -> None:
        self.app = app
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            self.hosts.update({"127.0.0.1", "localhost"})

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", ())
        hosts = [v.decode("latin-1") for k, v in headers if k.lower() == b"host"]
        if len(hosts) != 1 or hosts[0] not in self.hosts:
            await _error(403, "HOST_FORBIDDEN", "The request host is not allowed.")(scope, receive, send)
            return
        if scope["method"] == "POST":
            origins = [v.decode("latin-1") for k, v in headers if k.lower() == b"origin"]
            if origins != [f"{scope.get('scheme', 'http')}://{hosts[0]}"]:
                await _error(403, "ORIGIN_FORBIDDEN", "A matching local Origin is required.")(scope, receive, send)
                return
            types = [v.decode("latin-1").split(";", 1)[0].strip().lower()
                     for k, v in headers if k.lower() == b"content-type"]
            if types != ["application/json"]:
                await _error(422, "JSON_REQUIRED", "Controls require an application/json body.")(scope, receive, send)
                return

        async def secure_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                response_headers = [(k, v) for k, v in message.get("headers", ())
                                    if k.decode("latin-1").lower() not in
                                    {name.lower() for name in _HEADERS}]
                response_headers.extend((k.lower().encode(), v.encode()) for k, v in _HEADERS.items())
                message["headers"] = response_headers
            await send(message)

        await self.app(scope, receive, secure_send)


def create_web_app(*, manager: Any, queries: Any,
                   allowed_configs: Mapping[str, RunConfig], reports: Any,
                   host: str = "127.0.0.1", port: int = 8000,
                   lifespan: Any = None) -> FastAPI:
    """Build inert routes. The composition root owns startup/shutdown resources.

    Configs and archives are explicit dependencies. No path, provider setting,
    credentials, or action supplied through HTTP can become runtime input.
    Every runtime access remains synchronous inside the ASGI event loop.
    """
    if host not in {"127.0.0.1", "localhost"} or isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Web serving requires a local host and valid port.")
    configs = MappingProxyType({
        _StartBody(config_name=name).config_name: RunConfig.model_validate_json(config.model_dump_json())
        for name, config in allowed_configs.items()
    })
    app = FastAPI(title="Guandan Agent local viewer", docs_url=None,
                  redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.router.route_class = _SafeRoute
    app.add_middleware(_LocalBoundary, port=port)

    @app.exception_handler(QueryError)
    async def query_error(_: Request, exc: QueryError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.safe_message,
                      retained_from_seq=getattr(exc, "retained_from_seq", None))

    @app.exception_handler(RunManagerError)
    async def manager_error(_: Request, exc: RunManagerError) -> JSONResponse:
        return _error(_manager_status(exc.error.code), exc.error.code, exc.error.safe_message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        # Validation objects may contain the submitted body and its credentials.
        return _error(422, "INVALID_REQUEST", "Request parameters or JSON body are invalid.")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        return _error(exc.status_code, code, "The requested resource or method is unavailable.")

    @app.exception_handler(Exception)
    async def internal_error(_: Request, __: Exception) -> JSONResponse:
        return _error(503, "APP_UNAVAILABLE", "The local application is unavailable.")

    @app.get("/api/v1/configs")
    async def list_configs() -> dict:
        return {"items": [{"config_name": name,
                           "policy": "/".join(sorted({seat.policy for seat in config.run.seats})),
                           "model_calls": config.batch.max_model_calls}
                          for name, config in configs.items()]}

    @app.get("/api/v1/runs")
    async def list_runs() -> dict:
        return {"items": queries.list_runs()}

    @app.post("/api/v1/runs", status_code=202)
    async def start(body: _StartBody) -> Any:
        config = configs.get(body.config_name)
        if config is None:
            return _error(422, "CONFIG_NOT_ALLOWED", "The configuration name is not allowed.")
        return manager.start(config)

    @app.post("/api/v1/runs/{run_id}/stop", status_code=202)
    async def stop(run_id: str, body: _StopBody) -> Any:
        return manager.stop(run_id)

    @app.get("/api/v1/runs/{run_id}")
    async def run_view(run_id: str) -> Any:
        return queries.view(run_id)

    @app.get("/api/v1/runs/{run_id}/sessions/{session_id}/state")
    async def state(run_id: str, session_id: str) -> Any:
        return queries.session_view(run_id, session_id)

    @app.get("/api/v1/runs/{run_id}/sessions/{session_id}/decisions")
    async def decisions(run_id: str, session_id: str,
                        after_seq: Annotated[int, Query(ge=0)] = 0,
                        limit: Annotated[int, Query(ge=1, le=200)] = 50) -> Any:
        return queries.decisions(run_id, session_id, after_seq=after_seq, limit=limit)

    @app.get("/api/v1/runs/{run_id}/sessions/{session_id}/decisions/{request_id}")
    async def decision(run_id: str, session_id: str, request_id: str) -> Any:
        return queries.decision(run_id, session_id, request_id)

    @app.get("/api/v1/runs/{run_id}/reports")
    async def report_list(run_id: str) -> Any:
        queries.closed_record(run_id)
        return reports.list(run_id)

    @app.get("/api/v1/runs/{run_id}/reports/{report_id}")
    async def report(run_id: str, report_id: str) -> Any:
        queries.closed_record(run_id)
        return reports.get(run_id, report_id)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        if not (_STATIC / "index.html").is_file():
            raise HTTPException(404)
        return FileResponse(_STATIC / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=_STATIC, check_dir=False), name="static")
    return app
