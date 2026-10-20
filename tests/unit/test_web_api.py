"""The HTTP boundary uses in-memory dependencies and never starts a real game."""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from guandan_agent.config import RunConfig
from guandan_agent.runtime.manager import RunManagerError
from guandan_agent.web.api import create_web_app
from guandan_agent.recording.query import QueryError


ORIGIN = "http://127.0.0.1:8000"
RUN = "ef1adc95-0c5a-4c2c-bde3-37e2c8aac72c"
SESSION = "8c4016f6-555e-41ed-85a6-9d8e5b2c7735"
REQUEST = f"{SESSION}:q1"
ROOT = f"/api/v1/runs/{RUN}"
SEAT = f"{ROOT}/sessions/{SESSION}"


class Manager:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.failure = None
        self.view = {"run_id": RUN, "status": "running", "sessions": []}

    def start(self, config):
        assert asyncio.get_running_loop()
        if self.failure is not None:
            raise self.failure
        self.started.append(config)
        return self.view

    def stop(self, run_id):
        assert asyncio.get_running_loop()
        if self.failure is not None:
            raise self.failure
        self.stopped.append(run_id)
        return {**self.view, "status": "stopping"}


class Queries:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.closed = True

    def _call(self, name, *args):
        assert asyncio.get_running_loop()
        self.calls.append((name, args))
        if self.failure is not None:
            raise self.failure

    def list_runs(self):
        self._call("list_runs")
        return [{"run_id": RUN, "read_only": True, "label": "fixed record"}]

    def view(self, run_id):
        self._call("view", run_id)
        return {"run_id": run_id, "sessions": [{"session_id": SESSION, "seat": 2}]}

    def session_view(self, run_id, session_id):
        self._call("session_view", run_id, session_id)
        return {"session_id": session_id, "state": {"self_hand": ["S2"], "remaining_counts": [27, 27, 1, 27]}}

    def decisions(self, run_id, session_id, *, after_seq, limit):
        self._call("decisions", run_id, session_id, after_seq, limit)
        return {"items": [{"request_id": REQUEST, "opened_seq": 2}], "next_after_seq": 2}

    def decision(self, run_id, session_id, request_id):
        self._call("decision", run_id, session_id, request_id)
        return {"request_id": request_id, "state": {"self_hand": ["S2"]}, "calls": []}

    def closed_record(self, run_id):
        self._call("closed_record", run_id)
        if not self.closed:
            raise QueryError(code="RUN_ACTIVE", status_code=409, safe_message="The record is still active.")
        return object()


class Reports:
    def __init__(self):
        self.calls = []

    def list(self, run_id):
        self.calls.append(("list", run_id))
        return {"items": [], "log_complete": True}

    def get(self, run_id, report_id):
        self.calls.append(("get", run_id, report_id))
        return {"report_id": report_id, "status": "partial", "entries": []}


def harness(**kwargs):
    manager, queries, reports = Manager(), Queries(), Reports()
    config = RunConfig.model_validate({"batch": {"max_model_calls": 0}})
    app = create_web_app(manager=manager, queries=queries,
                         allowed_configs={"local.example.toml": config}, reports=reports, **kwargs)
    return app, manager, queries, reports, config


def request(app, method, path, **kwargs):
    async def execute():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(execute())


def assert_error(response, status, code):
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    error = response.json()["error"]
    assert set(error) == {"code", "component", "safe_message", "related_event_id"}
    assert error["code"] == code
    assert error["component"] == "web"
    assert "sensitive-value" not in response.text


def test_factory_and_read_requests_are_inert():
    app, manager, queries, reports, config = harness()
    assert manager.started == manager.stopped == queries.calls == reports.calls == []
    result = request(app, "GET", "/api/v1/configs")
    assert result.json() == {"items": [{"config_name": "local.example.toml", "policy": "V0", "model_calls": 0}]}
    assert config.platform.url not in result.text
    assert "API_KEY" not in result.text
    for path in ["/api/v1/runs", ROOT, f"{SEAT}/state", f"{SEAT}/decisions", f"{SEAT}/decisions/{REQUEST}", f"{ROOT}/reports", f"{ROOT}/reports/fixture"]:
        assert request(app, "GET", path).status_code == 200
    assert manager.started == manager.stopped == []
    assert ("session_view", (RUN, SESSION)) in queries.calls
    assert ("decision", (RUN, SESSION, REQUEST)) in queries.calls


def test_start_only_uses_validated_allowlisted_config():
    app, manager, _, _, config = harness()
    response = request(app, "POST", "/api/v1/runs", json={"config_name": "local.example.toml"}, headers={"Origin": ORIGIN})
    assert response.status_code == 202
    assert response.json()["run_id"] == RUN
    assert len(manager.started) == 1
    assert manager.started[0] == config
    assert manager.started[0] is not config


def test_stop_uses_authority_and_remains_idempotent():
    app, manager, _, _, _ = harness()
    for _ in range(2):
        response = request(app, "POST", f"{ROOT}/stop", json={}, headers={"Origin": ORIGIN})
        assert response.status_code == 202
        assert response.json()["status"] == "stopping"
    assert manager.stopped == [RUN, RUN]


@pytest.mark.parametrize("body", [
    {}, {"config_name": "unknown.toml"}, {"config_name": "../sensitive-value"},
    {"config_name": "/sensitive-value"}, {"config_name": "..\\sensitive-value"},
    {"config_name": "local.example.toml", "api_key": "sensitive-value"},
    {"config_name": True}, {"config_name": None}, {"config_name": 12},
    {"config_name": ""}, {"config_name": "a" * 129}, ["local.example.toml"], None,
])
def test_start_rejects_unapproved_or_invalid_input(body):
    app, manager, _, _, _ = harness()
    response = request(app, "POST", "/api/v1/runs", json=body, headers={"Origin": ORIGIN})
    assert response.status_code == 422
    assert "sensitive-value" not in response.text
    assert set(response.json()) == {"error"}
    assert manager.started == []


@pytest.mark.parametrize("body", [{"action_id": "L0001"}, [], None, "sensitive-value"])
def test_stop_requires_empty_object(body):
    app, manager, _, _, _ = harness()
    response = request(app, "POST", f"{ROOT}/stop", json=body, headers={"Origin": ORIGIN})
    assert response.status_code == 422
    assert manager.stopped == []


@pytest.mark.parametrize("origin", [None, "null", "http://evil.test", "http://localhost:8000", "https://127.0.0.1:8000", "http://127.0.0.1:9000", "http://127.0.0.1:8000/"])
def test_post_requires_matching_origin(origin):
    app, manager, queries, _, _ = harness()
    headers = {} if origin is None else {"Origin": origin}
    response = request(app, "POST", "/api/v1/runs", json={"config_name": "local.example.toml"}, headers=headers)
    assert_error(response, 403, "ORIGIN_FORBIDDEN")
    assert manager.started == queries.calls == []


@pytest.mark.parametrize("host", ["evil.test:8000", "127.0.0.1:9000", "127.0.0.1", "localhost", "127.0.0.1:8000.evil.test", "127.0.0.1:8000@evil.test", "0.0.0.0:8000"])
def test_host_is_checked_on_reads_too(host):
    app, manager, queries, _, _ = harness()
    response = request(app, "GET", "/api/v1/runs", headers={"Host": host})
    assert_error(response, 403, "HOST_FORBIDDEN")
    assert manager.started == queries.calls == []


def test_localhost_alias_with_its_own_origin_is_accepted():
    app, manager, _, _, _ = harness()
    response = request(app, "POST", "/api/v1/runs", json={"config_name": "local.example.toml"}, headers={"Host": "localhost:8000", "Origin": "http://localhost:8000"})
    assert response.status_code == 202
    assert len(manager.started) == 1


@pytest.mark.parametrize("name,values,code", [
    ("Host", ["127.0.0.1:8000", "evil.test"], "HOST_FORBIDDEN"),
    ("Origin", [ORIGIN, "http://evil.test"], "ORIGIN_FORBIDDEN"),
    ("Content-Type", ["application/json", "text/plain"], "JSON_REQUIRED"),
])
def test_duplicate_control_headers_are_rejected(name, values, code):
    app, manager, _, _, _ = harness()
    headers = [("Origin", ORIGIN), ("Content-Type", "application/json")]
    headers = [(key, value) for key, value in headers if key != name]
    headers.extend((name, value) for value in values)
    response = request(app, "POST", "/api/v1/runs", content='{"config_name":"local.example.toml"}', headers=headers)
    assert_error(response, 422 if name == "Content-Type" else 403, code)
    assert manager.started == []


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "application/problem+json"])
def test_controls_require_application_json(content_type):
    app, manager, _, _, _ = harness()
    headers = {"Origin": ORIGIN}
    if content_type:
        headers["Content-Type"] = content_type
    response = request(app, "POST", "/api/v1/runs", content='{"config_name":"local.example.toml"}', headers=headers)
    assert_error(response, 422, "JSON_REQUIRED")
    assert manager.started == []


def test_malformed_json_does_not_echo_body():
    app, manager, _, _, _ = harness()
    response = request(app, "POST", "/api/v1/runs", content='{"sensitive-value"', headers={"Origin": ORIGIN, "Content-Type": "application/json; charset=utf-8"})
    assert_error(response, 422, "INVALID_REQUEST")
    assert manager.started == []


@pytest.mark.parametrize("query", ["after_seq=-1", "after_seq=true", "after_seq=private", "limit=0", "limit=201", "limit=sensitive-value"])
def test_cursor_parameters_are_bounded(query):
    app, _, queries, _, _ = harness()
    response = request(app, "GET", f"{SEAT}/decisions?{query}")
    assert_error(response, 422, "INVALID_REQUEST")
    assert queries.calls == []


def test_cursor_is_forwarded_without_switching_session():
    app, _, queries, _, _ = harness()
    response = request(app, "GET", f"{SEAT}/decisions?after_seq=7&limit=200")
    assert response.status_code == 200
    assert queries.calls == [("decisions", (RUN, SESSION, 7, 200))]


@pytest.mark.parametrize("status,code", [(404, "SESSION_NOT_FOUND"), (409, "RUN_ACTIVE"), (410, "HISTORY_UNAVAILABLE"), (422, "QUERY_INVALID"), (503, "QUERY_UNAVAILABLE")])
def test_query_errors_preserve_safe_contract(status, code):
    app, _, queries, _, _ = harness()
    queries.failure = QueryError(code=code, status_code=status, safe_message="A safe fixed message.")
    response = request(app, "GET", f"{SEAT}/state")
    assert_error(response, status, code)


@pytest.mark.parametrize("status,code", [(404, "RUN_NOT_FOUND"), (409, "RUN_CONFLICT"), (422, "INVALID_CONFIG"), (503, "POLICY_UNAVAILABLE")])
def test_manager_errors_preserve_safe_contract(status, code):
    app, manager, _, _, _ = harness()
    manager.failure = RunManagerError(code, "A safe fixed message.")
    response = request(app, "POST", "/api/v1/runs", json={"config_name": "local.example.toml"}, headers={"Origin": ORIGIN})
    assert_error(response, status, code)


def test_unexpected_errors_are_safe_and_not_reraised_to_server():
    app, _, queries, _, _ = harness()
    queries.failure = RuntimeError("sensitive-value in exception repr")
    response = request(app, "GET", ROOT)
    assert_error(response, 503, "APP_UNAVAILABLE")


def test_evicted_history_exposes_only_a_known_earliest_cursor():
    app, _, queries, _, _ = harness()
    queries.failure = QueryError(code="HISTORY_UNAVAILABLE", status_code=410,
                                 safe_message="History was evicted.", retained_from_seq=37)
    response = request(app, "GET", f"{SEAT}/decisions")
    assert response.status_code == 410
    assert response.json()["retained_from_seq"] == 37
    assert set(response.json()) == {"error", "retained_from_seq"}


@pytest.mark.parametrize("path", [f"{ROOT}/reports", f"{ROOT}/reports/fixture"])
def test_reports_do_not_read_open_runs(path):
    app, _, queries, reports, _ = harness()
    queries.closed = False
    response = request(app, "GET", path)
    assert_error(response, 409, "RUN_ACTIVE")
    assert reports.calls == []


@pytest.mark.parametrize("path", ["/.env", "/runs", "/configs/local.example.toml", "/api/v1/actions", "/api/v1/retry", "/api/v1/generate-report", "/static/.env", "/static/%2e%2e/%2e%2e/.env", "/openapi.json", "/docs"])
def test_only_package_static_files_and_declared_routes_are_exposed(path):
    app, manager, _, reports, _ = harness()
    response = request(app, "GET", path)
    assert_error(response, 404, "NOT_FOUND")
    assert manager.started == reports.calls == []


def test_static_index_has_same_security_headers_and_no_wildcard_cors():
    app, manager, _, _, _ = harness()
    response = request(app, "GET", "/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "access-control-allow-origin" not in response.headers
    assert manager.started == []


def test_caller_owns_lifespan_without_implicit_run_start():
    flags = []

    @asynccontextmanager
    async def lifespan(app):
        flags.append("start")
        yield
        flags.append("stop")

    app, manager, _, _, _ = harness(lifespan=lifespan)

    async def check():
        async with app.router.lifespan_context(app):
            assert flags == ["start"]
    asyncio.run(check())
    assert flags == ["start", "stop"]
    assert manager.started == []


@pytest.mark.parametrize("kwargs", [{"host": "0.0.0.0"}, {"host": "evil.test"}, {"port": 0}, {"port": 65536}, {"port": True}, {"port": "8000"}])
def test_factory_rejects_nonlocal_bindings(kwargs):
    with pytest.raises(ValueError):
        harness(**kwargs)
