"""Import and in-process compatibility checks, not a product HTTP service."""

import importlib
import socket
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_locked_dependencies_import_without_credentials():
    assert sys.version_info[:2] == (3, 12)
    for name in (
        "guandan_agent", "pydantic", "websockets", "langgraph.graph",
        "langchain_deepseek", "fastapi", "uvicorn",
    ):
        assert importlib.import_module(name) is not None


def test_fastapi_in_process_compatibility():
    app = FastAPI()

    @app.get("/smoke")
    def smoke():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/smoke")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_ordinary_tests_reject_network():
    with socket.socket() as connection:
        with pytest.raises(AssertionError, match="network connections"):
            connection.connect(("127.0.0.1", 9))
