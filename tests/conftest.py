"""Ordinary tests are offline even when local provider credentials exist."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def reject_network(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        raise AssertionError("Ordinary tests must not open network connections")

    def reject_network_ex(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect_ex(sock, address)
        raise AssertionError("Ordinary tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network_ex)
