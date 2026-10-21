"""Ordinary tests are offline even when local provider credentials exist."""

import socket
from contextvars import ContextVar

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_socketpair = socket.socketpair
    creating_socketpair = ContextVar("creating_socketpair", default=False)

    def local_control_socket(sock, address):
        # CPython implements Windows socketpair via a temporary loopback TCP
        # connection. Allow only this synchronous internal operation, not
        # arbitrary loopback services or connections from another thread.
        return (sock.family == getattr(socket, "AF_UNIX", None)
                or (creating_socketpair.get()
                    and sock.family in (socket.AF_INET, socket.AF_INET6)
                    and address[0] in ("127.0.0.1", "::1")))

    def control_socketpair(*args, **kwargs):
        token = creating_socketpair.set(True)
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            creating_socketpair.reset(token)

    def reject_network(sock, address):
        if local_control_socket(sock, address):
            return original_connect(sock, address)
        raise AssertionError("Ordinary tests must not open network connections")

    def reject_network_ex(sock, address):
        if local_control_socket(sock, address):
            return original_connect_ex(sock, address)
        raise AssertionError("Ordinary tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network_ex)
    monkeypatch.setattr(socket, "socketpair", control_socketpair)
