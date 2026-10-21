"""Windows event-loop control sockets must not open the network boundary."""
import socket

import pytest


def test_socketpair_can_exchange_control_bytes():
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(1)
        writer.sendall(b"control")
        assert reader.recv(7) == b"control"


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
@pytest.mark.parametrize("host", ["127.0.0.1", "192.0.2.1"])
def test_connections_remain_blocked_after_socketpair(method, host):
    reader, writer = socket.socketpair()
    reader.close()
    writer.close()
    with socket.socket() as connection:
        with pytest.raises(AssertionError, match="must not open network"):
            getattr(connection, method)((host, 18181))
