"""One receiver and one gated send channel per OpenGuanDan connection."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
import json
from typing import Any, Protocol

from guandan_agent.clock import Clock, SystemClock
from guandan_agent.contracts import (
    ErrorInfo, GateDecision, PlatformEvent, SeatConnection, SendResult, SubmissionIntent,
)
from guandan_agent.platform.parser import parse_platform_frame


class SendGate(Protocol):
    def begin_write(self, intent: SubmissionIntent) -> GateDecision: ...


class PlatformPort(Protocol):
    async def connect(self, config: SeatConnection) -> None: ...
    def events(self) -> AsyncIterator[PlatformEvent]: ...
    async def write(self, intent: SubmissionIntent, gate: SendGate) -> SendResult: ...
    async def close(self) -> None: ...


class PlatformConnectionError(RuntimeError):
    """Static transport failure; the Session decides how to mark interruption."""


class _Socket(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...
    async def send(self, data: str) -> None: ...
    async def close(self) -> None: ...


async def _connect_websocket(url: str) -> _Socket:
    from websockets.asyncio.client import connect

    return await connect(url, proxy=None, ping_interval=None, open_timeout=10,
                         close_timeout=2, max_size=16 * 1024 * 1024)


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="platform", safe_message=message, related_event_id=None)


def admission_command(config: SeatConnection) -> dict[str, Any]:
    data: dict[str, Any] = {"userId": config.user_name, "seatNum": config.seat}
    if config.mode == "create":
        data["round"] = config.server_rounds
        kind = "CREATE_ROOM"
    else:
        data["roomId"] = config.room_id
        kind = "JOIN_ROOM"
    return {"type": kind, "data": data}


def encode_submission(intent: SubmissionIntent, *, room_id: int, seat: int) -> dict[str, Any]:
    """Encode an already-authorized frozen intent; no candidate reconstruction."""
    if type(room_id) is not int or room_id < 0 or type(seat) is not int or seat not in range(4):
        raise ValueError("invalid confirmed room or seat")
    if intent.binding.seat != seat:
        raise ValueError("intent belongs to a different seat")
    data: dict[str, Any] = {"roomId": room_id, "player": seat,
                            "act": intent.model_dump(mode="json")["wire_action"]}
    if intent.stage == "back":
        assert intent.platform_context is not None  # SubmissionIntent validates this.
        data.update(tributePos=intent.platform_context.tribute_pos,
                    tribute=intent.platform_context.tribute_card)
    return {"type": {"play": "PLAY", "tribute": "TRIBUTE", "back": "PAYTRIBUTE"}[intent.stage],
            "data": data}


class OpenGuanDanPlatform:
    """No reconnect, resend, state mutation, background consumer, or model call.

    ``connect`` sends admission once. The caller consumes ``events`` and learns
    the assigned room from ``joined``; socket-open alone never grants writes.
    After each yielded event, the caller must synchronously apply it before
    requesting the next one (the Session contract).
    """

    def __init__(self, *, clock: Clock | None = None,
                 connector: Callable[[str], Awaitable[_Socket]] | None = None) -> None:
        self._clock = clock if clock is not None else SystemClock()
        self._connector = connector if connector is not None else _connect_websocket
        self._config: SeatConnection | None = None
        self._socket: _Socket | None = None
        self._room_id: int | None = None
        self._joined = False
        self._closed = False
        self._started = False
        self._receiver_claimed = False
        self._write_uncertain = False
        self._rx_seq = 0
        self._send_lock = asyncio.Lock()

    @property
    def room_id(self) -> int | None:
        return self._room_id

    async def connect(self, config: SeatConnection) -> None:
        if self._started or self._closed:
            raise PlatformConnectionError("Platform connection cannot be reused")
        self._started = True
        self._config = config
        try:
            socket = await self._connector(config.url)
            self._socket = socket
            if self._closed:
                await socket.close()
                raise PlatformConnectionError("Platform closed during connection")
            await socket.send(json.dumps(admission_command(config), ensure_ascii=False))
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception:
            await self.close()
            raise PlatformConnectionError("Platform admission transport failed") from None

    async def events(self) -> AsyncIterator[PlatformEvent]:
        if self._receiver_claimed:
            raise PlatformConnectionError("Platform receive stream has one consumer")
        if self._socket is None or self._config is None or self._closed:
            raise PlatformConnectionError("Platform is not connected")
        self._receiver_claimed = True
        match_closed = False
        try:
            async for raw in self._socket:
                if self._closed:
                    break
                self._rx_seq += 1
                event = parse_platform_frame(
                    raw, config=self._config, room_id=self._room_id, rx_seq=self._rx_seq,
                    received_at=self._clock.utc_now(), received_mono=self._clock.monotonic(),
                )
                if event.kind == "joined":
                    if self._joined:
                        event = self._invalid_order(event)
                    else:
                        self._joined = True
                        self._room_id = event.payload["data"]["roomId"]
                elif event.kind == "join_failed":
                    self._joined = False
                elif event.kind not in ("unknown", "auxiliary_notice") and not self._joined:
                    event = self._invalid_order(event)
                if event.kind == "match_closed":
                    match_closed = True
                    self._joined = False
                yield event
                if match_closed:
                    return
            if not self._closed and not match_closed:
                raise PlatformConnectionError("Platform receive stream closed before match completion")
        except asyncio.CancelledError:
            raise
        except PlatformConnectionError:
            raise
        except Exception:
            raise PlatformConnectionError("Platform receive transport failed") from None
        finally:
            self._joined = False
            # A receive stream is never resumed with its old authority.

    @staticmethod
    def _invalid_order(event: PlatformEvent) -> PlatformEvent:
        return PlatformEvent(**{**event.model_dump(mode="json"), "kind": "unknown", "payload": {
            "raw": event.payload,
            "error": _error("INVALID_PLATFORM_ORDER", "Platform admission order is invalid").model_dump(mode="json"),
        }})

    def _preflight(self, intent: SubmissionIntent) -> ErrorInfo | None:
        config = self._config
        if self._closed or not self._joined or self._socket is None or self._write_uncertain:
            return _error("PLATFORM_NOT_READY", "Platform connection is not ready to submit")
        assert config is not None
        binding = intent.binding
        if (binding.session_id, binding.seat, binding.connection_epoch) != (
                config.session_id, config.seat, config.connection_epoch):
            return _error("CONNECTION_BINDING_MISMATCH", "Intent belongs to a different connection")
        if self._clock.monotonic() >= intent.final_deadline_mono:
            return _error("FINAL_DEADLINE", "Final submission deadline has elapsed")
        return None

    async def write(self, intent: SubmissionIntent, gate: SendGate) -> SendResult:
        def result(status: str, error: ErrorInfo | None) -> SendResult:
            return SendResult(submission_id=intent.submission_id, status=status, error=error)

        error = self._preflight(intent)
        if error is not None:
            return result("not_started", error)
        started = False
        try:
            # Limit waiting for this send channel and the actual write by the
            # same local deadline. Always recheck the injected clock after wait.
            async with asyncio.timeout(max(0.0, intent.final_deadline_mono - self._clock.monotonic())):
                async with self._send_lock:
                    error = self._preflight(intent)
                    if error is not None:
                        return result("not_started", error)
                    assert self._room_id is not None and self._config is not None and self._socket is not None
                    encoded = json.dumps(encode_submission(intent, room_id=self._room_id,
                                                            seat=self._config.seat), ensure_ascii=False)
                    decision = gate.begin_write(intent)
                    if decision.status != "allow":
                        return result("not_started", decision.error or _error("SEND_DENIED", "Submission gate denied write"))
                    started = True
                    await self._socket.send(encoded)
                    return result("sent", None)
        except asyncio.CancelledError:
            # Once writing starts, cancellation cannot prove zero bytes left.
            error = _error("SEND_CANCELLED", "Submission cancelled; write may have started" if started else "Submission cancelled before write")
        except TimeoutError:
            error = _error("SEND_TIMEOUT", "Submission deadline elapsed")
        except Exception:
            error = _error("SEND_FAILED", "Submission transport or gate failed")
        if started:
            self._write_uncertain = True
        return result("unknown" if started else "not_started", error)

    async def close(self) -> None:
        self._closed = True
        self._joined = False
        if self._socket is not None:
            try:
                await self._socket.close()
            except Exception:
                # Closing never changes an unknown submission to a confirmed one.
                pass


_END = object()


class _ScriptedSocket:
    def __init__(self, frames: Iterable[str | bytes | dict[str, Any]]) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[str] = []
        self.closed = False
        for frame in frames:
            self.feed(frame)

    def feed(self, frame: str | bytes | dict[str, Any]) -> None:
        if self.closed:
            raise PlatformConnectionError("Scripted connection is closed")
        # Copy through JSON at the same boundary as a real socket.
        self.incoming.put_nowait(json.dumps(frame, ensure_ascii=False) if isinstance(frame, dict) else frame)

    def disconnect(self) -> None:
        self.closed = True
        self.incoming.put_nowait(_END)

    def __aiter__(self) -> _ScriptedSocket:
        return self

    async def __anext__(self) -> str | bytes:
        item = await self.incoming.get()
        if item is _END:
            raise StopAsyncIteration
        return item

    async def send(self, data: str) -> None:
        if self.closed:
            raise PlatformConnectionError("Scripted connection is closed")
        self.sent.append(data)

    async def close(self) -> None:
        if not self.closed:
            self.disconnect()


class ScriptedPlatform(OpenGuanDanPlatform):
    """A network-free transport using the exact production parser/send path.

    Feed only this seat's frames. An empty queue waits for ``feed``; ``disconnect``
    simulates EOF, while ``close`` is the caller's intentional shutdown.
    """

    def __init__(self, frames: Iterable[str | bytes | dict[str, Any]] = (), *, clock: Clock | None = None) -> None:
        self._script = _ScriptedSocket(frames)

        async def connector(_: str) -> _Socket:
            return self._script

        super().__init__(clock=clock, connector=connector)

    @property
    def sent_frames(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(raw) for raw in self._script.sent)

    def feed(self, frame: str | bytes | dict[str, Any]) -> None:
        self._script.feed(frame)

    def disconnect(self) -> None:
        self._script.disconnect()
