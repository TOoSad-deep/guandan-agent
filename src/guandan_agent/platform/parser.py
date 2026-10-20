"""Strict parsing of the pinned OpenGuanDan wire protocol; no player state."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from guandan_agent.contracts import (
    Card, ErrorInfo, LegalAction, ParsedActions, PlatformEvent, SeatConnection,
    Stage, TributeContext, WireAction,
)

_CARD = TypeAdapter(Card, config={"strict": True})
_ACTION = TypeAdapter(WireAction, config={"strict": True})
_RANKS = frozenset("23456789TJQKA")
_PASS = ("PASS", "PASS", "PASS")


def _object(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("expected object")
    return value


def _array(value: Any, length: int | None = None) -> list[Any]:
    if type(value) is not list or (length is not None and len(value) != length):
        raise ValueError("expected array")
    return value


def _integer(value: Any, low: int = 0, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError("invalid integer")
    return value


def _string(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("expected string")
    return value


def _rank(value: Any) -> str:
    if _string(value) not in _RANKS:
        raise ValueError("invalid rank")
    return value


def _cards(value: Any) -> list[str]:
    return [_CARD.validate_python(card) for card in _array(value)]


def _action(value: Any) -> tuple[Any, ...]:
    # Only JSON arrays are accepted at the wire boundary.
    _array(value, 3)
    if value != ["PASS", "PASS", "PASS"]:
        _array(value[2])
    return _ACTION.validate_python(value)


def parse_actions(raw: Any, stage: Stage) -> ParsedActions:
    """Deduplicate exact JSON action values, retaining every original index."""
    if stage not in ("play", "tribute", "back"):
        raise ValueError("unknown action stage")
    rows = _array(raw)
    if not rows:
        raise ValueError("empty legal action set")
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, raw_action in enumerate(rows):
        action = _action(raw_action)
        if stage == "play":
            if action[0] in ("tribute", "back"):
                raise ValueError("wrong action stage")
        elif action[0] != stage or action[1] != stage or len(action[2]) != 1:
            raise ValueError("wrong action stage")
        groups.setdefault(action, []).append(index)
    return ParsedActions(
        raw_count=len(rows),
        actions=[LegalAction(action_id=f"L{indices[0]:04d}", source_indices=indices,
                             wire_action=action) for action, indices in groups.items()],
    )


def _leader_fields(message: dict[str, Any]) -> None:
    position = _integer(message["greaterPos"], -1, 3)
    action = message["greaterAction"]
    # Preserve server sentinels for the Session's paired normalization.
    if action is None or action == [] or action == ["PASS", "PASS", "PASS"]:
        if position != -1:
            raise ValueError("inconsistent leader sentinel")
    else:
        if position == -1 or _action(action)[0] in ("tribute", "back"):
            raise ValueError("invalid leading action")


def _context_fields(message: dict[str, Any]) -> None:
    position = _integer(message["curPos"], -1, 3)
    action = message["curAction"]
    if action is None:
        if position != -1:
            raise ValueError("inconsistent current action")
    elif position == -1 or _action(action)[0] in ("tribute", "back"):
        raise ValueError("invalid current action")


def _public_info(value: Any) -> None:
    for row in _array(value, 4):
        _integer(_object(row)["rest"], 0, 27)


def _transfers(value: Any) -> None:
    rows = _array(value)
    if not rows:
        raise ValueError("empty transfers")
    senders, recipients = set(), set()
    for row in rows:
        sender, recipient, card = _array(row, 3)
        _integer(sender, 0, 3)
        _integer(recipient, 0, 3)
        _CARD.validate_python(card)
        if sender == recipient or sender in senders or recipient in recipients:
            raise ValueError("invalid transfer ownership")
        senders.add(sender)
        recipients.add(recipient)


def _select(message: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {field: message[field] for field in fields}


def _parse(message: dict[str, Any], config: SeatConnection,
           room_id: int | None) -> tuple[str, dict[str, Any]]:
    kind = _string(message["type"])
    if kind in ("CREATE_ROOM", "JOIN_ROOM"):
        expected = "CREATE_ROOM" if config.mode == "create" else "JOIN_ROOM"
        if kind != expected:
            raise ValueError("unexpected admission response")
        code = _integer(message["code"], -2**31, 2**31 - 1)
        data = _object(message["data"])
        if code != 200:
            _string(data["msg"])
            return "join_failed", {"code": code, "data": {"msg": data["msg"]}}
        seat = _integer(data["userNum"], 0, 3)
        received_room = _integer(data["roomId"])
        expected_room = room_id if room_id is not None else config.room_id
        if seat != config.seat or (expected_room is not None and received_room != expected_room):
            raise ValueError("admission ownership mismatch")
        return "joined", {"code": code, "data": {"roomId": received_room, "userNum": seat}}
    if kind == "JOIN":
        code = _integer(message["code"], -2**31, 2**31 - 1)
        announced_room = _integer(_object(message["data"])["roomId"])
        expected_room = room_id if room_id is not None else config.room_id
        if expected_room is not None and announced_room != expected_room:
            raise ValueError("announcement ownership mismatch")
        return "auxiliary_notice", {"type": kind, "code": code, "data": {"roomId": announced_room}}
    if kind == "tribute" and "stage" not in message:
        _integer(message["code"], -2**31, 2**31 - 1)
        _object(_object(message["data"])["actionList"])
        return "auxiliary_notice", _select(message, "code", "data")
    stage = _string(message["stage"])
    if kind == "act" and stage in ("play", "tribute", "back"):
        hand = _cards(message["handCards"])
        for field in ("selfRank", "oppoRank", "curRank"):
            _rank(message[field])
        parsed = parse_actions(message["actionList"], stage)
        for action in parsed.actions:
            if action.wire_action != _PASS and Counter(action.wire_action[2]) - Counter(hand):
                raise ValueError("candidate consumes absent cards")
        payload = _select(message, "handCards", "selfRank", "oppoRank", "curRank", "actionList")
        if "indexRange" in message:
            _integer(message["indexRange"])
            payload["indexRange"] = message["indexRange"]
        if stage == "play":
            _public_info(message["publicInfo"])
            if message["publicInfo"][config.seat]["rest"] != len(hand):
                raise ValueError("own count mismatch")
            _context_fields(message)
            _leader_fields(message)
            payload.update(_select(message, "publicInfo", "curPos", "curAction", "greaterPos", "greaterAction"))
        elif "publicInfo" in message:
            _public_info(message["publicInfo"])
            payload["publicInfo"] = message["publicInfo"]
        context = None
        if stage == "back":
            context = TributeContext(tribute_pos=message["tributePos"], tribute_card=message["tribute"])
            if context.tribute_pos == config.seat:
                raise ValueError("self tribute recipient")
            payload.update(_select(message, "tributePos", "tribute"))
        payload.update(stage=stage, parsed_actions=parsed.model_dump(mode="json"),
                       platform_context=context.model_dump(mode="json") if context else None)
        return "action_requested", payload
    if kind != "notify":
        raise ValueError("unknown message type")
    if stage == "beginning":
        if _integer(message["myPos"], 0, 3) != config.seat:
            raise ValueError("beginning ownership mismatch")
        _cards(message["handCards"])
        _rank(message["levelCard"])
        return "round_started", _select(message, "myPos", "handCards", "levelCard")
    if stage == "play":
        _integer(message["curPos"], 0, 3)
        if _action(message["curAction"])[0] in ("tribute", "back"):
            raise ValueError("nonplay notification")
        _leader_fields(message)
        return "play_observed", _select(message, "curPos", "curAction", "greaterPos", "greaterAction")
    if stage in ("tribute", "back"):
        _transfers(message["result"])
        payload = _select(message, "result")
        if stage == "back":
            _integer(message["count"], 0, 27)
            payload["count"] = message["count"]
        return ("tribute_announced" if stage == "tribute" else "exchange_completed"), payload
    if stage == "anti-tribute":
        count = _integer(message["antiNums"], 1, 2)
        seats = [_integer(seat, 0, 3) for seat in _array(message["antiPos"])]
        if count != len(seats) or len(set(seats)) != count:
            raise ValueError("inconsistent anti tribute")
        return "anti_tribute", _select(message, "antiNums", "antiPos")
    if stage == "episodeOver":
        _rank(message["curRank"])
        order = [_integer(seat, 0, 3) for seat in _array(message["order"], 4)]
        if len(set(order)) != 4:
            raise ValueError("invalid finishing order")
        revealed = set()
        for row in _array(message["restCards"]):
            seat, cards = _array(row, 2)
            _integer(seat, 0, 3)
            _cards(cards)
            if seat in revealed:
                raise ValueError("repeated revealed seat")
            revealed.add(seat)
        return "round_result", _select(message, "order", "curRank", "restCards")
    if stage == "gameResult":
        _integer(message["victory"], 0, 1)
        for rank in _array(message["victoryRank"], 2):
            _rank(rank)
        for field in ("victoryNum", "gameResult"):
            for count in _array(message[field], 4):
                _integer(count)
        _integer(message["gameRound"], 1)
        return "match_result", _select(message, "victory", "victoryRank", "victoryNum", "gameResult", "gameRound")
    if stage == "gameOver":
        for field in ("curTimes", "settingTimes", "gameRound"):
            _integer(message[field], 1)
        return "match_closed", _select(message, "curTimes", "settingTimes", "gameRound")
    raise ValueError("unknown stage")


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_platform_frame(raw: str | bytes | dict[str, Any], *, config: SeatConnection,
                         rx_seq: int, received_at: str, received_mono: float,
                         room_id: int | None = None) -> PlatformEvent:
    """Return unknown plus a static error on malformed frames; never guess moves.

    ``room_id`` is the previously confirmed room, not an ID inferred from a play.
    Transport metadata must already be valid and belongs to the connection.
    """
    message: Any = raw
    try:
        if isinstance(raw, (str, bytes)):
            message = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        kind, payload = _parse(_object(message), config, room_id)
        return PlatformEvent(session_id=config.session_id, connection_epoch=config.connection_epoch,
                             rx_seq=rx_seq, received_at=received_at, received_mono=received_mono,
                             kind=kind, payload=payload, raw_message=message)
    except (KeyError, ValueError, TypeError, UnicodeError, ValidationError):
        # Keep protocol diagnostics; no raw exception text enters the safe error.
        # Invalid/nonfinite JSON cannot be stored in a FrozenJSONObject.
        try:
            encoded = json.dumps(message, ensure_ascii=False, allow_nan=False)
            diagnostic = json.loads(encoded)
        except (ValueError, TypeError):
            diagnostic = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        error = ErrorInfo(code="INVALID_PLATFORM_FRAME", component="platform",
                          safe_message="Unrecognized or invalid platform frame", related_event_id=None)
        return PlatformEvent(session_id=config.session_id, connection_epoch=config.connection_epoch,
                             rx_seq=rx_seq, received_at=received_at, received_mono=received_mono,
                             kind="unknown", payload={"raw": diagnostic, "error": error.model_dump(mode="json")})
