#!/usr/bin/env python3
"""Read-only-source, local OpenGuanDan WebSocket protocol experiments.

Run with the phase-3 venv (requires websockets):
  python probe_platform.py --mode all --url ws://127.0.0.1:18181

Four independent WebSocket connections receive their own hand/request. The
decision function receives ONLY that seat's current request, never the recorder
or another seat's hand. Combined logs are offline evidence, not strategy input.
Each mode creates a separate room and closes only its own client connections.
No server source changes, server restarts, training, or external model calls.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import uuid

import websockets


PASS = ["PASS", "PASS", "PASS"]
MODES = ("normal", "invalid", "replay", "disconnect", "timeout")


def action_key(action: Any) -> str:
    """Compare reconstructed broadcasts; sending always preserves original list."""
    if isinstance(action, list) and len(action) == 3 and isinstance(action[2], list):
        action = [action[0], action[1], sorted(action[2])]
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def choose_action(request: dict[str, Any], own_seat: int) -> list[Any]:
    """Deterministic, legal-list-only policy using this player's visible request."""
    actions = request.get("actionList")
    if not isinstance(actions, list) or not actions:
        raise ValueError("No nonempty actionList in current request")
    if request.get("stage") != "play":
        return actions[0]
    playable = [a for a in actions if a != PASS]
    if not playable:
        return actions[0]
    hand_size = len(request.get("handCards", []))
    finishing = [a for a in playable if isinstance(a[2], list) and len(a[2]) == hand_size]
    if finishing:
        return finishing[0]
    greater_seat = request.get("greaterPos", -1)
    if greater_seat in range(4) and greater_seat % 2 == own_seat % 2 and PASS in actions:
        return next(a for a in actions if a == PASS)
    # Greedy only: largest offered combination, stable original-list tie-break.
    return max(playable, key=lambda a: len(a[2]) if isinstance(a[2], list) else 0)


class RoomProbe:
    def __init__(self, args: argparse.Namespace, mode: str, output: Path):
        self.args, self.mode, self.output = args, mode, output
        output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.events: asyncio.Queue[tuple[int, int, Any]] = asyncio.Queue()
        self.connections: dict[int, Any] = {}
        self.readers: list[asyncio.Task] = []
        self.generations = Counter()
        self.files = {s: (output / f"seat-{s}.jsonl").open("w", encoding="utf-8") for s in range(4)}
        self.event_file = (output / "events.jsonl").open("w", encoding="utf-8")
        self.users = {s: f"phase3-{mode}-{uuid.uuid4().hex[:10]}-{s}" for s in range(4)}
        self.room_id: int | None = None
        self.seq = 0
        self.request_seq = Counter()
        self.pending: list[dict[str, Any]] = []
        self.finished = {"gameOver": set(), "gameResult": set()}
        self.summary: dict[str, Any] = {
            "mode": mode, "url": args.url,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": "e92db7fa9d915fb318b3be78703d5ccd4a09c887",
            "strategy": "deterministic own-request legal-list greedy with public teammate passing",
            "information_boundary": "Each decision receives only that seat's request. Multi-seat raw logs are offline evidence; no cross-seat hand information is used for choosing actions.",
            "requested_full_matches": 1,
            "requests_by_stage": {}, "sent_by_stage": {}, "confirmed_by_stage": {},
            "notifications_by_stage": {}, "first_level_by_seat": {},
            "settlements": [], "errors": [], "observations": [],
            "status": "running", "complete_match": False,
        }

    def tick(self) -> float:
        return round(time.monotonic() - self.started, 6)

    @staticmethod
    def bump(target: dict, key: str) -> None:
        target[key] = target.get(key, 0) + 1

    def record(self, seat: int, generation: int, direction: str, message: Any) -> int:
        self.seq += 1
        item = {"sequence": self.seq, "elapsed_seconds": self.tick(), "seat": seat,
                "connection_generation": generation, "direction": direction, "message": message}
        self.files[seat].write(json.dumps(item, ensure_ascii=False) + "\n")
        self.files[seat].flush()
        return self.seq

    def event(self, name: str, **data: Any) -> None:
        self.event_file.write(json.dumps({"elapsed_seconds": self.tick(), "event": name, **data}, ensure_ascii=False) + "\n")
        self.event_file.flush()

    async def connect(self, seat: int) -> None:
        connection = await websockets.connect(self.args.url, max_size=16 * 1024 * 1024,
                                               proxy=None, ping_interval=None,
                                               open_timeout=10, close_timeout=2)
        self.generations[seat] += 1
        generation = self.generations[seat]
        self.connections[seat] = connection
        self.event("connected", seat=seat, generation=generation)

        async def reader() -> None:
            try:
                async for raw in connection:
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        message = {"unparsed_wire_text": str(raw)}
                    self.record(seat, generation, "receive", message)
                    await self.events.put((seat, generation, message))
            except Exception as exc:
                self.event("reader_exception", seat=seat, generation=generation,
                           exception=f"{type(exc).__name__}: {exc}")
            finally:
                self.event("connection_closed", seat=seat, generation=generation)
        self.readers.append(asyncio.create_task(reader()))

    async def send(self, seat: int, message: dict[str, Any]) -> int:
        sequence = self.record(seat, self.generations[seat], "send", message)
        await self.connections[seat].send(json.dumps(message, ensure_ascii=False))
        return sequence

    def observe(self, seat: int, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if message.get("code") not in (None, 200):
            self.summary["errors"].append({"seat": seat, "message": message})
        stage = message.get("stage")
        if message.get("type") == "act":
            self.request_seq[seat] += 1
            self.bump(self.summary["requests_by_stage"], str(stage))
        if message.get("type") != "notify":
            return
        self.bump(self.summary["notifications_by_stage"], str(stage))
        if stage == "beginning":
            self.summary["first_level_by_seat"].setdefault(str(seat), message.get("levelCard"))
        if stage in self.finished:
            self.finished[stage].add(seat)
        if stage in {"episodeOver", "gameOver", "gameResult"} and seat == 0:
            self.summary["settlements"].append(message)
        # Confirmation is observed on the submitting seat's own connection only.
        for pending in self.pending:
            if pending["confirmed"] or pending["seat"] != seat or pending["stage"] != stage:
                continue
            action = pending["action"]
            matches = False
            exact = False
            if stage == "play" and message.get("curPos") == seat:
                matches = action_key(message.get("curAction")) == action_key(action)
                exact = message.get("curAction") == action
            elif stage in {"tribute", "back"}:
                matches = any(isinstance(row, list) and len(row) >= 3 and row[0] == seat
                              and isinstance(action[2], list) and action[2] == [row[2]]
                              for row in message.get("result", []))
                exact = matches
            if matches:
                pending["confirmed"] = True
                pending["confirmed_at_seconds"] = self.tick()
                pending["confirmation_latency_seconds"] = round(self.tick() - pending["sent_at_seconds"], 6)
                pending["confirmation"] = message
                pending["exact_action_echo"] = exact
                self.bump(self.summary["confirmed_by_stage"], str(stage))
                self.event("action_confirmed", **pending)
                break

    async def receive(self, timeout: float | None = None) -> tuple[int, Any]:
        seat, generation, message = await asyncio.wait_for(self.events.get(), timeout=timeout)
        self.observe(seat, message)
        return seat, message

    async def until_message(self, seat: int, kind: str, timeout: float = 10) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            received_seat, message = await self.receive(max(0.01, deadline - time.monotonic()))
            if received_seat == seat and isinstance(message, dict) and message.get("type") == kind:
                return message

    async def setup(self) -> None:
        for seat in range(4):
            await self.connect(seat)
        await self.send(0, {"type": "CREATE_ROOM", "data": {
            "userId": self.users[0], "round": 1, "seatNum": 0}})
        response = await self.until_message(0, "CREATE_ROOM")
        if response.get("code") != 200:
            raise RuntimeError(f"Room creation failed: {response}")
        self.room_id = response["data"]["roomId"]
        self.summary["room_id"] = self.room_id
        for seat in range(1, 4):
            await self.send(seat, {"type": "JOIN_ROOM", "data": {
                "userId": self.users[seat], "roomId": self.room_id, "seatNum": seat}})
            response = await self.until_message(seat, "JOIN_ROOM")
            if response.get("code") != 200:
                raise RuntimeError(f"Seat join failed: {response}")

    async def submit(self, seat: int, request: dict, action: list, *, label: str = "legal",
                     raw_frame: dict | None = None) -> dict:
        stage = request["stage"]
        data = {"roomId": self.room_id, "player": seat, "act": action}
        if stage == "back":
            data.update(tributePos=request["tributePos"], tribute=request["tribute"])
        frame = raw_frame or {"type": {"play": "PLAY", "tribute": "TRIBUTE", "back": "PAYTRIBUTE"}[stage], "data": data}
        in_candidates = action in request.get("actionList", [])
        if label == "legal" and not in_candidates:
            raise ValueError("Local candidate membership check failed")
        pending = {"seat": seat, "stage": stage, "request_number": self.request_seq[seat],
                   "action": action, "frame": frame, "label": label,
                   "in_current_candidates": in_candidates, "sent_at_seconds": self.tick(),
                   "confirmed": False}
        self.pending.append(pending)
        pending["send_sequence"] = await self.send(seat, frame)
        self.bump(self.summary["sent_by_stage"], stage)
        self.event("action_submitted", **pending)
        return pending

    async def observe_window(self, seconds: float) -> list[dict]:
        deadline = time.monotonic() + seconds
        captured = []
        while time.monotonic() < deadline:
            try:
                seat, message = await self.receive(deadline - time.monotonic())
            except asyncio.TimeoutError:
                break
            captured.append({"seat": seat, "message": message})
        return captured

    async def first_play_request(self) -> tuple[int, dict]:
        while True:
            seat, message = await self.receive(15)
            if isinstance(message, dict) and message.get("type") == "act" and message.get("stage") == "play":
                return seat, message

    async def normal(self) -> None:
        while not all(len(seats) == 4 for seats in self.finished.values()):
            seat, message = await self.receive(20)
            if isinstance(message, dict) and message.get("type") == "act":
                await self.submit(seat, message, choose_action(message, seat))
        self.summary["complete_match"] = True
        self.summary["status"] = "completed_match"

    async def invalid(self) -> None:
        seat, request = await self.first_play_request()
        if PASS in request.get("actionList", []):
            raise RuntimeError("Initial lead unexpectedly permits PASS; invalid-PASS probe cannot establish nonmembership")
        pending = await self.submit(seat, request, PASS.copy(), label="intentional_invalid_lead_pass")
        captured = await self.observe_window(self.args.observe_seconds)
        self.summary["observations"].append({
            "test": "PASS absent from first lead actionList", "window_seconds": self.args.observe_seconds,
            "confirmed_despite_not_offered": pending["confirmed"],
            "received_messages": len(captured),
            "interpretation": "Broadcast confirmation proves this non-offered action was processed; lack of feedback alone does not prove rejection.",
        })
        self.summary["status"] = "observation_complete"

    async def replay(self) -> None:
        first_seat, first_request = await self.first_play_request()
        old_action = choose_action(first_request, first_seat)
        original = await self.submit(first_seat, first_request, old_action)
        while True:
            seat, message = await self.receive(20)
            if not isinstance(message, dict) or message.get("type") != "act":
                continue
            if seat == first_seat and message.get("stage") == "play" and old_action not in message.get("actionList", []):
                replayed = await self.submit(seat, message, old_action, label="stale_frame_replay", raw_frame=original["frame"])
                captured = await self.observe_window(self.args.observe_seconds)
                self.summary["observations"].append({
                    "test": "Replay previous raw PLAY frame when same seat has a new request but old action is absent",
                    "original_confirmed": original["confirmed"], "replay_confirmed": replayed["confirmed"],
                    "old_action_in_new_candidates": False, "received_messages": len(captured),
                    "window_seconds": self.args.observe_seconds,
                    "interpretation": "Same-seat stale-action processing test; does not imply cross-seat authorization or all duplicate cases are covered.",
                })
                self.summary["status"] = "observation_complete"
                return
            await self.submit(seat, message, choose_action(message, seat))

    async def disconnect(self) -> None:
        seat, request = await self.first_play_request()
        self.event("deliberate_disconnect", seat=seat, room_id=self.room_id)
        await self.connections[seat].close()
        await self.observe_window(0.5)
        await self.connect(seat)
        await self.send(seat, {"type": "JOIN_ROOM", "data": {
            "userId": self.users[seat], "roomId": self.room_id, "seatNum": seat}})
        captured = await self.observe_window(self.args.observe_seconds)
        joins = [x["message"] for x in captured if x["seat"] == seat and x["message"].get("type") == "JOIN_ROOM"]
        self.summary["observations"].append({
            "test": "Close active seat and reconnect same user to same room/seat", "seat": seat,
            "join_responses": joins,
            "new_beginning_messages": sum(x["seat"] == seat and x["message"].get("stage") == "beginning" for x in captured),
            "new_action_requests": sum(x["seat"] == seat and x["message"].get("type") == "act" for x in captured),
            "window_seconds": self.args.observe_seconds,
            "interpretation": "JOIN success alone would not prove state restoration; no old action is replayed after reconnect.",
        })
        self.summary["status"] = "observation_complete"

    async def timeout(self) -> None:
        seat, request = await self.first_play_request()
        started = time.monotonic()
        captured = await self.observe_window(self.args.observe_seconds)
        elapsed = time.monotonic() - started
        progress = [x for x in captured if x["message"].get("type") == "notify"
                    and x["message"].get("stage") in {"play", "episodeOver", "gameOver", "gameResult"}]
        accepted_after_wait = None
        if not progress:
            pending = await self.submit(seat, request, choose_action(request, seat))
            await self.observe_window(min(self.args.observe_seconds, 3))
            accepted_after_wait = pending["confirmed"]
        self.summary["observations"].append({
            "test": "Withhold first action for bounded observation window", "observed_wait_seconds": round(elapsed, 6),
            "progress_notifications_during_wait": len(progress),
            "original_action_confirmed_after_wait": accepted_after_wait,
            "interpretation": "No timeout observed in this window cannot establish that no timeout exists, nor the teacher's match limit.",
        })
        self.summary["status"] = "observation_complete"

    async def run(self) -> dict:
        try:
            async with asyncio.timeout(self.args.max_seconds):
                await self.setup()
                await getattr(self, self.mode)()
        except Exception as exc:
            self.summary["status"] = "failed_or_incomplete"
            self.summary["errors"].append({"exception": f"{type(exc).__name__}: {exc}"})
            self.event("experiment_exception", exception=f"{type(exc).__name__}: {exc}")
        finally:
            await asyncio.gather(*(ws.close() for ws in self.connections.values()), return_exceptions=True)
            await asyncio.gather(*self.readers, return_exceptions=True)
            self.summary["elapsed_seconds"] = self.tick()
            self.summary["sent_actions"] = len(self.pending)
            self.summary["confirmed_actions"] = sum(p["confirmed"] for p in self.pending)
            self.summary["unconfirmed_actions"] = [p for p in self.pending if not p["confirmed"]]
            self.summary["game_over_seen_by_seats"] = sorted(self.finished["gameOver"])
            self.summary["game_result_seen_by_seats"] = sorted(self.finished["gameResult"])
            self.summary["raw_logs"] = [str((self.output / f"seat-{s}.jsonl").resolve()) for s in range(4)]
            self.summary["coverage"] = {
                "play_confirmed": self.summary["confirmed_by_stage"].get("play", 0) > 0,
                "pass_confirmed": any(p["confirmed"] and p["action"] == PASS for p in self.pending),
                "tribute_confirmed": self.summary["confirmed_by_stage"].get("tribute", 0) > 0,
                "back_confirmed": self.summary["confirmed_by_stage"].get("back", 0) > 0,
                "anti_tribute_observed": self.summary["notifications_by_stage"].get("anti-tribute", 0) > 0,
            }
            (self.output / "summary.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for handle in self.files.values():
                handle.close()
            self.event_file.close()
        return self.summary


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18181")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="normal")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-seconds", type=float, default=180, help="Per-experiment hard limit")
    parser.add_argument("--observe-seconds", type=float, default=5, help="Bounded negative-result observation window")
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.observe_seconds <= 0:
        parser.error("Time limits must be positive")
    output = args.out or Path(__file__).parent / "results" / ("platform-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    modes = MODES if args.mode == "all" else (args.mode,)
    summaries = []
    for mode in modes:
        result = await RoomProbe(args, mode, output / mode).run()
        summaries.append(result)
        print(json.dumps({"mode": mode, "status": result["status"], "room_id": result.get("room_id"),
                          "complete_match": result["complete_match"], "sent": result["sent_actions"],
                          "confirmed": result["confirmed_actions"], "coverage": result["coverage"],
                          "output": str((output / mode).resolve())}, ensure_ascii=False), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(any(s["status"] == "failed_or_incomplete" for s in summaries))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
