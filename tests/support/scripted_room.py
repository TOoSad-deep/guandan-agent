"""A three-play script, not an implementation of Guandan game rules.

Private hands live in this test server. Real policies receive only their own
Session projection. The production parser, sender and gate still execute.
"""

from guandan_agent.platform import ScriptedPlatform


class ScriptedRooms:
    def __init__(self, clock, *, auto_play=True, pair_options=False):
        self.clock = clock
        self.auto_play = auto_play
        self.pair_options = pair_options
        self.rooms = {}
        self.platforms = []
        self.writes = []

    def factory(self, connection):
        platform = _RoomPlatform(self, connection)
        self.platforms.append(platform)
        return platform

    def admit(self, platform):
        connection = platform.connection
        if connection.mode == "create":
            room_id = 700 + len(self.rooms)
            self.rooms[room_id] = {"members": {}, "hands": {seat: [card] for seat, card in
                                                            enumerate(("S8", "S9", "ST", "SJ"))},
                                   "played": [], "last_action": None, "last_seat": -1}
            if self.pair_options:
                self.rooms[room_id]["hands"] = {seat: ["S" + rank, "H" + rank]
                                                 for seat, rank in enumerate("89TJ")}
        else:
            room_id = connection.room_id
        room = self.rooms[room_id]
        platform.script_room_id = room_id
        room["members"][connection.seat] = platform
        platform.feed({"type": "CREATE_ROOM" if connection.mode == "create" else "JOIN_ROOM",
                       "code": 200, "data": {"roomId": room_id, "userNum": connection.seat}})
        if len(room["members"]) == 4:
            for seat, member in room["members"].items():
                member.feed({"type": "notify", "stage": "beginning", "myPos": seat,
                             "handCards": room["hands"][seat].copy(), "levelCard": "2"})
            if self.auto_play:
                self._request(room, 0)

    def _broadcast(self, room, frame):
        for member in room["members"].values():
            member.feed(frame)

    def _request(self, room, seat):
        card = room["hands"][seat][0]
        room["members"][seat].feed({
            "type": "act", "stage": "play", "handCards": room["hands"][seat].copy(),
            "selfRank": "2", "oppoRank": "2", "curRank": "2", "indexRange": 0,
            "publicInfo": [{"rest": len(room["hands"][s])} for s in range(4)],
            "curPos": room["last_seat"], "curAction": room["last_action"],
            "greaterPos": room["last_seat"], "greaterAction": room["last_action"],
            "actionList": ([["Pair", card[1], room["hands"][seat].copy()],
                            ["Pair", card[1], list(reversed(room["hands"][seat]))]]
                           if self.pair_options else [["Single", card[1], [card]]]),
        })

    def accept(self, platform, intent):
        room = self.rooms[platform.script_room_id]
        seat = platform.connection.seat
        assert seat == len(room["played"])
        action = intent.model_dump(mode="json")["wire_action"]
        assert action[:2] == ["Pair" if self.pair_options else "Single", room["hands"][seat][0][1]]
        assert sorted(action[2]) == sorted(room["hands"][seat])
        self.writes.append((platform.script_room_id, seat, intent))
        self._broadcast(room, {"type": "notify", "stage": "play", "curPos": seat,
                               "curAction": action, "greaterPos": room["last_seat"],
                               "greaterAction": room["last_action"]})
        room["hands"][seat] = []
        room["played"].append(seat)
        room["last_seat"], room["last_action"] = seat, action
        if seat < 2:
            self._request(room, seat + 1)
        else:
            self._broadcast(room, {"type": "notify", "stage": "episodeOver", "order": [0, 1, 2, 3],
                                   "curRank": "2", "restCards": [[3, room["hands"][3]]]})
            self._broadcast(room, {"type": "notify", "stage": "gameResult", "victory": 0,
                                   "victoryRank": ["3", "2"], "victoryNum": [1, 0, 0, 0],
                                   "gameResult": [1, 0, 0, 0], "gameRound": 1})
            self._broadcast(room, {"type": "notify", "stage": "gameOver", "curTimes": 1,
                                   "settingTimes": 1, "gameRound": 1})


class _RoomPlatform(ScriptedPlatform):
    def __init__(self, room, connection):
        super().__init__(clock=room.clock)
        self.room = room
        self.connection = connection
        self.script_room_id = None

    async def connect(self, connection):
        await super().connect(connection)
        self.room.admit(self)

    async def write(self, intent, gate):
        result = await super().write(intent, gate)
        if result.status == "sent":
            self.room.accept(self, intent)
        return result
