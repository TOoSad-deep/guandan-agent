"""Smoke the exact read-only fixtures used for browser inspection."""
import asyncio
import json
from pathlib import Path

import httpx

from guandan_agent.app import create_web_application
from guandan_agent.recording.query import QueryError, QueryService

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/web"


def test_demo_candidate_faces_and_report_links_come_from_selected_request(tmp_path):
    async def scenario():
        index = json.loads((FIXTURES / "fixture-index.json").read_text())
        sample = index["review_changed"]
        app = create_web_application(recording_root=tmp_path, allowed_config_names=(),
                                     archive_root=FIXTURES)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as client:
                assert (await client.get("/api/v1/configs")).json() == {"items": []}
                assert len((await client.get("/api/v1/runs")).json()["items"]) == 3
                base = f"/api/v1/runs/{sample['run_id']}"
                detail = (await client.get(f"{base}/sessions/{sample['session_id']}/decisions/{sample['request_id']}")).json()
                assert len(detail["candidate_actions"]) == len(detail["candidates"]["candidates"]) == 4
                assert {a["action_id"] for a in detail["candidate_actions"]} == {c["action_id"] for c in detail["candidates"]["candidates"]}
                assert any(a["wire_action"] == ["Single", "K", ["SK"]] for a in detail["candidate_actions"])
                completed = (await client.get(base + "/reports/demo-completed")).json()
                partial = (await client.get(base + "/reports/demo-partial")).json()
                assert completed["items"][0]["evidence_available"] is True
                assert partial["items"][1]["evidence_available"] is False
                assert app.state.manager.list_views() == ()
    asyncio.run(scenario())


def test_truncated_window_can_recover_using_only_the_server_cursor(tmp_path):
    async def scenario():
        from guandan_agent.app import create_manager
        source = Path(__file__).resolve().parents[1] / "fixtures/audit/valid_scripted_run"
        queries = QueryService(create_manager(recording_root=tmp_path), cache_events=60)
        view = await queries.register_archive(source)
        for session in view.sessions:
            cursor = 0
            for _ in range(5):
                try:
                    page = queries.decisions(view.run_id, session.session_id, after_seq=cursor)
                    assert all(item.opened_seq > cursor for item in page.items)
                    break
                except QueryError as exc:
                    assert exc.status_code == 410 and exc.retained_from_seq > cursor + 1
                    cursor = exc.retained_from_seq - 1
            else:
                raise AssertionError("History recovery must advance and terminate")
    asyncio.run(scenario())
