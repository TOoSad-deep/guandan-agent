"""A mislabeled or changed experiment must not enter the final comparison."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from guandan_agent.config import RunConfig
from scripts import summarize_b10


@pytest.mark.parametrize("changed", ["config_id", "effective_config"])
def test_summary_rejects_archive_outside_frozen_version(tmp_path, monkeypatch, changed):
    frozen = RunConfig()
    other = RunConfig.model_validate({"model": {"timeout_s": 90.0, "max_output_tokens": 8192}})
    out = tmp_path / "artifacts/b10"
    (out / "live/V0").mkdir(parents=True)
    (out / "frozen-plan.json").write_text(json.dumps({"configs": {"V0": {
        "config_id": frozen.config_id, "config": frozen.model_dump(mode="json")}}}))
    (out / "live/V0/result.json").write_text(json.dumps({"run_id": "wrong-version"}))
    run = tmp_path / "runs/wrong-version"
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text("")
    record = SimpleNamespace(config_id=other.config_id if changed == "config_id" else frozen.config_id,
                             config=frozen if changed == "config_id" else other)

    async def load(*args, **kwargs):
        return record

    monkeypatch.setattr(summarize_b10, "ROOT", tmp_path)
    monkeypatch.setattr(summarize_b10, "OUT", out)
    monkeypatch.setattr(summarize_b10.ClosedRunLog, "load", load)
    with pytest.raises(ValueError, match="frozen experiment configuration"):
        asyncio.run(summarize_b10.main())
    assert not (out / "comparison.json").exists()
