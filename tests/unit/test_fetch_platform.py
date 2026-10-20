"""Pinned download integrity and repeatability, using local archive bytes only."""
from pathlib import Path

import pytest

from scripts import fetch_platform as fetcher
from scripts.prepare_platform import BuildError, SNAPSHOT_NAME, verify_snapshot


def test_cached_archive_restores_source_and_repeat_fetch_is_offline(tmp_path: Path):
    archive = Path("research/archives") / f"{SNAPSHOT_NAME}.tar.gz"
    target = tmp_path / archive
    target.parent.mkdir(parents=True)
    target.write_bytes((fetcher.PROJECT / archive).read_bytes())
    assert fetcher.fetch(tmp_path) == 65
    assert len(verify_snapshot(tmp_path)) == 65
    assert fetcher.fetch(tmp_path) == 65


def test_wrong_archive_never_installs_source(tmp_path: Path):
    archive = tmp_path / "research/archives" / f"{SNAPSHOT_NAME}.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"not the pinned upstream archive")
    with pytest.raises(BuildError, match="SHA-256 mismatch"):
        fetcher.fetch(tmp_path)
    assert not (tmp_path / "research/sources" / SNAPSHOT_NAME).exists()
    assert archive.read_bytes() == b"not the pinned upstream archive"


def test_modified_source_is_not_overwritten(tmp_path: Path):
    source = tmp_path / "research/sources" / SNAPSHOT_NAME
    source.mkdir(parents=True)
    readme = source / "README.md"
    readme.write_text("local edit")
    with pytest.raises(BuildError):
        fetcher.fetch(tmp_path)
    assert readme.read_text() == "local edit"
