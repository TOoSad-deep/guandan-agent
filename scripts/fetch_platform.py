"""Fetch and verify the pinned upstream platform; never build or execute it.

Run from the repository root: uv run python -m scripts.fetch_platform
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from scripts.prepare_platform import (
    ARCHIVE_SHA256, PROJECT, SNAPSHOT_NAME, UPSTREAM_COMMIT, BuildError, verify_snapshot,
)

URL = ("https://codeload.github.com/GameAI-NJUPT/OpenGuanDan-Simulator/tar.gz/"
       + UPSTREAM_COMMIT)


def fetch(project: Path = PROJECT) -> int:
    archive = project / "research/archives" / f"{SNAPSHOT_NAME}.tar.gz"
    source = project / "research/sources" / SNAPSHOT_NAME
    if source.exists():
        return len(verify_snapshot(project))
    if archive.exists():
        payload = archive.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "guandan-agent/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(30 * 1024 * 1024 + 1)
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise BuildError("Pinned archive SHA-256 mismatch; no snapshot was installed")

    (project / "research").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".platform-", dir=project / "research") as temporary:
        staging = Path(temporary)
        staged_archive = staging / "research/archives" / archive.name
        staged_archive.parent.mkdir(parents=True)
        staged_archive.write_bytes(payload)
        sources = staging / "research/sources"
        sources.mkdir(parents=True)
        with tarfile.open(staged_archive, "r:gz") as bundle:
            bundle.extractall(sources, filter="data")
        extracted = sources / f"OpenGuanDan-Simulator-{UPSTREAM_COMMIT}"
        staged_source = sources / SNAPSHOT_NAME
        extracted.rename(staged_source)
        count = len(verify_snapshot(staging))
        archive.parent.mkdir(parents=True, exist_ok=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            staged_archive.rename(archive)
        staged_source.rename(source)
    return count


def main() -> int:
    try:
        count = fetch()
    except (BuildError, OSError, tarfile.TarError) as error:
        print(f"Platform fetch failed: {error}")
        return 2
    print(f"Verified {count} upstream files at research/sources/{SNAPSHOT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
