"""Source/JDK boundaries of the build entry point; no Maven or network required."""
from pathlib import Path
import shutil

import pytest

from scripts import prepare_platform as build


def test_fixed_archive_and_loopback_copy_leave_original_unchanged(tmp_path: Path) -> None:
    files = build.verify_snapshot(build.PROJECT)
    assert len(files) == 65
    assert build.UPSTREAM_COMMIT == "e92db7fa9d915fb318b3be78703d5ccd4a09c887"
    runtime = tmp_path / "runtime"
    workspace = build.prepare_workspace(files, runtime)
    assert build.BINDING_AFTER in (workspace / build.BINDING_FILE).read_bytes()
    assert build.BINDING_BEFORE in files[build.BINDING_FILE]
    assert build.prepare_workspace(files, runtime) == workspace
    assert build.verify_snapshot(build.PROJECT) == files
    (workspace / build.BINDING_FILE).write_bytes(b"local edit")
    with pytest.raises(build.BuildError, match="refusing to overwrite"):
        build.prepare_workspace(files, runtime)
    assert (workspace / build.BINDING_FILE).read_bytes() == b"local edit"


def test_archive_tampering_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "research/archives" / f"{build.SNAPSHOT_NAME}.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"untrusted archive")
    with pytest.raises(build.BuildError, match="SHA-256 mismatch"):
        build.verify_snapshot(tmp_path)


def test_source_tampering_is_rejected_without_overwriting(tmp_path: Path) -> None:
    archive = tmp_path / "research/archives" / f"{build.SNAPSHOT_NAME}.tar.gz"
    archive.parent.mkdir(parents=True)
    shutil.copyfile(build.PROJECT / "research/archives" / archive.name, archive)
    source = tmp_path / "research/sources" / build.SNAPSHOT_NAME
    source.mkdir(parents=True)
    (source / "README.md").write_text("local edit")
    with pytest.raises(build.BuildError, match="Original snapshot differs"):
        build.verify_snapshot(tmp_path)
    assert (source / "README.md").read_text() == "local edit"


@pytest.mark.parametrize("java,javac", [
    ('openjdk version "26.0.1"', "javac 26.0.1"),
    ('java version "21.0.11"', "javac 17.0.2"),
    ("unrecognized", "javac 21.0.11"),
])
def test_java_mismatch_is_rejected(java: str, javac: str) -> None:
    with pytest.raises(build.BuildError, match="JDK 21"):
        build.require_jdk21(java, javac)


def test_jdk21_and_symlink_output_boundary(tmp_path: Path) -> None:
    build.require_jdk21('openjdk version "21.0.11" 2026-04-21', "javac 21.0.11")
    runtime = tmp_path / "runtime"
    runtime.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(build.BuildError, match="symlink"):
        build.safe_output(runtime / "platform/build.json", runtime)


def test_cache_seed_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "artifact.jar").write_bytes(b"seed")
    runtime = tmp_path / "runtime"
    cache = runtime / "platform/cache/m2"
    assert build.seed_cache(seed, cache, runtime) == 1
    (cache / "artifact.jar").write_bytes(b"current")
    assert build.seed_cache(seed, cache, runtime) == 0
    assert (cache / "artifact.jar").read_bytes() == b"current"
    assert (seed / "artifact.jar").read_bytes() == b"seed"
    with pytest.raises(build.BuildError, match="separate directory"):
        build.seed_cache(tmp_path, cache, runtime)
