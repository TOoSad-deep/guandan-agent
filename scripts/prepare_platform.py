"""Build the pinned Java platform in runtime/, without starting a game or model.

Exit codes: 0 success, 2 source/JDK/path validation, 3 missing build tool,
4 Maven build failure, 5 unexpected local I/O failure. No .env file is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import uuid

from guandan_agent.java_runtime import find_jdk_home, jdk_tool

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "e92db7fa9d915fb318b3be78703d5ccd4a09c887"
ARCHIVE_SHA256 = "cb4015c37a4c19e5e6f59d856ee98e1d5a3f5708e46f86411ee4267e790fddc5"
SNAPSHOT_NAME = "OpenGuanDan-Simulator-e92db7f"
MODULE_JARS = {
    "guandan-websocket": "guandan-java-pure-1.0.0.jar",
    "guandan": "guandan-java-action.jar",
}
BINDING_FILE = "guandan-websocket/src/main/java/com/guandan/socket/NettyWebSocketServer.java"
BINDING_BEFORE = b"bootstrap.bind(config.getServerPort())"
BINDING_AFTER = b'bootstrap.bind("127.0.0.1", config.getServerPort())'


class BuildError(Exception):
    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_snapshot(project: Path) -> dict[str, bytes]:
    """Check the pinned archive and every original file; never extract into source."""
    archive = project / "research/archives" / f"{SNAPSHOT_NAME}.tar.gz"
    if not archive.is_file() or digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise BuildError("Pinned archive missing or SHA-256 mismatch")
    snapshot = project / "research/sources" / SNAPSHOT_NAME
    prefix = f"OpenGuanDan-Simulator-{UPSTREAM_COMMIT}"
    files: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != prefix or ".." in parts:
                raise BuildError("Archive does not match the fixed upstream commit")
            if member.isdir():
                continue
            if not member.isfile():
                raise BuildError("Archive contains a non-regular source file")
            relative = PurePosixPath(*parts[1:]).as_posix()
            reader = bundle.extractfile(member)
            if reader is None or relative in files:
                raise BuildError("Archive contains an unreadable or duplicate file")
            data = reader.read()
            original = snapshot / relative
            if original.is_symlink() or not original.is_file() or original.read_bytes() != data:
                raise BuildError(f"Original snapshot differs from archive: {relative}")
            files[relative] = data
    return files


def safe_output(path: Path, runtime: Path) -> Path:
    """Reject symlink escapes before writing; never delete an existing directory."""
    if not path.is_relative_to(runtime):
        raise BuildError("Build output must stay inside the project's runtime directory")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise BuildError(f"Build output contains a symlink: {item}")
        if item == runtime:
            break
    return path


def prepare_workspace(files: dict[str, bytes], runtime: Path) -> Path:
    workspace = safe_output(runtime / "platform/workspace", runtime)
    expected = {
        name: data for name, data in files.items()
        if name.split("/", 1)[0] in MODULE_JARS
        and (name.endswith("/pom.xml") or "/src/" in name)
    }
    if BINDING_FILE not in expected or expected[BINDING_FILE].count(BINDING_BEFORE) != 1:
        raise BuildError("Unexpected upstream listener; loopback adaptation refused")
    expected[BINDING_FILE] = expected[BINDING_FILE].replace(BINDING_BEFORE, BINDING_AFTER)
    if workspace.exists():
        for existing in workspace.rglob("*"):
            safe_output(existing, runtime)
            relative = existing.relative_to(workspace).as_posix()
            if existing.is_file() and "/src/" in relative and relative not in expected:
                raise BuildError(f"Unexpected source in existing build copy: {relative}")
    for name, data in expected.items():
        output = safe_output(workspace / name, runtime)
        if output.exists() and (not output.is_file() or output.read_bytes() != data):
            raise BuildError(f"Existing build source changed; refusing to overwrite: {name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            output.write_bytes(data)
    return workspace


def require_jdk21(java_version: str, javac_version: str) -> None:
    runtime = re.search(r'(?:java|openjdk) version "(\d+)', java_version)
    compiler = re.search(r"\bjavac (\d+)", javac_version)
    if not runtime or not compiler or runtime.group(1) != "21" or compiler.group(1) != "21":
        raise BuildError("JDK 21 is required for this pinned platform (both java and javac)")


def version(command: list[str], env: dict[str, str]) -> str:
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BuildError(f"Unable to execute build tool: {command[0]}", 3) from error
    if result.returncode:
        raise BuildError(f"Build tool version check failed: {command[0]}", 3)
    return (result.stdout + result.stderr).strip()


def seed_cache(seed: Path, cache: Path, runtime: Path) -> int:
    """Copy missing cache files, keeping the seed read-only and all writes local."""
    seed = seed.resolve()
    if (not seed.is_dir() or seed.is_relative_to(cache.resolve())
            or cache.resolve().is_relative_to(seed)):
        raise BuildError("Maven cache seed must be an existing, separate directory")
    copied = 0
    for source in seed.rglob("*"):
        if source.is_symlink():
            raise BuildError("Maven cache seed may not contain symlinks")
        if source.is_file():
            target = safe_output(cache / source.relative_to(seed), runtime)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path,
                        help="JDK 21 directory; defaults to JAVA_HOME, local macOS JDK, then PATH")
    parser.add_argument("--offline", action="store_true", help="Forbid Maven dependency downloads")
    parser.add_argument("--maven-cache-seed", type=Path, help="Read-only seed copied into runtime cache")
    args = parser.parse_args(argv)
    runtime = PROJECT / "runtime"
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    logs = runtime / "platform/builds" / build_id
    metadata: dict = {
        "build_id": build_id, "upstream_commit": UPSTREAM_COMMIT,
        "archive_sha256": ARCHIVE_SHA256, "status": "failed", "commands": [],
        "adaptation": {"file": BINDING_FILE, "before": BINDING_BEFORE.decode(), "after": BINDING_AFTER.decode()},
        "model_calls": 0, "started_game": False,
    }
    code = 0
    try:
        safe_output(logs, runtime).mkdir(parents=True, exist_ok=False)
        files = verify_snapshot(PROJECT)
        metadata["source_files"] = {name: digest(data) for name, data in sorted(files.items())}
        try:
            java_home = find_jdk_home(args.java_home)
        except ValueError as error:
            raise BuildError(str(error), 3) from error
        env = os.environ.copy()
        env["JAVA_HOME"] = str(java_home)
        env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
        java_version = version([str(jdk_tool(java_home, "java")), "-version"], env)
        javac_version = version([str(jdk_tool(java_home, "javac")), "-version"], env)
        metadata.update(java_home=str(java_home), java_version=java_version, javac_version=javac_version)
        require_jdk21(java_version, javac_version)
        maven = shutil.which("mvn", path=env["PATH"])
        if maven is None:
            raise BuildError("Maven is required on PATH", 3)
        metadata["maven_version"] = version([maven, "-version"], env)
        if not re.search(r"Java version: 21(?:\.|,)", metadata["maven_version"]):
            raise BuildError("Maven is not using the selected JDK 21")
        workspace = prepare_workspace(files, runtime)
        cache = safe_output(runtime / "platform/cache/m2", runtime)
        cache.mkdir(parents=True, exist_ok=True)
        for existing in cache.rglob("*"):
            safe_output(existing, runtime)
        if args.maven_cache_seed:
            metadata["seeded_cache_files"] = seed_cache(args.maven_cache_seed, cache, runtime)
        artifacts = []
        for module, jar_name in MODULE_JARS.items():
            command = [maven, "-B", "-ntp"] + (["--offline"] if args.offline else [])
            command += ["-f", str(workspace / module / "pom.xml"), f"-Dmaven.repo.local={cache}", "package"]
            log = logs / f"{module}.log"
            entry = {"command": command, "cwd": str(workspace / module), "log": str(log)}
            metadata["commands"].append(entry)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=workspace / module, env=env, stdout=stream, stderr=subprocess.STDOUT)
            entry["exit_code"] = result.returncode
            if result.returncode:
                raise BuildError(f"{module} failed; inspect {log}", 4)
            artifact = safe_output(workspace / module / "target" / jar_name, runtime)
            if not artifact.is_file():
                raise BuildError(f"Expected JAR missing: {artifact}", 4)
            artifacts.append({"path": str(artifact), "sha256": digest(artifact.read_bytes()), "bytes": artifact.stat().st_size})
        if verify_snapshot(PROJECT) != files:
            raise BuildError("Original source changed during build")
        metadata.update(status="completed", artifacts=artifacts, source_unchanged=True)
    except BuildError as error:
        code = error.code
        metadata["error"] = str(error)
    except (OSError, tarfile.TarError) as error:
        code = 5
        metadata["error"] = f"{type(error).__name__}: {error}"
    metadata["exit_code"] = code
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    if logs.is_dir() and not logs.is_symlink():
        try:
            safe_output(logs / "build.json", runtime).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, BuildError):
            code = code or 5
    print(json.dumps({"status": metadata["status"], "exit_code": code, "metadata": str(logs / "build.json"), "error": metadata.get("error")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
