#!/usr/bin/env python3
"""Prepare a local B10 delivery ZIP from an explicit allowlist; never publish.

The default is a candidate package. ``--final`` additionally requires all three
closed final runs and the named delivery documents. No credentials are loaded,
no external commands are executed, and no live game/model is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from uuid import UUID
import zipfile

from scripts.prepare_platform import ARCHIVE_SHA256, SNAPSHOT_NAME, UPSTREAM_COMMIT


ROOT = Path(__file__).resolve().parents[1]
TOP = "guandan-agent-b10"
BASE_FILES = (
    "README.md", "pyproject.toml", "uv.lock", ".python-version", ".env.example",
    "docs/examples/phase5-contract-example.json",
)
CONFIG_FILES = (
    "local.example.toml", "offline-facts.example.toml", "b05-v1.example.toml",
    "b04-v0.toml", "b06-v1.toml", "b07-v2-fixed.toml", "b09-review.toml",
    "b09-review-remainder.toml", "b10-v0.toml", "b10-v1.toml", "b10-v2.toml",
)
SCRIPT_FILES = (
    "__init__.py", "prepare_platform.py", "verify_v0_run.py", "verify_b06_run.py",
    "run_b06_probe.py", "probe_b07_review.py", "serve_b08_demo.py",
    "run_model_validation25.py", "run_model_retest15.py", "run_b10_batch.py",
    "verify_b10_run.py", "summarize_b10.py", "package_b10_delivery.py",
)
DESIGN_FILES = (
    "项目目标与核心功能.md", "需求与接口规格.md", "架构与技术选型.md",
    "实施计划.md", "阶段七实施批次.md", "开发流程与阶段产出.md",
    "事实语义修复记录.md", "输入摘要与解释分离修复.md",
    "原3个错误场景定点复测.md", "修复后15次复测.md", "追加25次模型实测.md",
    "扩展24次复测.md", "High档24次对照复测.md",
)
REPLAY_FILES = ("events.jsonl", "seat-0.jsonl", "seat-1.jsonl", "seat-2.jsonl",
                "seat-3.jsonl", "summary.json")
B10_FILES = (
    "frozen-plan.json", "preserved-input-hashes.json", "verification.json",
    "credential-scan.json", "clean-install-verification.json", "source-integrity.json",
    "comparison.json", "evaluation-match.json", "evaluation-round.json",
    "v0-legacy-audit.json", "V0-audit.json", "V1-audit.json", "V2-audit.json",
    "v0-evaluation.json", "v1-evaluation.json", "v2-evaluation.json",
    "evaluation-v0.json", "evaluation-v1.json", "evaluation-v2.json",
    "clean-demo.json", "clean-build-metadata.json", "pytest-preflight.log", "pytest-final.log", "build.log",
    "clean-install.log", "clean-java-build.log", "clean-demo.log",
)
FINAL_DOCS = ("运行与交付说明.md", "实验报告.md", "证据索引.md", "来源与使用范围.md",
              "答辩讲稿与演示顺序.md", "个人报告模板/填写说明.md",
              "个人报告模板/A-平台与运行.md", "个人报告模板/B-决策与模型.md",
              "个人报告模板/C-记录与展示.md")
PRESENTATION_FILES = (
    "presentation/output/掼蛋Agent-课程答辩.pptx",
    "presentation/build/build_deck.mjs", "presentation/build/deck-data.json",
    "presentation/README.md", "presentation/build/assets/table-preview.jpg",
    "presentation/build/final-validation.json",
    "presentation/build/visual-verification.json",
) + tuple(f"presentation/preview/slide-{i:02d}.png" for i in range(1, 13))
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SENSITIVE_PARTS = {".git", ".aws", ".codex", ".agents", ".ssh", ".venv",
                   ".venv-phase3", ".idea", ".vscode", "node_modules"}
SECRET_PATTERNS = (
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bBearer\s+[A-Za-z0-9_.-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)


class PackageError(ValueError):
    """The candidate is incomplete or contains an unsafe input."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, relative: str, *, upstream: bool = False) -> Path:
    """Reject symlinks in the complete path before opening a selected file."""
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative or not pure.parts:
        raise PackageError("Package paths must be relative POSIX paths")
    parts = pure.parts
    if any(part == ".env" or (part.startswith(".env.") and part != ".env.example")
           for part in parts):
        raise PackageError(f"Credential file refused: {relative}")
    if not upstream and any(part in SENSITIVE_PARTS or part in CACHE_DIRS for part in parts):
        raise PackageError(f"Local private/cache path refused: {relative}")
    current = root
    if current.is_symlink():
        raise PackageError("Project root may not be a symlink")
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise PackageError(f"Symlink refused: {relative}")
    if not current.is_relative_to(root):
        raise PackageError("Package path escaped project")
    return current


def scan_content(relative: str, data: bytes) -> None:
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        raise PackageError(f"Potential credential content refused: {relative}")
    if relative == ".env.example":
        assignments = [line.partition("=") for line in data.decode("utf-8").splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
        for name, equals, value in assignments:
            if equals and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                if value.strip():
                    raise PackageError("Credential placeholders in .env.example must be empty")
    # PPTX/Office files are ZIP containers: scan their actual XML rather than
    # trusting compressed bytes to reveal accidental credential content.
    if relative.endswith(".pptx"):
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                if member.filename.endswith((".xml", ".rels")):
                    if any(pattern.search(archive.read(member)) for pattern in SECRET_PATTERNS):
                        raise PackageError(f"Potential credential content refused: {relative}")


def verify_final_run(version: str, run_id: str, config: dict, summary: dict,
                     audit: dict, plan: dict, core: dict[str, bytes]) -> None:
    """A final package must carry the successful audit of these exact bytes."""
    expected = plan.get("configs", {}).get(version, {})
    config_id = expected.get("config_id")
    if (not config_id or config.get("config_id") != config_id
            or config.get("config") != expected.get("config")):
        raise PackageError(f"{version} source differs from its frozen configuration")
    if (summary.get("config_id") != config_id or summary.get("log_complete") is not True
            or summary.get("pending_call_ids") != []
            or summary.get("unconfirmed_submission_ids") != []
            or summary.get("error") is not None):
        raise PackageError(f"{version} summary is incomplete or has unresolved work")
    if (audit.get("passed") is not True or audit.get("accepted_live") is not True
            or audit.get("config_id") != config_id or audit.get("run_id") != run_id):
        raise PackageError(f"{version} audit did not accept this frozen live run")
    hashes = {name: digest(data) for name, data in core.items()}
    if audit.get("files_sha256") != hashes:
        raise PackageError(f"{version} audit is stale or source bytes changed")


def collect(root: Path, *, final: bool = False) -> tuple[dict[str, bytes], dict]:
    files: dict[str, bytes] = {}
    missing_optional: list[str] = []

    def add(relative: str, *, required: bool = True, upstream: bool = False) -> bytes | None:
        path = safe_path(root, relative, upstream=upstream)
        if not path.is_file():
            if required:
                raise PackageError(f"Required delivery input missing: {relative}")
            missing_optional.append(relative)
            return None
        data = path.read_bytes()
        scan_content(relative, data)
        files[relative] = data
        return data

    def tree(relative: str, suffixes: set[str], *, required: bool = True) -> None:
        directory = safe_path(root, relative)
        if not directory.is_dir():
            if required:
                raise PackageError(f"Required delivery directory missing: {relative}")
            missing_optional.append(relative)
            return
        for base, directories, names in os.walk(directory, followlinks=False):
            for name in directories + names:
                path = Path(base) / name
                if path.is_symlink():
                    raise PackageError(f"Symlink refused: {path.relative_to(root).as_posix()}")
                if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
                    raise PackageError("Credential file inside selected delivery tree")
            directories[:] = sorted(name for name in directories if name not in CACHE_DIRS)
            for name in sorted(names):
                path = Path(base) / name
                if path.suffix.lower() in suffixes:
                    add(path.relative_to(root).as_posix())

    for name in BASE_FILES:
        add(name)
    for name in CONFIG_FILES:
        add(f"configs/{name}")
    for name in SCRIPT_FILES:
        add(f"scripts/{name}")
    for name in DESIGN_FILES:
        add(f"docs/{name}")
    tree("src/guandan_agent", {".py", ".html", ".css", ".js"})
    tree("tests", {".py", ".json", ".jsonl", ".md"})
    tree("docs/交付", {".md", ".pptx", ".pdf", ".png", ".jpg", ".svg", ".html", ".json"}, required=final)
    for name in REPLAY_FILES:
        add(f"experiments/phase3/results/probe_platform_run_02/normal/{name}")

    # Snapshot members are selected by the pinned upstream archive, never by a
    # local directory walk. Its original .idea/target files are dependency data.
    archive_name = f"research/archives/{SNAPSHOT_NAME}.tar.gz"
    archive_data = add(archive_name)
    if digest(archive_data) != ARCHIVE_SHA256:
        raise PackageError("Pinned platform archive hash mismatch")
    archive_path = safe_path(root, archive_name)
    prefix = f"OpenGuanDan-Simulator-{UPSTREAM_COMMIT}"
    original_names: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != prefix or ".." in parts:
                raise PackageError("Unexpected pinned platform archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise PackageError("Pinned platform archive has a non-regular member")
            relative = PurePosixPath(*parts[1:]).as_posix()
            if relative in original_names:
                raise PackageError("Duplicate pinned platform archive member")
            original_names.add(relative)
            reader = archive.extractfile(member)
            original = add(f"research/sources/{SNAPSHOT_NAME}/{relative}", upstream=True)
            if reader is None or original != reader.read():
                raise PackageError(f"Pinned platform snapshot differs: {relative}")

    for name in B10_FILES:
        required = name == "frozen-plan.json" or (final and name in {
            "V0-audit.json", "V1-audit.json", "V2-audit.json", "comparison.json",
            "clean-demo.json", "clean-build-metadata.json", "clean-install-verification.json",
        })
        add(f"artifacts/b10/{name}", required=required)
    plan = json.loads(files["artifacts/b10/frozen-plan.json"])
    tree("artifacts/b10/evaluation", {".json"}, required=final)
    for name in PRESENTATION_FILES:
        add(f"artifacts/b10/{name}", required=final)
    run_ids: dict[str, str] = {}
    for version in ("V0", "V1", "V2"):
        for name in ("start.json", "server-ready.json", "provenance.json", "result.json"):
            data = add(f"artifacts/b10/live/{version}/{name}", required=final)
            if name != "result.json" or data is None:
                continue
            result = json.loads(data)
            run_id = result.get("run_id")
            if not isinstance(run_id, str) or str(UUID(run_id)) != run_id:
                if final:
                    raise PackageError(f"{version} has no valid recorded run ID")
                continue
            run_ids[version] = run_id
            for source in ("config.json", "events.jsonl", "summary.json"):
                add(f"runs/{run_id}/{source}")
            summary = json.loads(files[f"runs/{run_id}/summary.json"])
            config = json.loads(files[f"runs/{run_id}/config.json"])
            if summary["run_id"] != run_id or config["run_id"] != run_id:
                raise PackageError("B10 source run ownership mismatch")
            if (root / "runs" / run_id / "summary.pending").exists():
                raise PackageError("B10 source run is not closed")
            if not summary.get("closed"):
                raise PackageError("B10 source run is not closed")
            if final and result.get("status") != "completed":
                raise PackageError(f"{version} did not complete; retain candidate status")
            if final:
                audit = json.loads(files[f"artifacts/b10/{version}-audit.json"])
                verify_final_run(version, run_id, config, summary, audit, plan,
                                 {name: files[f"runs/{run_id}/{name}"]
                                  for name in ("config.json", "events.jsonl", "summary.json")})

    clean_demo_run_id = None
    if "artifacts/b10/clean-demo.json" in files:
        demo = json.loads(files["artifacts/b10/clean-demo.json"])
        clean_demo_run_id = demo.get("run_id")
        if (not isinstance(clean_demo_run_id, str)
                or str(UUID(clean_demo_run_id)) != clean_demo_run_id
                or set(demo.get("core_sha256", {})) != {"config.json", "events.jsonl", "summary.json"}):
            raise PackageError("Clean demo manifest has invalid run ID or core file inventory")
        for name, expected in demo["core_sha256"].items():
            data = add(f"artifacts/b10/clean-demo-evidence/{clean_demo_run_id}/{name}")
            if digest(data) != expected:
                raise PackageError(f"Clean demo evidence hash mismatch: {name}")

    if final:
        for name in FINAL_DOCS:
            if f"docs/交付/{name}" not in files:
                raise PackageError(f"Final delivery document missing: {name}")
    manifest = {
        "schema_version": "1.0", "package": TOP,
        "status": "prepared_final" if final else "candidate",
        "published": False, "model_calls": 0,
        "run_ids": run_ids,
        "clean_demo_run_id": clean_demo_run_id,
        "files": [{"path": name, "sha256": digest(data), "bytes": len(data)}
                  for name, data in sorted(files.items())],
        "source_rules": {
            "selection": "explicit file lists and restricted source/test/delivery trees",
            "upstream_snapshot_exception": "Pinned archive member bytes retained, including original upstream .idea/target files; no local IDE or dependency caches",
            "historical_evidence": "Only six phase-three replay files required by offline tests; historical runs and experimental batches omitted",
            "links": "Historical links in retained design documents may refer to omitted development evidence; delivery evidence index is authoritative for this package",
            "sensitivity_scan": "Credential patterns and empty .env.example placeholders; .env is never read",
        },
        "optional_files_not_present": missing_optional,
    }
    return files, manifest


def build(root: Path = ROOT, *, final: bool = False) -> dict:
    root = root.absolute()
    files, manifest = collect(root, final=final)
    output = safe_path(root, "artifacts/b10/delivery")
    output.mkdir(parents=True, exist_ok=True)
    destination = safe_path(root, "artifacts/b10/delivery/guandan-agent-b10.zip")
    manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".candidate-", suffix=".zip", dir=output)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, data in sorted({**files, "DELIVERY-MANIFEST.json": manifest_data}.items()):
                    info = zipfile.ZipInfo(f"{TOP}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    safe_path(root, "artifacts/b10/delivery/manifest.json").write_bytes(manifest_data)
    result = {"status": manifest["status"], "path": destination.relative_to(root).as_posix(),
              "sha256": digest(destination.read_bytes()), "bytes": destination.stat().st_size,
              "file_count": len(files), "run_ids": manifest["run_ids"], "published": False}
    safe_path(root, "artifacts/b10/delivery/package.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(final=args.final), ensure_ascii=False))
        return 0
    except (PackageError, OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "refused", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
