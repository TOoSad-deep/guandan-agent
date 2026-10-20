"""Delivery boundaries are exercised with synthetic files and no network."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts import package_b10_delivery as package


@pytest.fixture
def project(tmp_path, monkeypatch):
    for constant, value in {
        "BASE_FILES": ("README.md", ".env.example"), "CONFIG_FILES": (),
        "SCRIPT_FILES": (), "DESIGN_FILES": (), "REPLAY_FILES": (),
        "B10_FILES": ("frozen-plan.json",),
    }.items():
        monkeypatch.setattr(package, constant, value)
    for name, data in {
        "README.md": b"Project", ".env.example": b"DEEPSEEK_API_KEY=\n",
        "src/guandan_agent/app.py": b"VALUE = 1\n", "tests/fixtures/data.json": b"{}",
        "artifacts/b10/frozen-plan.json": b"{}",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    archive = tmp_path / "research/archives" / f"{package.SNAPSHOT_NAME}.tar.gz"
    archive.parent.mkdir(parents=True)
    upstream = {
        "README.md": b"Original platform",
        ".idea/compiler.xml": b"<upstream/>",
        "target/original.class": b"original upstream artifact",
    }
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in upstream.items():
            item = tarfile.TarInfo(f"OpenGuanDan-Simulator-{package.UPSTREAM_COMMIT}/{name}")
            item.size = len(data)
            stream.addfile(item, io.BytesIO(data))
            path = tmp_path / "research/sources" / package.SNAPSHOT_NAME / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    monkeypatch.setattr(package, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    return tmp_path


def test_allowlist_excludes_local_secrets_caches_other_runs_and_research(project):
    for name in (".env", ".aws/credentials", "src/guandan_agent/__pycache__/app.pyc",
                 "runs/not-in-b10/events.jsonl", "research/private-notes.md"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("LOCAL_MATERIAL")
    files, manifest = package.collect(project)
    assert all(b"LOCAL_MATERIAL" not in data for data in files.values())
    assert "tests/fixtures/data.json" in files
    assert f"research/sources/{package.SNAPSHOT_NAME}/.idea/compiler.xml" in files
    assert f"research/sources/{package.SNAPSHOT_NAME}/target/original.class" in files
    assert manifest["status"] == "candidate" and manifest["published"] is False


@pytest.mark.parametrize("name", ["README.md", "src/guandan_agent/app.py"])
def test_rejects_selected_symlink_files(project, name):
    selected = project / name
    selected.unlink()
    selected.symlink_to(project / ".env.example")
    with pytest.raises(package.PackageError, match="Symlink"):
        package.collect(project)


def test_rejects_symlink_parent_before_reading(project):
    target = project / "external"
    target.mkdir()
    (target / "data.json").write_text("{}")
    (project / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(package.PackageError, match="Symlink"):
        package.safe_path(project, "link/data.json")


@pytest.mark.parametrize("relative", ["../secret", "/tmp/secret", "foo\\bar", ".env", ".env.production", ".aws/config"])
def test_rejects_nonportable_or_sensitive_paths(project, relative):
    with pytest.raises(package.PackageError):
        package.safe_path(project, relative)


def test_credentials_in_selected_content_or_filled_template_are_refused(project):
    (project / "README.md").write_bytes(b"credential=" + b"sk-" + b"x" * 24)
    with pytest.raises(package.PackageError, match="credential content"):
        package.collect(project)
    (project / "README.md").write_text("Project")
    (project / ".env.example").write_text("DEEPSEEK_API_KEY=filled-placeholder\n")
    with pytest.raises(package.PackageError, match="must be empty"):
        package.collect(project)


def test_selected_tree_refuses_env_even_with_disallowed_suffix(project):
    (project / "tests/.env.local").write_text("PRIVATE_DATA")
    with pytest.raises(package.PackageError, match="Credential file"):
        package.collect(project)


def test_modified_upstream_member_cannot_use_dependency_exception(project):
    snapshot = project / "research/sources" / package.SNAPSHOT_NAME
    (snapshot / ".idea/compiler.xml").write_text("local IDE preferences")
    with pytest.raises(package.PackageError, match="snapshot differs"):
        package.collect(project)


def test_zip_is_reproducible_and_every_payload_has_matching_manifest(project):
    first = package.build(project)
    second = package.build(project)
    assert first == second
    archive = project / first["path"]
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(f"{package.TOP}/DELIVERY-MANIFEST.json"))
        assert len(bundle.infolist()) == len(manifest["files"]) + 1
        for item in manifest["files"]:
            assert not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts
            data = bundle.read(f"{package.TOP}/{item['path']}")
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            assert len(data) == item["bytes"]
    assert first["status"] == "candidate" and first["run_ids"] == {}


def test_final_status_requires_complete_inputs(project):
    with pytest.raises(package.PackageError, match="delivery directory missing"):
        package.build(project, final=True)


def test_output_symlink_is_refused(project):
    output = project / "artifacts/b10/delivery"
    output.mkdir()
    (output / "guandan-agent-b10.zip").symlink_to(project / "README.md")
    with pytest.raises(package.PackageError, match="Symlink"):
        package.build(project)
    assert (project / "README.md").read_text() == "Project"


def test_pptx_xml_is_scanned_after_decompression():
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("ppt/slides/slide1.xml", b"<text>" + b"sk-" + b"x" * 24 + b"</text>")
    with pytest.raises(package.PackageError, match="credential content"):
        package.scan_content("docs/交付/slides.pptx", content.getvalue())


def final_inputs():
    run_id = "25fa15e0-8c75-47e2-b2ae-de42bbca50e9"
    config = {"run_id": run_id, "config_id": "frozen-id", "config": {"model": "frozen-model"}}
    summary = {"config_id": "frozen-id", "log_complete": True,
               "pending_call_ids": [], "unconfirmed_submission_ids": [], "error": None}
    plan = {"configs": {"V1": {"config_id": "frozen-id", "config": config["config"]}}}
    core = {"config.json": json.dumps(config).encode(), "events.jsonl": b"test evidence",
            "summary.json": json.dumps(summary).encode()}
    audit = {"passed": True, "accepted_live": True, "run_id": run_id, "config_id": "frozen-id",
             "files_sha256": {name: package.digest(data) for name, data in core.items()}}
    return run_id, config, summary, audit, plan, core


def test_final_run_accepts_only_successful_audit_of_exact_packaged_source():
    package.verify_final_run("V1", *final_inputs())


@pytest.mark.parametrize("target,field,value", [
    ("audit", "passed", False),
    ("audit", "accepted_live", False),
    ("audit", "config_id", "different-frozen-run"),
    ("summary", "log_complete", False),
    ("summary", "pending_call_ids", ["unresolved-call"]),
    ("summary", "unconfirmed_submission_ids", ["unconfirmed-action"]),
])
def test_final_status_rejects_existing_but_unsuccessful_or_unfinished_evidence(target, field, value):
    arguments = list(final_inputs())
    arguments[3 if target == "audit" else 2][field] = value
    with pytest.raises(package.PackageError):
        package.verify_final_run("V1", *arguments)


def test_final_status_rejects_stale_audit_even_if_its_flags_are_true():
    arguments = list(final_inputs())
    arguments[5]["events.jsonl"] = b"changed after audit"
    with pytest.raises(package.PackageError, match="audit is stale"):
        package.verify_final_run("V1", *arguments)


def test_clean_demo_selection_is_limited_to_manifest_core_and_checks_hashes(project, monkeypatch):
    run_id = "25fa15e0-8c75-47e2-b2ae-de42bbca50e9"
    monkeypatch.setattr(package, "B10_FILES", ("frozen-plan.json", "clean-demo.json"))
    directory = project / "artifacts/b10/clean-demo-evidence" / run_id
    directory.mkdir(parents=True)
    core = {name: b"{}" for name in ("config.json", "events.jsonl", "summary.json")}
    for name, data in core.items():
        (directory / name).write_bytes(data)
    (directory / "private-log.txt").write_text("LOCAL_MATERIAL")
    demo = {"run_id": run_id, "core_sha256": {name: package.digest(data) for name, data in core.items()}}
    (project / "artifacts/b10/clean-demo.json").write_text(json.dumps(demo))
    files, manifest = package.collect(project)
    assert manifest["clean_demo_run_id"] == run_id
    assert sum(name.startswith("artifacts/b10/clean-demo-evidence/") for name in files) == 3
    assert not any(name.endswith("private-log.txt") for name in files)
    (directory / "events.jsonl").write_text("changed")
    with pytest.raises(package.PackageError, match="Clean demo evidence hash mismatch"):
        package.collect(project)
