"""JDK discovery on Windows and Unix, including paths containing spaces."""
from pathlib import Path

import pytest

from guandan_agent import java_runtime as java


@pytest.mark.parametrize("platform,suffix", [("win32", ".exe"), ("linux", ""), ("darwin", "")])
def test_explicit_home_and_environment_take_precedence(tmp_path, monkeypatch, platform, suffix):
    monkeypatch.setattr(java.sys, "platform", platform)
    explicit, configured = tmp_path / "Explicit JDK", tmp_path / "Configured JDK"
    for home in (explicit, configured):
        (home / "bin").mkdir(parents=True)
        for name in ("java", "javac"):
            (home / "bin" / (name + suffix)).touch()
    monkeypatch.setenv("JAVA_HOME", str(configured))
    assert java.find_jdk_home(explicit) == explicit.resolve()
    assert java.find_jdk_home() == configured.resolve()
    assert java.java_executable() == str(configured.resolve() / "bin" / ("java" + suffix))


def test_path_discovery_and_missing_jdk(tmp_path, monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(java.sys, "platform", "linux")
    home = tmp_path / "Path JDK"
    (home / "bin").mkdir(parents=True)
    for name in ("java", "javac"):
        (home / "bin" / name).touch()
    monkeypatch.setattr(java.shutil, "which", lambda name: str(home / "bin" / name))
    assert java.find_jdk_home() == home.resolve()
    assert java.java_executable() == str(home / "bin/java")
    monkeypatch.setattr(java.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="JDK 21 not found"):
        java.find_jdk_home()
    assert java.java_executable() == "java"


def test_invalid_explicit_home_is_not_replaced_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "other"))
    with pytest.raises(ValueError, match="must point to a JDK"):
        java.find_jdk_home(tmp_path / "missing")


def test_macos_existing_jdk_remains_a_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(java.sys, "platform", "darwin")
    monkeypatch.setattr(java, "_MAC_JDK21", tmp_path)
    assert java.java_executable() == str(tmp_path / "bin/java")
