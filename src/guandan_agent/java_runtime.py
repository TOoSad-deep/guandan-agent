"""Java discovery shared by the build and runtime entry points."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


_MAC_JDK21 = Path("/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home")


def jdk_tool(home: Path, name: str) -> Path:
    return home / "bin" / (name + (".exe" if sys.platform == "win32" else ""))


def _configured_home(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if home := os.environ.get("JAVA_HOME"):
        return Path(home).expanduser().resolve()
    if sys.platform == "darwin" and _MAC_JDK21.is_dir():
        return _MAC_JDK21
    return None


def find_jdk_home(explicit: Path | None = None) -> Path:
    """Honor explicit configuration; never silently replace a broken JAVA_HOME."""
    home = _configured_home(explicit)
    if home is None:
        javac = shutil.which("javac")
        if javac is None:
            raise ValueError("JDK 21 not found; set JAVA_HOME or pass --java-home")
        home = Path(javac).resolve().parent.parent
    if not all(jdk_tool(home, name).is_file() for name in ("java", "javac")):
        raise ValueError("JAVA_HOME / --java-home must point to a JDK with bin/java and bin/javac")
    return home


def java_executable() -> str:
    """Leave unavailable Java to the existing bounded runtime fallback."""
    home = _configured_home()
    return str(jdk_tool(home, "java")) if home else (shutil.which("java") or "java")
