"""Code/environment capture, run/instance IDs, hashing, prereg lookup."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from llmlab.schemas import CodeInfo, EnvInfo

logger = logging.getLogger(__name__)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip()
        return None
    except Exception:
        return None


def capture_code(cwd: Path | None = None) -> CodeInfo:
    """Capture git commit + dirty flag. Formula: dirty = status --porcelain non-empty."""
    from llmlab import __version__

    root = cwd or Path.cwd()
    commit = _git(["rev-parse", "HEAD"], root) or "unknown"
    status = _git(["status", "--porcelain"], root)
    dirty = bool(status) if status is not None else False
    return CodeInfo(commit=commit, dirty=dirty, llmlab_version=__version__)


def capture_env() -> EnvInfo:
    """Capture Python/platform/CPU identity + pip-freeze hash."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        freeze = out.stdout if out.returncode == 0 else ""
    except Exception:
        freeze = ""
    digest = hashlib.sha256(freeze.encode()).hexdigest()
    return EnvInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        pip_freeze_sha256=digest,
    )


def new_run_id(exp: str, config: str, rep: int, n: int, when: str, rng: random.Random) -> str:
    """Run ID: {exp}-{config}-r{rep}-n{N}-{UTC}-{4hex}."""
    return f"{exp}-{config}-r{rep}-n{n}-{when}-{rng.randrange(16**4):04x}"


def new_instance_id(exp: str, config: str, rep: int, when: str, rng: random.Random) -> str:
    """Engine instance ID: {exp}-{config}-r{rep}-{UTC}-{4hex}."""
    return f"{exp}-{config}-r{rep}-{when}-{rng.randrange(16**4):04x}"


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def prereg_commit(prereg_path: Path, repo: Path | None = None) -> tuple[str, str]:
    """Last commit touching the PREREG file: (sha, iso-utc).

    Formula: git log -1 --format=%H|%cI -- <path>.
    """
    root = repo or Path.cwd()
    out = _git(["log", "-1", "--format=%H|%cI", "--", str(prereg_path)], root)
    if not out:
        return ("unknown", "")
    parts = out.split("|", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")
