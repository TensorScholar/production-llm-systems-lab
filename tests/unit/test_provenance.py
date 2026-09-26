"""Unit tests for llmlab.provenance (temporary git repos)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from llmlab import provenance


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.t"], cwd=cwd, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=cwd, check=True, capture_output=True)


def test_clean_and_dirty(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    (tmp_path / "f.txt").write_text("a")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "x")
    info = provenance.capture_code(tmp_path)
    assert info.dirty is False
    assert len(info.commit) == 40
    (tmp_path / "f.txt").write_text("b")
    assert provenance.capture_code(tmp_path).dirty is True


def test_hashes_stable(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert provenance.sha256_file(p) == provenance.sha256_file(p)
    assert provenance.sha256_json({"b": 1, "a": 2}) == provenance.sha256_json({"a": 2, "b": 1})


def test_prereg_lookup(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    pre = tmp_path / "PREREG.md"
    pre.write_text("hypotheses")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "prereg")
    sha, when = provenance.prereg_commit(pre, tmp_path)
    assert len(sha) == 40
    assert when != ""


def test_run_id_format() -> None:
    import random

    rng = random.Random(0)
    rid = provenance.new_run_id("A4", "pool-full", 0, 8, "20260926T0000Z", rng)
    assert rid.startswith("A4-pool-full-r0-n8-20260926T0000Z-")
    iid = provenance.new_instance_id("A4", "pool-full", 0, "20260926T0000Z", rng)
    assert iid.startswith("A4-pool-full-r0-20260926T0000Z-")
