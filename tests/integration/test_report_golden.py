"""Report golden test: figure data + markdown blocks from a fixed mock dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llmlab import report as r

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "report"


def _fixture_derived(tmp: Path) -> Path:
    d = tmp / "derived"
    d.mkdir()
    pd.DataFrame(
        [
            {
                "config": "pool-full",
                "reported_pool_tokens": 80000,
                "predicted_pool_tokens": 80100,
            },
            {
                "config": "pool-reduced",
                "reported_pool_tokens": 70000,
                "predicted_pool_tokens": 69900,
            },
        ]
    ).to_csv(d / "reconciliation.csv", index=False)
    pd.DataFrame(
        [
            {
                "config": "pool-full",
                "N": 1,
                "observed": 2567.5,
                "predicted": 2567.5,
                "predicted_semi": 2567.5,
                "rel_err": 0.0,
                "rel_err_semi": 0.0,
            },
            {
                "config": "pool-full",
                "N": 8,
                "observed": 20540.0,
                "predicted": 20540.0,
                "predicted_semi": 20540.0,
                "rel_err": 0.0,
                "rel_err_semi": 0.0,
            },
        ]
    ).to_csv(d / "occupancy.csv", index=False)
    pd.DataFrame(
        [
            {
                "claim_id": "A1-01",
                "status": "supported",
                "observed": "56128",
                "expected": "56128",
                "residual": "",
                "reason": "golden",
            }
        ]
    ).to_json(d / "claims_eval.json", orient="records", indent=2)
    (d / "decision.json").write_text(
        json.dumps({"cap": 24, "binding": "kv_capacity", "levers": ["raise pool"]}),
        encoding="utf-8",
    )
    return d


def test_figure_data_shapes(tmp_path: Path) -> None:
    d = _fixture_derived(tmp_path)
    data = r.figure_data(d)
    assert set(data["reconciliation"]["config"]) == {"pool-full", "pool-reduced"}
    assert len(data["occupancy"]) == 2
    assert data["timeseries"].empty  # missing file -> empty frame


def test_generated_blocks_golden(tmp_path: Path) -> None:
    d = _fixture_derived(tmp_path)
    blocks = r.generated_blocks(d)
    for name in ("claims_eval", "decision", "occupancy"):
        golden = GOLDEN_DIR / f"{name}.md"
        assert golden.exists(), f"missing golden {golden}"
        assert blocks[name] == golden.read_text(encoding="utf-8")


def test_render_story_roundtrip(tmp_path: Path) -> None:
    d = _fixture_derived(tmp_path)
    template = tmp_path / "story.md"
    template.write_text(
        "# Story\n<!-- BEGIN:GENERATED claims_eval -->\nOLD\n<!-- END:GENERATED -->\n"
        "Prose stays.\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.md"
    r.render_story(template, d, out)
    text = out.read_text(encoding="utf-8")
    assert "OLD" not in text
    assert "Prose stays." in text
    assert "A1-01" in text
