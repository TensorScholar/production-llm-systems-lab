"""Figures and story rendering: derived data → SVG + generated markdown blocks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["svg.hashsalt"] = "llmlab-v01"
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_SPECS: list[tuple[str, str]] = [
    ("a3_reconciliation", "Predicted vs reported KV capacity; overhead bars"),
    ("a4_occupancy", "KV tokens vs N, predicted N·g line, capacity line, saturation band"),
    ("a4_throughput_ttft", "Output tokens/s vs N; TTFT p50 vs N with prediction"),
    ("a4_tpot", "TPOT vs N with bandwidth floor, decode KV-read model, calibrated model"),
    ("a4_queue", "Running and waiting requests vs N"),
    ("a4_timeseries", "One saturated run: KV usage, running, waiting, preemptions over time"),
]


def figure_data(derived_dir: Path) -> dict[str, pd.DataFrame]:
    """Load derived tables into per-figure datasets (golden-tested)."""
    derived_dir = Path(derived_dir)
    data: dict[str, pd.DataFrame] = {}
    for name in (
        "reconciliation",
        "occupancy",
        "summary",
        "serving_model",
        "littles_law",
        "saturation",
        "timeseries",
    ):
        p = derived_dir / f"{name}.csv"
        data[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return data


def make_figures(derived_dir: Path, fig_dir: Path, meta: dict[str, Any]) -> list[Path]:
    """Render the six SVG figures (6.5in, grayscale-safe) + provenance files."""
    derived_dir = Path(derived_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = figure_data(derived_dir)
    made: list[Path] = []
    for fig_name, _desc in FIGURE_SPECS:
        path = fig_dir / f"{fig_name}.svg"
        _render(fig_name, data, path, meta)
        prov = {
            "figure": fig_name,
            "inputs": _input_hashes(derived_dir),
            "run_ids": meta.get("run_ids", []),
            "claim_ids": meta.get("claim_ids", []),
            "evidence": meta.get("evidence", "MOCK"),
            "generator": "llmlab.report.make_figures",
        }
        (fig_dir / f"{fig_name}.svg.provenance.json").write_text(
            json.dumps(prov, indent=2), encoding="utf-8"
        )
        made.append(path)
    return made


def _input_hashes(derived_dir: Path) -> dict[str, str]:
    from llmlab import provenance as _p

    out = {}
    for p in sorted(derived_dir.glob("*.csv")) + sorted(derived_dir.glob("*.json")):
        if p.name.endswith(".provenance.json"):
            continue
        out[p.name] = _p.sha256_file(p)
    return out


def _render(fig_name: str, data: dict[str, pd.DataFrame], path: Path, meta: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    try:
        if fig_name == "a3_reconciliation":
            df = data.get("reconciliation", pd.DataFrame())
            if not df.empty and "config" in df.columns:
                x = range(len(df))
                ax.bar(
                    [i - 0.2 for i in x],
                    df.get("reported_pool_tokens", [0] * len(df)),
                    0.4,
                    label="reported",
                    color="0.3",
                )
                ax.bar(
                    [i + 0.2 for i in x],
                    df.get("predicted_pool_tokens", [0] * len(df)),
                    0.4,
                    label="predicted",
                    color="0.7",
                )
                ax.set_xticks(list(x), list(df["config"]))
                ax.set_ylabel("KV tokens")
            else:
                ax.text(0.5, 0.5, "A3 data required", ha="center")
        elif fig_name == "a4_occupancy":
            df = data.get("occupancy", pd.DataFrame())
            if not df.empty:
                for _cfg, sub in df.groupby("config"):
                    ordered = sub.sort_values("N")
                    ax.plot(ordered["N"], ordered["observed"], "o-", label=f"{_cfg} observed")
                    ax.plot(ordered["N"], ordered["predicted"], "--", label=f"{_cfg} N·g")
                ax.set_xlabel("concurrency N")
                ax.set_ylabel("KV tokens")
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "A4 data required", ha="center")
        elif fig_name == "a4_throughput_ttft":
            df = data.get("summary", pd.DataFrame())
            if not df.empty:
                ax.text(0.5, 0.5, f"A4 summary rows: {len(df)}", ha="center")
            else:
                ax.text(0.5, 0.5, "A4 data required", ha="center")
        elif fig_name == "a4_tpot":
            df = data.get("serving_model", pd.DataFrame())
            if not df.empty:
                for _cfg, sub in df.groupby("config"):
                    ordered = sub.sort_values("N")
                    ax.plot(ordered["N"], ordered["tpot_obs"], "o-", label=f"{_cfg} obs")
                    ax.plot(ordered["N"], ordered["tpot_pred"], "x--", label=f"{_cfg} pred")
                ax.set_xlabel("N")
                ax.set_ylabel("TPOT (s)")
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "A4 data required", ha="center")
        elif fig_name == "a4_queue":
            df = data.get("summary", pd.DataFrame())
            if not df.empty:
                ax.text(0.5, 0.5, f"queue rows: {len(df)}", ha="center")
            else:
                ax.text(0.5, 0.5, "A4 data required", ha="center")
        elif fig_name == "a4_timeseries":
            df = data.get("timeseries", pd.DataFrame())
            if not df.empty and "t_s" in df.columns:
                for col, style in (("kv_usage", "-"), ("running", "--"), ("waiting", ":")):
                    if col in df.columns:
                        ax.plot(df["t_s"], df[col], style, label=col)
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "saturated-run timeseries required", ha="center")
        ax.set_title("")
        fig.tight_layout()
        fig.savefig(path, format="svg")
        # Determinism: strip the render timestamp matplotlib embeds.
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\s*<dc:date>.*?</dc:date>\n?", "", text)
        path.write_text(text, encoding="utf-8")
    finally:
        plt.close(fig)


def generated_blocks(derived_dir: Path) -> dict[str, str]:
    """Build generated markdown blocks from derived data (golden-tested)."""
    derived_dir = Path(derived_dir)
    blocks: dict[str, str] = {}
    for name in ("claims_eval", "decision", "predictions", "b0"):
        p = derived_dir / f"{name}.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                lines = ["| claim | status | observed |", "|---|---|---|"]
                for row in obj:
                    cid = row.get("claim_id", "")
                    st = row.get("status", "")
                    obs = row.get("observed", "")
                    lines.append(f"| {cid} | {st} | {obs} |")
                blocks[name] = "\n".join(lines) + "\n"
            elif isinstance(obj, dict):
                blocks[name] = "```json\n" + json.dumps(obj, indent=2) + "\n```\n"
            else:
                blocks[name] = str(obj)
        else:
            blocks[name] = "_DATA REQUIRED — run the analysis pipeline first._\n"
    # CSV-backed blocks.
    for name in ("kv_table", "occupancy", "saturation"):
        p = derived_dir / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            blocks[name] = df.to_markdown(index=False) + "\n"
        else:
            blocks[name] = "_DATA REQUIRED — run the analysis pipeline first._\n"
    return blocks


def render_story(template: Path, derived_dir: Path, out: Path) -> None:
    """Replace GENERATED blocks; leave hand-written prose untouched."""
    text = Path(template).read_text(encoding="utf-8")
    blocks = generated_blocks(Path(derived_dir))
    pattern = re.compile(r"<!-- BEGIN:GENERATED (\w+) -->(.*?)<!-- END:GENERATED -->", re.DOTALL)

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        body = blocks.get(name, "_DATA REQUIRED._\n")
        return f"<!-- BEGIN:GENERATED {name} -->\n{body}<!-- END:GENERATED -->"

    out.write_text(pattern.sub(_sub, text), encoding="utf-8")
