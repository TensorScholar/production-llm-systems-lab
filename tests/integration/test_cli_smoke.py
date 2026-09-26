"""CLI smoke tests (fast commands; demo-mock covered by make target in CI)."""

from __future__ import annotations

from pathlib import Path


def test_kv_smoke() -> None:
    from llmlab.cli import main

    try:
        main(
            [
                "kv",
                "--model",
                "configs/models/qwen2.5-7b-instruct.yaml",
                "--spec-json",
                "tests/fixtures/hf_configs/qwen2.5-7b.json",
                "--pool",
                "4.5GiB",
                "--context",
                "3072",
            ]
        )
    except SystemExit as e:
        assert e.code == 0


def test_a1_smoke(tmp_path: Path) -> None:
    from llmlab.cli import main

    try:
        main(["a1", "experiments/A_kv_cache/A1_calculator/manifest.yaml"])
    except SystemExit as e:
        assert e.code == 0
    assert Path("results/derived/A1/kv_table.csv").exists()


def test_predict_smoke() -> None:
    from llmlab.cli import main

    try:
        main(
            [
                "predict",
                "experiments/A_kv_cache/A4_kv_saturation/manifest.yaml",
                "--spec-json",
                "tests/fixtures/hf_configs/qwen2.5-7b.json",
                "--weight-bytes",
                "15233241088",
            ]
        )
    except SystemExit as e:
        assert e.code == 0
    assert Path("results/derived/A4/predictions_pre.json").exists()


def test_schema_smoke(tmp_path: Path) -> None:
    from llmlab.cli import main

    try:
        main(["schema", "--out", str(tmp_path / "schemas")])
    except SystemExit as e:
        assert e.code == 0
    assert (tmp_path / "schemas" / "RunMetadata.schema.json").exists()


def test_validate_smoke() -> None:
    from llmlab.cli import main

    try:
        main(["validate", "--results", "results/"])
    except SystemExit as e:
        assert e.code == 0


def test_archive_smoke(tmp_path: Path) -> None:
    from llmlab.cli import main

    try:
        main(["archive", "--out", str(tmp_path / "arc")])
    except SystemExit as e:
        assert e.code == 0
    assert (tmp_path / "arc" / "raw-chunks.tar.gz").exists()
    # archive_manifest.json is a T24 release artifact; don't leave test output.
    Path("results/archive_manifest.json").unlink(missing_ok=True)


def test_run_dry_run_smoke(tmp_path: Path) -> None:
    from llmlab.cli import main

    try:
        main(
            [
                "run",
                "experiments/B_serving/B0_harness_accuracy/manifest.yaml",
                "--dry-run",
                "--allow-dirty",
                "--results",
                str(tmp_path / "build"),
            ]
        )
    except SystemExit as e:
        assert e.code == 0


def test_analyze_report_smoke() -> None:
    from llmlab.cli import main

    try:
        main(
            [
                "analyze",
                "experiments/A_kv_cache/A4_kv_saturation/manifest.yaml",
                "--spec-json",
                "tests/fixtures/hf_configs/qwen2.5-7b.json",
            ]
        )
    except SystemExit as e:
        assert e.code == 0
    try:
        main(["report", "experiments/A_kv_cache/A4_kv_saturation/manifest.yaml"])
    except SystemExit as e:
        assert e.code == 0
