# ADR 0003 — storage split (git vs archive)

- **Status:** accepted
- **Date:** 2026-09-26

## Context

Raw per-chunk event timing is large; summaries and provenance must stay
reviewable in git.

## Decision

- Committed: `run.json`, `requests.parquet`, `server_metrics.parquet`,
  `gpu.parquet`, raw `/metrics` snapshots, `checks.json`, derived CSVs/JSON,
  figures, story.
- Archived externally per release (`.gitignore`d): `results/**/chunks.parquet`,
  bundled by `llmlab archive` into a tarball with SHA-256 + DOI recorded in
  `results/archive_manifest.json`.

## Consequences

- Full reproducibility requires fetching the archived chunks; everything
  needed to check claims is in git.
