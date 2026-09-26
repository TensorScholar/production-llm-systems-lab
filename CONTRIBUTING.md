# Contributing

## Workflow

- One task (T00–T24) per PR with a conventional commit message.
- Do not squash history: commit history is part of the experiment provenance model.
- Never force-push, rewrite published history, or delete branches.

## Rules for AI-assisted changes

1. Implement only what the task specifies; ask before adding dependencies, abstractions, or files not listed.
2. Bytes are `int`, time is `int` nanoseconds; display units are always labelled.
3. Library code has no side effects outside `runner`, `engines`, `loadgen`, `telemetry`; use `logging`, never `print`, in library code.
4. Every formula has a docstring giving the formula and its book section.
5. Tests come with the code in the same task; `ruff`, `mypy`, and `pytest` must pass.
6. Never write mock output under `results/`. Never edit derived files by hand.
7. Never fabricate GPU measurements. Unmeasured results stay `DATA REQUIRED`.

## Independent reproduction runs

- Follow `docs/reproduce.md` and the run card.
- Record your own provenance (commit, engine version, hardware) — do not overwrite existing `results/runs/`.
- Report contradictions with the book via the errata path (`claims/ledger.yaml` + story limitations section).
