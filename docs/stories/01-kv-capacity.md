# The KV-capacity story

Question, theory, predictions, measurement, comparison, decision. Figures
carry evidence labels; refuted claims stay visible.

## 1. Question

How many long-context requests can one 24 GB GPU serve at once, and what
limits that number?

## 2. Theory

KV-cache accounting and the calibrated serving model
(`docs/methodology/kv-accounting.md`).

## 3. Predictions (committed before runs)

<!-- BEGIN:GENERATED predictions -->
_DATA REQUIRED — run the analysis pipeline first._
<!-- END:GENERATED -->

## 4. Implementation

Formulas as tested code; vLLM external; closed-loop load generator.

## 5. Measurement

<!-- BEGIN:GENERATED occupancy -->
_DATA REQUIRED — run the analysis pipeline first._
<!-- END:GENERATED -->

<!-- BEGIN:GENERATED saturation -->
_DATA REQUIRED — run the analysis pipeline first._
<!-- END:GENERATED -->

## 6. Comparison (claims table)

<!-- BEGIN:GENERATED claims_eval -->
_DATA REQUIRED — run the analysis pipeline first._
<!-- END:GENERATED -->

## 7. Engineering decision

<!-- BEGIN:GENERATED decision -->
_DATA REQUIRED — run the analysis pipeline first._
<!-- END:GENERATED -->

## 8. Limitations and threats to validity

See `docs/limitations.md`. Single GPU/model/shape; closed-loop only.

## 9. Reproduce

See `docs/reproduce.md` and the run card.
