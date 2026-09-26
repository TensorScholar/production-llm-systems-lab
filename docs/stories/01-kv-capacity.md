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
| claim | status | observed |
|---|---|---|
| A1-01 | supported | m_kv rows={'llama_m_kv': 131072, 'qwen_m_kv': 57344}, free_slots=56128 |
| A1-02 | supported | 57344 |
| A3-01 | data_required |  |
| A3-02 | data_required |  |
| A4-01 | data_required |  |
| A4-02 | data_required |  |
| A4-03 | data_required |  |
| A4-04 | data_required |  |
| A4-05 | data_required |  |
| B0-01 | inconclusive | ttft p99 0.00045858818, itl p99 0.0019181820799999 |
<!-- END:GENERATED -->

## 7. Engineering decision

<!-- BEGIN:GENERATED decision -->
```json
{
  "cap": null,
  "binding": "unknown",
  "levers": [
    "collect more data"
  ],
  "predicted_knee": "",
  "measured_knee": "",
  "narrative": ""
}
```
<!-- END:GENERATED -->

## 8. Limitations and threats to validity

See `docs/limitations.md`. Single GPU/model/shape; closed-loop only.

## 9. Reproduce

See `docs/reproduce.md` and the run card.
