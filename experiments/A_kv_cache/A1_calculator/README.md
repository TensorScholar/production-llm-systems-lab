# A1 — calculator

Method: `llmlab a1` evaluates the KV accounting identities over registered
models plus the Llama-3-8B-shaped reference config, writing
`results/derived/A1/kv_table.csv`. CI diffs it byte-identically against
`tests/golden/a1_kv_table.csv`.

Evidence class: ANALYTICAL / IDENTITY (not a GPU result).

Deviations: none.
