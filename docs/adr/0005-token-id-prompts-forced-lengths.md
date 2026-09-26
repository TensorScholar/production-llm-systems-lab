# ADR 0005 — token-ID prompts and forced lengths

- **Status:** accepted
- **Date:** 2026-09-26

## Context

Text prompts tokenize variably, which would blur the fixed-shape (P=2048,
O=1024) capacity question. Forcing exact lengths requires engine support.

## Decision

Prompts are token-ID lists; `max_tokens` and `min_tokens` are both set to O
with `ignore_eos: true` and `temperature: 0` so completions are exactly O
tokens (engine extensions, verified in T13).

## Realism cost

Token-ID prompts are not natural text; results do not speak to output
quality or to length distributions of real traffic. Documented in
`docs/limitations.md`.
