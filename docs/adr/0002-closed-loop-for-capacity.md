# ADR 0002 — closed-loop load for capacity questions

- **Status:** accepted
- **Date:** 2026-09-26

## Context

The v0.1 question is how much KV cache concurrent in-flight requests fill,
which depends on how many requests are in flight — the variable to control
is concurrency, not arrival rate.

## Decision

v0.1 uses closed-loop fixed-concurrency load (exactly N in flight; a new
request starts when one finishes). Fixed arrival-rate (open-loop) testing
moves to v0.2.

Staggered starts prevent lock-step (identically shaped requests started
together would finish together and swing occupancy empty↔full).
