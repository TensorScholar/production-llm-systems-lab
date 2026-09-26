# ADR 0004 — Rule of two

- **Status:** accepted
- **Date:** 2026-09-26

## Context

Premature abstraction (plugin systems, registries, shared engine interfaces)
adds untested surface and obscures the v0.1 evidence chain.

## Decision

No abstract base class, plugin system, or registry until a second
implementation exists. A module becomes a package when it passes ~400 lines
or gets a second implementation.

In v0.1: vLLM is the only engine, so there is no shared engine interface;
`engines/vllm.py` also launches the mock server via `engine.name == "mock"`
with no separate abstraction.

## Consequences

- Simpler, reviewable v0.1 code.
- Second-engine work must introduce the abstraction then, not now.
