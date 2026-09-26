# ADR 0001 — engine instance per repetition

- **Status:** accepted
- **Date:** 2026-09-26

## Context

Engine state (allocator fragmentation, caches, background behaviour) can leak
across sweep points and repetitions.

## Decision

One fresh engine instance per (config, repetition). N order is shuffled
within each instance (seeded). Startup reports and versions are saved per
instance under `results/engines/<instance_id>/`.

## Consequences

- Higher GPU-hour cost; cleaner attribution of variance to repetitions.
