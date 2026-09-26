"""Generate results/derived/B0/harness_accuracy.csv (MOCK evidence).

In-process mock at N in {1, 8, 32, 64}; exact prompt-hash matching of client
receipts to server emissions on the same clock. Run: python generate_b0.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent


def _setup() -> None:
    sys.path.insert(0, str(REPO / "src"))
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("uvloop: yes")
    except ImportError:
        print("uvloop: no (stdlib loop)")


async def _run_n(n: int) -> dict[str, object]:
    from aiohttp.test_utils import TestClient, TestServer

    from llmlab.loadgen import run_closed_loop
    from llmlab.metrics import harness_accuracy
    from llmlab.schemas import ClientConfig, WorkloadSpec
    from llmlab.testing import mock_server as ms
    from llmlab.testing.mock_server import MockConfig, create_app
    from llmlab.workload import fixed_shape_request, request_stream, safe_token_ids

    mock = MockConfig(capacity_tokens=262144, ttft_ms=50.0, itl_ms=20.0, seed=0)
    app = create_app(mock)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        base = str(client.make_url("/"))[:-1]
        spec = WorkloadSpec(kind="fixed_shape", prompt_tokens=32, output_tokens=16, seed=7)
        safe = safe_token_ids({"added_tokens": []}, 1000)
        prompts = {
            f"req-{i:08d}": list(fixed_shape_request(spec, safe, i).prompt_token_ids)
            for i in range(6000)
        }
        cfg = ClientConfig(
            warmup_s=1.0, stagger_s=0.5, window_min_s=4.0, window_max_s=20.0,
            min_completed=60, request_timeout_s=60.0, drain_timeout_s=60.0,
        )
        res = await run_closed_loop(
            request_stream(spec, safe), base, "mock-model", n, cfg, f"B0-{n}"
        )
        ok_ids = {r.request_id for r in res.requests if r.in_window and r.status == "ok"}
        records = [
            {"request_id": r.request_id, "t_send_ns": r.t_send_ns}
            for r in res.requests
            if r.request_id in ok_ids
        ]
        chunks = [c for c in res.chunks if c["request_id"] in ok_ids]
        acc = harness_accuracy(records, chunks, prompts, app[ms._STATE_KEY].emitted, 16)
        row: dict[str, object] = {"N": n, **acc}
        row["loop_lag_p99_ms"] = res.loop_lag_p99_ms
        row["client_cpu_max"] = res.client_cpu_max
        # Scope flag: claim B0-01 is scoped to the reference setup. This table
        # was produced on a non-reference machine -> in_scope_reference=False,
        # so claim evaluation must report inconclusive (not refuted).
        row["in_scope_reference"] = False
        return row
    finally:
        await client.close()


async def _main() -> list[dict[str, object]]:
    return [await _run_n(n) for n in (1, 8, 32, 64)]


def main() -> None:
    _setup()
    rows = asyncio.run(_main())
    out = REPO / "results/derived/B0/harness_accuracy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    prov = {
        "inputs": ["src/llmlab/loadgen.py", "src/llmlab/testing/mock_server.py"],
        "generator": "experiments/B_serving/B0_harness_accuracy/generate_b0.py",
        "environment": platform.platform(),
        "evidence": "MOCK",
        "note": "in-process mock; same perf_counter clock; prompt-hash matching",
    }
    (out.parent / "harness_accuracy.csv.provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
