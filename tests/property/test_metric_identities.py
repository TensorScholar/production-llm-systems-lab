"""Property tests: TPOT = mean-ITL identity (book Ch. 4.1)."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from llmlab import metrics


@given(
    t_send=st.integers(0, 10**12),
    gaps=st.lists(st.integers(1_000_000, 100_000_000), min_size=1, max_size=20),
    first=st.integers(1_000_000, 1_000_000_000),
)
def test_tpot_equals_mean_itl_single_token_chunks(t_send: int, gaps: list[int], first: int) -> None:
    # One token per chunk: chunk count == n_out, TPOT == mean(ITL).
    times = [t_send + first]
    for g in gaps:
        times.append(times[-1] + g)
    n_out = len(times)
    rt = metrics.request_timing(t_send, times, n_out)
    assert rt.tpot_s is not None
    assert rt.tpot_s == float(np.mean(rt.itl_s)) or (
        abs(rt.tpot_s - float(np.mean(rt.itl_s))) < 1e-6
    )
