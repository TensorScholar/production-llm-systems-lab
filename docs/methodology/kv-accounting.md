# KV accounting

Formulas implemented in `src/llmlab/calc/kv.py` and `calc/roofline.py`
(book Ch. 3.2–3.3, 3.4, 4.1.1, 4.3, 4.4, 4.4.1).

## Per-token footprint

m_kv = 2·L·n_kv·d_h·b bytes/token (K and V, all layers, dtype bytes).

## Paged accounting (block size q)

- blocks(T) = ceil(T/q); slack(T) = (q − T mod q) mod q.
- Pool: num_blocks = floor(pool_bytes/(q·m_kv)); tokens = num_blocks·q.
- Paged max concurrency = floor(num_blocks / ceil(T_req/q)).

## Mean decode footprint ḡ

ḡ = (1/(O−1))·Σ_{k=1}^{O−1} q·ceil((P+k)/q), summed exactly.
For P=2048, O=1024, q=16 the exact rational is 2,626,560/1023 ≈ 2567.507;
the spec quotes 2567.5 (1-decimal). Peak g_max = q·ceil((P+O)/q) = 3072.

Time-weighted (semi-empirical) variant with measured TTFT/TPOT:
[TTFT·(q·ceil(P/q)/2) + (O−1)·TPOT·ḡ] / [TTFT + (O−1)·TPOT].

## Saturation band

(C/g_max, C/ḡ): below C/g_max requests fit even at full length; above C/ḡ
average demand exceeds capacity.

## Unit-mixing regression (Ch. 4 case study)

Free slots = 196,608 − 140,480 = 56,128 (before reserve); 31,552 after a
3 GiB reserve. The erroneous ~53k figure comes from mixing decimal GB with a
binary-ish divisor (6.85e9/128e3); labelled units (`units.py`) prevent it.

## Roofline bounds

- Decode-streamed weights W_dec = param_count·b − vocab·hidden·b (untied only;
  the embedding table is gathered, not streamed).
- TPOT floor = (W_dec + ḡ·m_kv)/peak_BW; TTFT floor = 2·N·P/peak FLOPS.

## Calibrated closed-loop serving model (valid only below saturation)

t_dec(N) = (W_dec + N·ḡ·m_kv)/BW_eff; n_c(N) = ceil(P/(B_tok − (N−1)));
λ_o = λ·(N−1)/N; f_o = λ_o·t_pf; S = (1−f_o)/t_dec + λ_o·n_c; TPOT = 1/S;
TTFT = t_pf + t_dec; L = TTFT + (O−1)·TPOT; λ = N/L (solved by bisection).

Assumptions: prefill cost independent of chunking; decode tokens riding in
mixed batches add no time; no queueing; uniform request shape. Breaks above
saturation, with mixed shapes, or with prefix sharing (which lowers ḡ).
