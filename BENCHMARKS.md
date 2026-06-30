# Benchmarks — DSpark vs MTP on b12x (2× RTX PRO 6000 Blackwell, sm120, TP2)

Single stream, greedy (`temperature=0`), warmed (tilelang JIT + cudagraphs captured). Harness + prompts in [`benchmarks/`](benchmarks/).

## Methodology
- **Decode tok/s = server-side** `Avg generation throughput` over pure-decode windows (`Avg prompt throughput: 0.0`, `Running: 1`). Speculative decoding emits tokens in *bursts*, so client-side SSE timing **inflates** the rate and prefix-cache-asymmetric differential timing **deflates** it — trust the engine log. Client-side end-to-end runs ~10–15% below server-side (detok + SSE + HTTP + async-step).
- **Acceptance is content-dependent** (code/structured ~4.5–4.8, reasoning ~3.3, creative ~2.2), and acceptance drives throughput — single-number averages hide a wide spread.

## Acceptance (per-position, mean accepted length out of block=5)
| Config | pos0 | pos1 | pos2 | pos3 | pos4 | mean |
|---|---|---|---|---|---|---|
| MTP (num_spec=2) | 0.787 | 0.439 | — | — | — | 2.2 |
| DSpark — initial port | 0.73 | 0.46 | 0.29 | 0.23 | 0.15 | 2.86 |
| **DSpark — after RoPE fix** | **0.80** | **0.61** | **0.37** | **0.27** | **0.19** | **3.24** |

These are *broad-mix* numbers; on code the block fills much deeper (accept_len ~4.5–4.8). DSpark holds acceptance far deeper into the block than MTP (MTP num_spec=5 control collapses: 0.787/0.42/0.12/0.03/0.006). The RoPE fix lifted pos-0 above MTP's — the trained block drafter behaving correctly.

## Decode throughput (server-side), vs the localmaxxing MTP run (181 tok/s)
| workload | context | config | **DSpark tok/s** | accept | MTP tok/s |
|---|---|---|---|---|---|
| code / structured | ~29k | 262k | **~262** (257–278) | 4.5 | ~208 |
| code / structured | ~82k | 262k | ~237 | 4.8 | — |
| reasoning (think=high) | 37k | 262k | ~187 (160–215) | 3.3 | 181 (the bench) |
| code / structured | ~29k | **1M** | **~238** (interactive) | 4.5 | — |
| jasl FP8, no spec decode | — | 1M | ~82 | — | — |

- **DSpark beats MTP by ~26% on identical code prompts** (262 vs 208) and clears the 181 leaderboard run.
- On reasoning the gap narrows (acceptance is content-bound); DSpark ~187 still edges the 181 bench.
- ~2× the no-spec jasl FP8 baseline (~82).

## Context-depth sweep — prefill tok/s, TTFT, decode (client-side, code content)
Decode here is client-side end-to-end (~10–15% under server-side), so read the **trend**, not the absolute.

| context | TTFT | prefill tok/s | decode tok/s (262k cfg) | decode tok/s (1M cfg) |
|---|---|---|---|---|
| ~4k | <1 s | ~5–6k | 234 | 203 |
| ~32k | ~5 s | ~6,100 | 244 | 203 |
| ~128k | ~25 s | ~5,200 | 225 | 191 |
| ~240k / 598k | ~57 s / ~3 min | ~4,300 / ~3,100 | 209 (240k) | 146 (598k) |
| ~942k | ~6.5 min | ~2,400 | — | ~129 |

- **Prefill tok/s** peaks ~6k around 32k (tiny prompts are overhead-bound, very deep prompts attention-bound), declining with depth.
- **TTFT** is prefill-dominated: sub-second at 4k → minutes near 1M.
- **Decode falls with real context depth** in both configs (attention over a deeper KV per step).

## The 1M "reservation tax" (~13%)
At fixed ~29k context, the **262k config decodes ~15–20% faster than 1M** even though the prompt is identical. Cause: the compiled b12x decode kernel (`sparse_mla_decode_forward`) does per-step work **proportional to the allocated KV pool's `num_blocks`** (~2,500 @262k vs ~4,260 @1M) — i.e. the cost of *reserving* enough KV to ever serve a 1M-token request, paid even on short prompts. Proven structural / not fixable from Python:
- `VLLM_DSPARK_INDEXER_PT_CAP` (cudagraph-stable cap of the indexer page-table width) recovers **0%** → the indexer page-table is *not* the bottleneck.
- `--block-size 512` (would halve `num_blocks`) is **unsupported** — b12x hardcodes 256.
- The KV pool can't shrink without losing the 1M ceiling; b12x ships compiled (no source).

Guidance: run **262k** if your contexts stay under ~256k and you want max speed; run **1M** (the default here) when you need the deep-context ceiling (~238 interactive).

## Concurrency (`max_num_seqs > 1`)
Sustained 2 long requests in flight + frequent short finishers (forces continuous-batch condense): accept_len **4.19** (vs 4.52 single-stream), **0** incoherent/errored outputs, no `ValueError`; single-stream speed unchanged. (Greedy output is *not* byte-identical solo-vs-concurrent due to benign batch FP non-determinism — judge by acceptance health + coherence.)

## Stage timing (DSpark proposer, diagnostic)
`draft ≈ 2.27 ms/step`, `prefill_main ≈ 5 ms` (3 draft layers × `store_main_kv`) — both already on the fast path. Per-step time is dominated by the target verify of 1+5 positions plus the `num_blocks`-proportional kernel cost above.

## Reproduce
```bash
./setup.sh && VLLM_API_KEY=sk-... docker compose -f docker-compose.dspark-b12x.yaml up -d
# server-side decode (canonical):  benchmarks/decode_sustained.sh
# prefill/TTFT/decode vs depth:     benchmarks/depth_sweep.py
# per-prompt acceptance:            benchmarks/accept_perprompt.py
# concurrency / long-context:       benchmarks/concurrency_test.py , benchmarks/needle.py
```
