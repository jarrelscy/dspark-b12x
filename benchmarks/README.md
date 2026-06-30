# Benchmark harness & prompts

The exact scripts (and the prompts embedded in them) used to produce the numbers in the top-level
README / BENCHMARKS. All read `VLLM_API_KEY` and `BENCH_MODEL` (default `deepseek/v4flash`) from the
environment — **no keys are hardcoded**. Single-stream, greedy (`temperature=0`) unless noted.

```bash
export VLLM_API_KEY=sk-...          # your serving key
export BENCH_MODEL=deepseek/v4flash # served-model-name
```

## Measurement methodology (read this first)
- **Decode tok/s: trust the server, not the client.** Speculative decoding emits tokens in *bursts*, so
  client-side SSE timing inflates the rate (burst buffering) and prefix-cache asymmetry deflates
  differential timing. The reliable decode number is the engine's `Avg generation throughput` over
  **pure-decode windows** (`Avg prompt throughput: 0.0`, `Running: 1`) — that's what `decode_sustained.sh`
  reads from the container logs.
- **TTFT / prefill tok/s** *are* reliable client-side: TTFT = first-token arrival; prefill tok/s =
  `prompt_tokens / TTFT`. `depth_sweep.py` measures decode as `(completion-1)/(stream_end - first_token)`
  using the true stream-end wall time (immune to the burst artifact).
- **Acceptance** is content-dependent: code/structured ~4.5–4.8, reasoning ~3.3, creative ~2.2. Read it
  from `/metrics` (`spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total + 1`).
- **Saturated back-to-back load reads ~10% low** vs interactive, because the next request's chunked
  prefill overlaps and steals decode windows — measure interactive with a single long request.

## Scripts

| script | what it measures | prompts used |
|---|---|---|
| `decode_sustained.sh` | server-side decode tok/s + acceptance (the canonical throughput number) | code: a synthetic Python module (`process_module_i`/`transform_i` ×N) + "write a complete production LRU+TTL sharded async KV store". `NFUNC` sets context depth, `REQS` the load. |
| `depth_sweep.py` | prefill tok/s, TTFT, decode tok/s across **context depths** | same code context+output as above, swept over `FUNCS` (≈4k/32k/128k/512k/… actual prompt tokens). |
| `accept_perprompt.py` | per-prompt acceptance length + per-position rates + tok/s | 8 broad prompts: Roman-empire essay, photosynthesis, dragon story, Hamlet, neural-net, water cycle, Python sort, quantum entanglement (mix of code/technical/creative to show content-dependence). |
| `concurrency_test.py` | acceptance + coherence under `max_num_seqs>1` | 2 long code requests always in flight + frequent short "Say hi." finishers (forces continuous-batch condense). |
| `needle.py` | long-context correctness (needle-in-haystack) | filler "quarterly report" sentences with a secret code embedded mid-context, at 4k/30k/120k depths. |

## Run
```bash
# canonical decode throughput (server-side), ~29k code context, sustained
BENCH_CONTAINER=dspark-b12x NFUNC=400 REQS=6 ./decode_sustained.sh

# prefill / TTFT / decode vs context depth
FUNCS=55,440,1750,7000 python3 depth_sweep.py

# per-prompt acceptance across content types
python3 accept_perprompt.py

# concurrency correctness (needs VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 + max-num-seqs>1)
python3 concurrency_test.py

# long-context correctness
python3 needle.py
```

`decode_sustained.sh` reads the engine log of the serving container — set `BENCH_CONTAINER` to your
container name (the compose service here is `dspark-b12x`).

## Headline results (2× RTX PRO 6000, TP2)
- Code @ ~29k, 262k config: **~262 tok/s decode** (server-side), accept_len 4.5 — vs MTP ~208 (+26%), vs the 181 MTP leaderboard run.
- 1M config: ~238 tok/s interactive (structural ~13% `num_blocks` reservation tax vs 262k).
- Prefill ~6k tok/s @ ~30k; TTFT scales with depth (sub-second @4k → minutes near 1M).
