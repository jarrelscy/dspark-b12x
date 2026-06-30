# DSpark speculative decoding on the B12X stack (DeepSeek-V4-Flash, RTX PRO 6000 / sm120)

This repo makes **DeepSeek's DSpark block speculative decoding** run on the
**B12X** SM120 inference stack for **DeepSeek-V4-Flash**, on **2× RTX PRO 6000
Blackwell (sm120, TP=2)**. DSpark drafts a block of tokens with a hidden-chain +
rank-256 Markov head and maintains acceptance deeper into the block than native
MTP — so it wins on technical / structured generation.

It is delivered as an **overlay** on the prebuilt B12X vLLM image
(`voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132`,
vLLM `0.11.2.dev279`, CUDA 13.2): 5 new files + small edits to 8 existing files,
plus the exact serve config and the two environment fixes the b12x image needs.

> Status: **working & coherent**, running on b12x's full **compile + cudagraph**
> path (target cudagraphed; DSpark draft eager at ~2.35 ms — not the bottleneck).

---

## TL;DR benchmarks (2× RTX PRO 6000, TP2, single stream, greedy)

Decode tok/s is **server-side** (`Avg generation throughput`); **content-dependent** — acceptance drives throughput (code/structured ~4.5, reasoning ~3.3, creative ~2.2).

| Config (code workload) | accept len | decode tok/s | context |
|---|---|---|---|
| jasl FP8 — no spec decode | — | ~82 | 1M |
| b12x MTP (num_spec=2) | 2.6 | ~208 | 262k |
| **b12x DSpark (num_spec=5)** | **4.5** | **~262** | 262k |
| b12x DSpark @ 1M | 4.5 | ~238 (interactive) | 1M |
| localmaxxing MTP n=1 run (the target) | — | 181 | 262k |

**DSpark beats MTP by ~26% on code** (262 vs 208) and clears the 181 tok/s leaderboard run; at 1M context it holds ~238 (a flat ~13% `num_blocks` reservation tax vs 262k). Prefill ~6k tok/s @ ~30k; TTFT scales with depth (sub-second @4k → minutes near 1M). See [BENCHMARKS.md](BENCHMARKS.md) and [`benchmarks/`](benchmarks/) for the full curves, the reasoning/creative numbers, and the harness + prompts.

The enabling fix was a draft-RoPE off-by-one (below): acceptance 2.86 → 3.24 broad (4.5+ on code), pos-0 now > MTP's — DSpark behaving like the trained block drafter it is.

---

## What changed

### New files (copied into `vllm/`, see `overlay/`)
- `models/deepseek_v4/nvidia/dspark.py` — the DSpark draft model (HC stages + Markov/HC/confidence heads). **Contains the key RoPE fix.**
- `models/deepseek_v4/nvidia/dspark_kernels.py` — DSpark Triton kernels (sparse-MLA draft attention, markov-argmax, etc.).
- `models/deepseek_v4/nvidia/ops/fp8_einsum.py` — vendored `deepseek_v4_fp8_einsum` (+ `deepseek_v4_fp8_einsum_config`); b12x lacks it. Returns the **(1,128,128)** recipe for sm120 (the b12x `o_proj.compute_fp8_einsum_recipe` returns the SM100 `(1,1,128)` → `scale_out_blocks=1024` which the einsum rejects).
- `v1/spec_decode/dspark_proposer.py` — `DSparkProposer` (V1 model-runner proposer; manages draft buffers + PIECEWISE draft cudagraph). Also carries the **concurrency** fix (req-id→KV-slot map + ragged `query_start_loc` path) — see the Update section.
- `v1/spec_decode/dspark.py` — DSpark proposer helpers.

### Edits to existing files (the complete change set is the full files in [`overlay/vllm/`](overlay/vllm/))
- `config/speculative.py` — route `dspark_block_size`-carrying checkpoints to method `dspark` (`DeepSeekV4DSparkModel`); `use_dspark()`.
- `model_executor/models/registry.py` — register `DeepSeekV4DSparkModel`.
- `models/deepseek_v4/__init__.py` — export `DeepSeekV4DSpark`.
- `v1/worker/gpu_model_runner.py` — dispatch `use_dspark()` → `DSparkProposer`; proposer isinstance unions; thread `req_ids` for the concurrency slot-map; `INDEXER_PT_CAP` over-cap eager-fallback.
- `v1/spec_decode/llm_base_proposer.py` — dspark hooks (noise-token id, aux-reduce, draft-class-skip, guard `compute_logits` introspection).
- `models/deepseek_v4/nvidia/model.py` — **EAGLE3 aux-hidden-state interface** on `DeepseekV4ForCausalLM` (collect HC-reduced hidden at layers 40/41/42 to seed the draft). **The aux collection uses `b12x_mhc_post` (not `mhc_post_tilelang`) so the target still compiles/cudagraphs** — calling tilelang in the aux path triggers torch.compile "function marked as skipped".
- `v1/attention/backends/mla/indexer.py` — `VLLM_DSPARK_BT_COPY_TRIM` (default on: trim the per-step block-table copy to live context) + cudagraph-stable `VLLM_DSPARK_INDEXER_PT_CAP` (high-context experiment; see Update).
- `model_executor/layers/sparse_attn_indexer.py` — consume the capped page-table width (paired with `INDEXER_PT_CAP`).

### THE acceptance fix (`dspark.py`, draft RoPE off-by-one)
`draft_input_ids[:,0]` is the **bonus** token (the just-sampled `next_token_ids`),
which logically sits at `last_context_pos + 1`. The original port positioned the
whole draft block at `last_context_pos + [0..4]` — one step too early — detuning
**every** draft↔context dot product (RoPE phase error present even at position 0).
Fix:

```python
# dspark.py — draft block RoPE positions
offsets = torch.arange(1, block_size + 1, ...)         # was arange(0, block_size)
draft_positions = (main_positions[:, -1:] + offsets).reshape(-1)
```

Result: per-position acceptance `0.73/0.46/0.29/0.23/0.15` → `0.80/0.61/0.37/0.27/0.19`
(mean 2.86 → 3.24); position-0 0.73 → 0.80 (now > MTP's 0.787). Found by auditing
against the DeepSeek reference (`inference_model.py`, draft positions
`start_pos+1 … start_pos+block`).

---

## Setup & run

Requires: 2× sm120 GPUs (RTX PRO 6000), Docker w/ NVIDIA runtime, the
`deepseek-ai/DeepSeek-V4-Flash-DSpark` checkpoint cached under `$HF_HOME/hub`.

```bash
./setup.sh                              # pull image, extract vLLM + libcudart, apply overlay
export HF_HOME=/data/huggingface        # where the DSpark checkpoint is cached
export VLLM_API_KEY=sk-...              # your key
docker compose -f docker-compose.dspark-b12x.yaml up -d
# wait for "Application startup complete" (first run JITs tilelang + captures graphs)
curl localhost:8001/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

Served as `deepseek/v4flash` on `:8001` (OpenAI-compatible).

### Two image gotchas the compose handles for you
1. **NCCL "unhandled system error"** — the image bakes `NCCL_GRAPH_FILE=` (empty);
   NCCL opens it as an XML topology and dies. The launch `unset`s it.
2. **tilelang libcudart stub** — `flashinfer.comm`'s `find_loaded_library("libcudart")`
   substring-matches tilelang's `libcudart_stub.so` (missing `cudaDeviceReset`).
   We bind-mount the real `libcudart.so.13` over the stub **and** `LD_PRELOAD` it.

---

## Why this config (and what doesn't work)

- **`VLLM_USE_BREAKABLE_CUDAGRAPH=0` (compile path):** target compiled+cudagraphed,
  draft eager. **Coherent + fast.** `=1` (breakable) captures the draft's tilelang
  and corrupts it → garbage. `--enforce-eager` is coherent but ~9 tok/s.
- **`VLLM_USE_V2_MODEL_RUNNER=0`:** the DSpark proposer is implemented for the V1
  runner; b12x's V2 runner only supports eagle3/dflash. b12x's FP8-GEMM/MoE/
  sparse-MLA kernels still engage under V1.
- **`num_speculative_tokens=5`** is required (== `dspark_block_size`); 3 crashes.
- The draft (per the stage timer) is ~**2.35 ms** — already PIECEWISE-cudagraphed,
  **not** the bottleneck. The per-step cost is the target verify of 1+5 positions.

## Tuning
- `--max-model-len` defaults to 1M here; lower it (e.g. 262k) for ~15-20% faster decode if you don't need the full ceiling. KV is sparse-MLA + fp8 so large
  contexts fit — the MTP profile runs 1M). Watch the cudagraph KV reservation;
  bump `--gpu-memory-utilization` if KV doesn't fit.
- `--max-num-seqs 4` is interactive-tuned; raise for batched throughput.

## Provenance / credits
- B12X stack: `lukealonso/b12x` + `local-inference-lab/vllm` (image above).
- DSpark reference: DeepSeek `DeepSeek-V4-Flash-DSpark`; vLLM port lineage from the
  jasl sm120 fork + rafaelcaricio's DSpark integration. This repo ports that onto
  the b12x image and fixes the draft-RoPE acceptance bug.

---

## Update (2026-06-29): concurrency support, high-context fix, and the benchmark result

### Benchmarks vs the localmaxxing DeepSeek-V4-Flash run (MTP, 181 tok/s)
Server-side decode throughput (`Avg generation throughput`, pure-decode windows, single stream, greedy, 262k-ctx config). **Content-dependent** — acceptance drives throughput:

| workload | context | DSpark decode tok/s | accept_len |
|---|---|---|---|
| **code / structured** | ~29k | **~262** (peak 278) | 4.5 |
| code / structured | ~82k | ~237 | 4.8 |
| reasoning (thinking=high) | 37k | ~187 | 3.3 |
| localmaxxing bench (MTP n=1) | 37.7k | 181 | — |

**Direct head-to-head, identical code prompts at 262k:** DSpark (num_spec=5) **~262 tok/s @ accept 4.5** vs MTP (num_spec=2) **~208 tok/s @ accept 2.6** — DSpark **+26%**. DSpark's higher acceptance more than pays for its heavier 6-position verify on structured content; on high-entropy reasoning the gap narrows (acceptance is content-bound).

> Measure server-side throughput, not the client. Speculative decoding emits tokens in **bursts**, so client-side SSE timing inflates decode rate (burst buffering) and prefix-cache asymmetry deflates differential timing. The engine's `Avg generation throughput` log is the truth.

### Concurrency (`max_num_seqs > 1`) — now supported
The original DSpark serving path forced single-request processing (two bugs, diagnosed by **[drowzeys/Keys-Concurrency-Patch](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)**): (1) the persistent draft `main_kv_cache` was keyed by **batch-row position**, so when continuous batching condensed the running set after a request finished, a request drafted against another's KV → acceptance collapse; (2) the context-prep assumed **uniform per-request rows**, raising `ValueError` under chunked prefill. This port fixes both:
- **Stable req-id→slot map** (always-on; identity fast-path keeps single-stream byte-identical) so the draft KV window follows the request across condense.
- **Ragged `query_start_loc` context path** (gated `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`) for mixed prefill+decode steps; permuted-slot steps run the draft eager so the captured fixed-shape cudagraph is never replayed with a remapped index.

Validated: under sustained load (2 long requests always in flight + constant short finishers forcing condense), acceptance held at **4.19** (vs 4.52 single-stream), **zero incoherent/errored outputs**, no `ValueError`; single-stream speed unchanged.

### High-context: 1M decode (`BT_COPY_TRIM`, `INDEXER_PT_CAP`)
At large `--max-model-len`, per-step block-table structures are sized by `cdiv(max_model_len, block_size)` regardless of live context. `VLLM_DSPARK_BT_COPY_TRIM=1` (default on) trims the expansion **copy** to live context (needle-validated to 120k).

**1M interactive decode = ~238 tok/s** (single stream, code; clears the 230 target) at full 1.1M-token KV. The 262k config is ~262 — the ~10–13% gap is **structural**. b12x is CuTe-DSL **source** (forkable, not a blob), and a full read of `attention/mla/kernel.py` found **no fixable per-step op**: no TMA descriptor over the pool, no pool-sized memset/loop, grid/loops topk-bounded, JIT kernel byte-identical across configs. The cost is GPU **L2/TLB/DRAM locality** of gathering topk-scattered KV blocks across the larger allocation that a 1M ceiling requires (vLLM mandates the pool hold ≥1 full max-len request, so you can't shrink it via `num-gpu-blocks-override`). We proved this isn't fixable from the kernel:
- `VLLM_DSPARK_INDEXER_PT_CAP=<tokens>` (default `0`/off) — a **cudagraph-stable** cap that pins the indexer page-table to a fixed `cdiv(CAP, block)` width (eager fallback for contexts > CAP). Capture succeeds and a needle retrieves correctly below and above CAP — **but it is throughput-neutral** (capping to the 262k-equivalent width recovered 0%), confirming the indexer page-table is *not* the bottleneck. Left in, off by default; it also forces >CAP requests eager, so don't enable it for a true-1M workload.
- `--block-size 512` (to halve `num_blocks`) is **unsupported** — the b12x kernels hardcode 256 (`setStorage ... out of bounds`).
- The KV pool can't shrink without losing the 1M ceiling (vLLM enforces it). So 238 is the 1M interactive ceiling on this stack; the only levers are *outside* b12x — a lower `--max-model-len`, or a vLLM block-allocator change to keep a sequence's active blocks compacted (so the gather working set tracks live context, not pool size).

Note: a *saturated back-to-back* throughput pattern reads lower (~227) because the next request's chunked prefill overlaps and steals decode windows — that's a serving-throughput artifact, not the interactive decode rate.

### Env flags added this round
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` — enable the ragged concurrency path.
- `VLLM_DSPARK_BT_COPY_TRIM` (default `1`) — trim per-step block-table copy to live context.
- `VLLM_DSPARK_INDEXER_PT_CAP` (default `0`/off) — cudagraph-stable indexer page-table cap (tokens); throughput-neutral here, see above. Don't enable for true-1M workloads (forces >CAP requests eager).
- `VLLM_DSPARK_BF16_O_PROJ` (default `0`) — bf16 draft o-proj (matches reference; measured throughput-neutral).

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for the people and projects this builds on.
