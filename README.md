# DSpark speculative decoding on the B12X stack (DeepSeek-V4-Flash, RTX PRO 6000 / sm120)

This repo makes **DeepSeek's DSpark block speculative decoding** run on the
**B12X** SM120 inference stack for **DeepSeek-V4-Flash**, on **2× RTX PRO 6000
Blackwell (sm120, TP=2)**. DSpark drafts a block of tokens with a hidden-chain +
rank-256 Markov head and maintains acceptance deeper into the block than native
MTP — so it wins on technical / structured generation.

It is delivered as an **overlay** on the prebuilt B12X vLLM image
(`voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132`,
vLLM `0.11.2.dev279`, CUDA 13.2): 5 new files + small edits to 6 existing files,
plus the exact serve config and the two environment fixes the b12x image needs.

> Status: **working & coherent**, running on b12x's full **compile + cudagraph**
> path (target cudagraphed; DSpark draft eager at ~2.35 ms — not the bottleneck).

---

## TL;DR benchmarks (2× RTX PRO 6000, TP2, single stream, greedy)

| Config | mean accept | pos-0 accept | broad avg tok/s | peak tok/s | context |
|---|---|---|---|---|---|
| jasl FP8 (no spec) | — | — | ~82 | — | 1M |
| **b12x MTP** (default) | 2.2 | 0.787 | ~175 | ~194 | 1M |
| b12x DSpark (initial port) | 2.86 | 0.73 | ~145 | 188 | — |
| **b12x DSpark + RoPE fix** | **3.24** | **0.80** | **168** | **234** | 16k* |

*16k is the current test value; raise `--max-model-len` (see Tuning). DSpark
**beats MTP on technical/structured content** (code 234, science 210) and matches
it on average; MTP is flatter across creative prompts. Acceptance is content
dependent (observed windows 2.6–4.3).

The headline correctness fix (see below) lifted DSpark from 2.86→3.24 mean
acceptance and ~145→168 avg tok/s, with position-0 acceptance now exceeding MTP's
— i.e. DSpark behaving like the trained block drafter it is.

---

## What changed

### New files (copied into `vllm/`, see `overlay/`)
- `models/deepseek_v4/nvidia/dspark.py` — the DSpark draft model (HC stages + Markov/HC/confidence heads). **Contains the key RoPE fix.**
- `models/deepseek_v4/nvidia/dspark_kernels.py` — DSpark Triton kernels (sparse-MLA draft attention, markov-argmax, etc.).
- `models/deepseek_v4/nvidia/ops/fp8_einsum.py` — vendored `deepseek_v4_fp8_einsum` (+ `deepseek_v4_fp8_einsum_config`); b12x lacks it. Returns the **(1,128,128)** recipe for sm120 (the b12x `o_proj.compute_fp8_einsum_recipe` returns the SM100 `(1,1,128)` → `scale_out_blocks=1024` which the einsum rejects).
- `v1/spec_decode/dspark_proposer.py` — `DSparkProposer` (V1 model-runner proposer; manages draft buffers + PIECEWISE draft cudagraph).
- `v1/spec_decode/dspark.py` — DSpark proposer helpers.

### Edits to existing files (see `patches/dspark-b12x.patch`)
- `config/speculative.py` — route `dspark_block_size`-carrying checkpoints to method `dspark` (`DeepSeekV4DSparkModel`); `use_dspark()`.
- `model_executor/models/registry.py` — register `DeepSeekV4DSparkModel`.
- `models/deepseek_v4/__init__.py` — export `DeepSeekV4DSpark`.
- `v1/worker/gpu_model_runner.py` — dispatch `use_dspark()` → `DSparkProposer`; add to the proposer isinstance unions.
- `v1/spec_decode/llm_base_proposer.py` — dspark hooks (noise-token id, aux-reduce, draft-class-skip, guard `compute_logits` introspection).
- `models/deepseek_v4/nvidia/model.py` — **EAGLE3 aux-hidden-state interface** on `DeepseekV4ForCausalLM` (collect HC-reduced hidden at layers 40/41/42 to seed the draft). **The aux collection uses `b12x_mhc_post` (not `mhc_post_tilelang`) so the target still compiles/cudagraphs** — calling tilelang in the aux path triggers torch.compile "function marked as skipped".

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

Served as `deepseek/v4flashdspark` on `:8001` (OpenAI-compatible).

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
- Raise `--max-model-len` (16384 is a test value; KV is sparse-MLA + fp8 so large
  contexts fit — the MTP profile runs 1M). Watch the cudagraph KV reservation;
  bump `--gpu-memory-utilization` if KV doesn't fit.
- `--max-num-seqs 4` is interactive-tuned; raise for batched throughput.

## Provenance / credits
- B12X stack: `lukealonso/b12x` + `local-inference-lab/vllm` (image above).
- DSpark reference: DeepSeek `DeepSeek-V4-Flash-DSpark`; vLLM port lineage from the
  jasl sm120 fork + rafaelcaricio's DSpark integration. This repo ports that onto
  the b12x image and fixes the draft-RoPE acceptance bug.
