# Acknowledgments & sources

This work stands on a lot of other people's. Thank you to:

## The B12X SM120 stack (the fast base this runs on)
- **[lukealonso/b12x](https://github.com/lukealonso)** — the custom SM120 kernels (FP8 GEMM, MoE, sparse-MLA, sparse indexer, MLA-SM120-unified, PCIe one-shot all-reduce) that make DeepSeek-V4-Flash fast on RTX PRO 6000.
- **[local-inference-lab/vllm](https://github.com/local-inference-lab)** — the vLLM fork the b12x image is built from, and **[local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro)** (`models/ds4-flash-v4.md`) — the serve recipe we started from.
- The prebuilt image `voipmonitor/vllm:chthonic-consecration-…-b12x…-pr20-cu132` (vLLM 0.11.2.dev279, CUDA 13.2), built against **DeepGEMM PR#324** and **FlashInfer PR#3395** for genuine SM120 support.

## DSpark itself
- **DeepSeek** — the **DeepSeek-V4-Flash** model and the **DeepSeek-V4-Flash-DSpark** checkpoint + reference implementation (`inference_model.py` / `inference_kernel.py`), which defines the block speculative drafter (hidden-chain + rank-256 Markov + confidence head) and which we audited line-by-line to find the draft-RoPE fix.
- **rafaelcaricio** — the original DSpark→vLLM integration our port lineage descends from.
- The **jasl** SM120 vLLM fork — earlier DSpark-on-sm120 porting groundwork.

## The concurrency fix
- **[drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)** — identified the two bugs that force single-request serving (batch-row KV-slot keying; the uniform-input constraint under chunked prefill) and the fix approach (stable req-id→slot map + ragged `query_start_loc` path). Our `max_num_seqs>1` support is a port of that diagnosis onto the b12x stack.

## Benchmarking
- **[localmaxxing.com](https://www.localmaxxing.com)** — the public benchmark leaderboard and the DeepSeek-V4-Flash run (181 tok/s, MTP) that set the target we measured against.

## Foundation
- The **[vLLM](https://github.com/vllm-project/vllm)** project and the **EAGLE3** speculative-decoding interface DSpark's draft seeding maps onto.
