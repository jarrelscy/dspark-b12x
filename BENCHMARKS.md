# Benchmarks — DSpark vs MTP on b12x (2× RTX PRO 6000, TP2, single stream, greedy)

All numbers: vLLM `0.11.2.dev279` b12x image, DeepSeek-V4-Flash(-DSpark) FP8,
`--temperature 0`, warmed (tilelang JIT + cudagraphs captured), `max_tokens≈256`.
Throughput is decode tok/s. **Acceptance is content-dependent** (technical/code
high, creative low), so single-number averages hide a wide spread — see per-prompt.

## Acceptance (per-position, mean accepted length out of block=5)

| Config | pos0 | pos1 | pos2 | pos3 | pos4 | mean |
|---|---|---|---|---|---|---|
| MTP (num_spec=2) | 0.787 | 0.439 | — | — | — | 2.2 |
| DSpark — initial port | 0.73 | 0.46 | 0.29 | 0.23 | 0.15 | 2.86 |
| **DSpark — RoPE fix** | **0.80** | **0.61** | **0.37** | **0.27** | **0.19** | **3.24** |

DSpark holds acceptance **much deeper** into the block than MTP (MTP collapses
after pos1; cf. MTP num_spec=5 control: 0.787/0.42/0.12/0.03/0.006). After the
RoPE fix, DSpark's pos-0 (0.80) also exceeds MTP's (0.787) — the trained block
drafter behaving correctly. Observed acceptance windows ranged 2.6–4.3 by content.

## Throughput — identical 8-prompt broad benchmark

essay / photosynthesis / dragon-story / Hamlet / neural-net / water-cycle / python-sort / quantum

| Config | per-prompt tok/s | AVG | min | max |
|---|---|---|---|---|
| **DSpark + RoPE fix** (16k ctx) | 140 210 132 154 159 181 **234** 154 | **168** | 129 | **234** |
| MTP (1M ctx) | 98 144 140 155 150 152 176 155 | 146 | 98 | 176 |
| MTP (200k ctx, 3-prompt) | 172 / 193 / 157 | ~175 | — | 194 |

Notes:
- **DSpark wins the head-to-head** at these settings and is strongest on
  technical/structured content (python-sort 234, photosynthesis 210, water-cycle 181).
- Context matters for MTP: at 1M ctx MTP averages 146 (larger KV/cudagraph
  overhead); at 200k it was ~175. DSpark here is at 16k (raise via `--max-model-len`;
  expect some slowdown at very large contexts, same as MTP).
- Baseline jasl FP8 (no spec decode) on the same box: ~82 tok/s. Both b12x spec
  paths roughly **2×** that.

## Stage timing (DSpark proposer, diagnostic)
`draft ≈ 2.35 ms/step` (already PIECEWISE-cudagraphed) — **not** the bottleneck.
Per-step time is dominated by the target verify of 1+5 positions. The RoPE fix
raised acceptance, which cuts the number of verifies → the throughput gain.

## Reproduce
```bash
./setup.sh && docker compose -f docker-compose.dspark-b12x.yaml up -d
python3 bench.py                  # see repo; edit MODEL=deepseek/v4flash[dspark]
```
