# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from typing import Any

import torch
from typing_extensions import override

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dspark import (
    DSparkDiagnostics,
    confidence_threshold_prefix_length,
    hardware_aware_prefix_schedule,
    make_dspark_warmup_draft_token_ids,
    score_prefix_lengths,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DSparkProposer(SpecDecodeBaseProposer):
    """DSpark proposer for DeepSeek V4 Flash DSpark.

    DSpark's draft model owns a small internal sliding-window cache over
    target-layer features. It does not allocate draft KV blocks through vLLM's
    normal speculative-decoding KV cache path.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        hf_config = self.draft_model_config.hf_config
        self.target_hidden_size = hf_config.hidden_size * len(
            hf_config.dspark_target_layer_ids
        )
        self.noise_token_id = int(hf_config.dspark_noise_token_id)
        self._prefilled = False
        self._runner = runner
        self._draft_graph_runner: CUDAGraphWrapper | None = None
        self._draft_graph_batch_size = 0
        self._draft_input_ids_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self._draft_hidden_buffer = torch.zeros(
            self.max_batch_size,
            self.target_hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._draft_positions_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.long,
            device=device,
        )
        # Bug 1: stable per-request KV-slot mapping. The persistent draft KV
        # window lives at a fixed row in the model's main_kv_cache; vLLM-v1
        # continuous batching condenses the running set when a request finishes,
        # so the batch-ROW position is not stable across steps. We map req-id ->
        # slot so reads/writes always hit the right request's window.
        self._req_id_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self.max_batch_size))
        # Persistent buffer for the per-row -> slot permutation handed to the
        # draft read. Initialized to identity so capture/warmup/single-stream
        # stay byte-identical. `_active_slot_index` is None on the identity
        # fast-path (no index_select; original code path) and otherwise a view
        # of this buffer.
        self._draft_slot_index_buffer = torch.arange(
            self.max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self._active_slot_index: torch.Tensor | None = None
        self.diagnostics = DSparkDiagnostics(
            max_spec_tokens=self.num_speculative_tokens
        )
        self.confidence_threshold = self._read_confidence_threshold()
        self.confidence_scheduler = self._read_confidence_scheduler(
            self.confidence_threshold
        )
        self._forced_draft_length = self._read_forced_draft_length(
            self.num_speculative_tokens
        )
        self._sps_curve = self._read_sps_curve()
        self._hardware_scheduler_early_stop = (
            self._read_hardware_scheduler_early_stop()
        )
        self._last_draft_lengths: list[int] | None = None
        self._last_draft_probs: torch.Tensor | None = None
        self._export_draft_probs = self._read_export_draft_probs()
        self._collect_confidence_diagnostics = (
            self._read_collect_confidence_diagnostics()
        )
        self._collect_position0_diagnostics = (
            self._read_position0_diagnostics()
        )
        self._gpu_rejected_context_mask = (
            self._read_gpu_rejected_context_mask()
        )
        self._stage_timing = self._read_stage_timing()
        self._stage_timing_log_every = self._read_stage_timing_log_every()
        self._stage_timing_count = 0
        self._stage_timing_totals_ms: dict[str, float] = {}
        self._last_confidence: torch.Tensor | None = None
        if self.confidence_threshold > 0.0:
            logger.info(
                "DSpark confidence-scheduled verification enabled with "
                "threshold %.4f.",
                self.confidence_threshold,
            )
        if self.confidence_scheduler == "hardware":
            logger.info(
                "DSpark hardware-aware confidence scheduler enabled with "
                "early_stop=%s and SPS curve=%s.",
                self._hardware_scheduler_early_stop,
                self._sps_curve or "constant",
            )
        if self._forced_draft_length is not None:
            logger.info(
                "DSpark forced draft verification length enabled for "
                "profiling: %d.",
                self._forced_draft_length,
            )
        if self._export_draft_probs:
            logger.info(
                "DSpark draft probability export enabled for quality profiling. "
                "This adds a draft-logit softmax on greedy requests."
            )
        if self._collect_confidence_diagnostics:
            logger.info(
                "DSpark confidence diagnostics enabled. This copies confidence "
                "scores to CPU on every draft step."
            )
        if self._collect_position0_diagnostics:
            logger.info(
                "DSpark position-0 diagnostics enabled. The confidence head "
                "runs on every draft step; the runner logs first-token "
                "target-argmax agreement."
            )
        if self._gpu_rejected_context_mask:
            logger.info(
                "DSpark GPU rejected-context mask enabled. Rejected target "
                "suffix rows are masked during draft main-KV cache update "
                "without synchronizing rejection counts to CPU."
            )
        if self._stage_timing:
            logger.info(
                "DSpark stage timing enabled. This synchronizes CUDA work and "
                "is intended for diagnostics, not speed-gate benchmarks."
            )
        if not self._needs_draft_logits() and not self._needs_confidence():
            logger.info(
                "DSpark fast draft-output mode enabled: confidence head and "
                "returned draft logits are skipped on the hot path."
            )

    @staticmethod
    def _read_confidence_threshold() -> float:
        raw = os.getenv("VLLM_DSPARK_CONFIDENCE_THRESHOLD", "0.0")
        try:
            threshold = float(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_THRESHOLD must be a float in [0, 1], "
                f"got {raw!r}"
            ) from exc
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_THRESHOLD must be in [0, 1], "
                f"got {threshold}"
            )
        return threshold

    @staticmethod
    def _read_confidence_scheduler(confidence_threshold: float) -> str:
        raw = os.getenv("VLLM_DSPARK_CONFIDENCE_SCHEDULER", "auto")
        scheduler = raw.strip().lower()
        if scheduler in {"", "auto"}:
            return "threshold" if confidence_threshold > 0.0 else "off"
        if scheduler not in {"off", "threshold", "hardware"}:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_SCHEDULER must be one of "
                "'off', 'threshold', 'hardware', or 'auto', "
                f"got {raw!r}"
            )
        return scheduler

    @staticmethod
    def _read_forced_draft_length(max_draft_length: int) -> int | None:
        raw = os.getenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "").strip()
        if raw == "":
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_FORCE_DRAFT_LENGTH must be an integer in "
                f"[0, {max_draft_length}] or empty, got {raw!r}"
            ) from exc
        if value < 0 or value > max_draft_length:
            raise ValueError(
                "VLLM_DSPARK_FORCE_DRAFT_LENGTH must be in "
                f"[0, {max_draft_length}], got {value}"
            )
        return value

    @staticmethod
    def _read_sps_curve() -> tuple[tuple[int, float], ...]:
        raw = os.getenv("VLLM_DSPARK_SPS_CURVE", "").strip()
        if not raw:
            return ()

        entries: dict[int, float] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                batch_tokens_raw, rate_raw = item.split(":", 1)
                batch_tokens = int(batch_tokens_raw)
                rate = float(rate_raw)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_DSPARK_SPS_CURVE must be a comma-separated table "
                    "of '<batch_tokens>:<steps_per_second>' entries, "
                    f"got {raw!r}"
                ) from exc
            if batch_tokens <= 0:
                raise ValueError(
                    "VLLM_DSPARK_SPS_CURVE batch-token keys must be positive, "
                    f"got {batch_tokens}"
                )
            if rate < 0.0:
                raise ValueError(
                    "VLLM_DSPARK_SPS_CURVE rates must be non-negative, "
                    f"got {rate}"
                )
            entries[batch_tokens] = rate
        return tuple(sorted(entries.items()))

    @staticmethod
    def _read_hardware_scheduler_early_stop() -> bool:
        raw = os.getenv("VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP", "1")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _steps_per_second(self, batch_tokens: int) -> float:
        curve = getattr(self, "_sps_curve", ())
        if not curve:
            return 1.0

        batch_tokens = int(batch_tokens)
        selected_rate = curve[0][1]
        for profiled_tokens, rate in curve:
            if batch_tokens < profiled_tokens:
                break
            selected_rate = rate
        return selected_rate

    def _effective_confidence_scheduler(self) -> str:
        scheduler = getattr(self, "confidence_scheduler", None)
        if scheduler is not None:
            return scheduler
        threshold = getattr(self, "confidence_threshold", 0.0)
        return "threshold" if threshold > 0.0 else "off"

    @staticmethod
    def _read_export_draft_probs() -> bool:
        raw = os.getenv("VLLM_DSPARK_EXPORT_DRAFT_PROBS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_collect_confidence_diagnostics() -> bool:
        raw = os.getenv("VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_position0_diagnostics() -> bool:
        raw = os.getenv("VLLM_DSPARK_POSITION0_DIAGNOSTICS", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_gpu_rejected_context_mask() -> bool:
        raw = os.getenv("VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_stage_timing() -> bool:
        raw = os.getenv("VLLM_DSPARK_STAGE_TIMING", "0")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_stage_timing_log_every() -> int:
        raw = os.getenv("VLLM_DSPARK_STAGE_TIMING_LOG_EVERY", "20")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSPARK_STAGE_TIMING_LOG_EVERY must be an integer, "
                f"got {raw!r}"
            ) from exc
        return max(1, value)

    def _record_stage_timing(self, name: str, elapsed_ms: float) -> None:
        self._stage_timing_totals_ms[name] = (
            self._stage_timing_totals_ms.get(name, 0.0) + float(elapsed_ms)
        )

    def _timed_stage(self, name: str, fn):
        if not getattr(self, "_stage_timing", False):
            return fn()
        if self.device.type != "cuda":
            started = time.perf_counter()
            result = fn()
            self._record_stage_timing(name, (time.perf_counter() - started) * 1000.0)
            return result

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        self._record_stage_timing(name, start.elapsed_time(end))
        return result

    def _maybe_log_stage_timing(self) -> None:
        if not getattr(self, "_stage_timing", False):
            return
        self._stage_timing_count += 1
        if self._stage_timing_count % self._stage_timing_log_every != 0:
            return

        names = (
            "context_prepare",
            "prefill_main",
            "graph_prepare",
            "draft",
            "postprocess",
            "total",
        )
        parts = []
        for name in names:
            total_ms = self._stage_timing_totals_ms.get(name, 0.0)
            parts.append(f"{name}={total_ms / self._stage_timing_count:.3f}ms")
        logger.info(
            "DSpark stage timing avg over %d proposals: %s",
            self._stage_timing_count,
            ", ".join(parts),
        )

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        del kv_cache_config, kernel_block_sizes
        self.block_size = 1

    @override
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del is_graph_capturing, slot_mappings
        batch_size = max(1, min(int(num_tokens), self.max_batch_size))
        (
            cudagraph_runtime_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        ) = self._determine_graph_batch(batch_size, use_cudagraphs=use_cudagraphs)
        self._prepare_draft_buffers(
            input_ids=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            hidden_states=torch.zeros(
                batch_size,
                self.target_hidden_size,
                dtype=self.dtype,
                device=self.device,
            ),
            positions=torch.arange(batch_size, dtype=torch.long, device=self.device),
            padded_batch_size=padded_batch_size,
        )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=padded_batch_size * self.num_speculative_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            self._run_draft_for_current_context()

    @override
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            dspark_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            dspark_cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_dispatcher.initialize_cudagraph_keys(dspark_cudagraph_mode)
        if dspark_cudagraph_mode != CUDAGraphMode.NONE and self.device.type == "cuda":
            self._draft_graph_runner = CUDAGraphWrapper(
                self._run_draft_from_buffers,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
            )

    def _determine_graph_batch(
        self,
        batch_size: int,
        *,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None, BatchDescriptor]:
        cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            batch_size,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        padded_batch_size = batch_descriptor.num_tokens
        num_tokens_across_dp = None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=batch_size,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=padded_batch_size,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for DSpark"
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                padded_batch_size = int(num_tokens_across_dp[dp_rank].item())
                cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                    padded_batch_size,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                assert batch_descriptor.num_tokens == padded_batch_size
                num_tokens_across_dp[dp_rank] = padded_batch_size
        return (
            cudagraph_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        )

    def _prepare_draft_buffers(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        padded_batch_size: int,
    ) -> None:
        batch_size = input_ids.shape[0]
        self._draft_graph_batch_size = padded_batch_size
        self._draft_input_ids_buffer[:batch_size].copy_(input_ids.to(torch.long))
        self._draft_hidden_buffer[:batch_size].copy_(hidden_states.to(self.dtype))
        self._draft_positions_buffer[:batch_size].copy_(positions.to(torch.long))
        if padded_batch_size > batch_size:
            pad_slice = slice(batch_size, padded_batch_size)
            self._draft_input_ids_buffer[pad_slice].fill_(self.noise_token_id)
            self._draft_hidden_buffer[pad_slice].zero_()
            self._draft_positions_buffer[pad_slice].zero_()

    def _run_draft_from_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = self._draft_graph_batch_size
        # _active_slot_index is None on the identity fast-path (single stream /
        # no condense), which keeps the original read path byte-identical and
        # cudagraph-replay-safe. When slots are permuted it is a buffer view of
        # length batch_size and the draft is run eager (see propose()).
        return self.model.draft_with_confidence(
            self._draft_input_ids_buffer[:batch_size],
            self._draft_hidden_buffer[:batch_size],
            self._draft_positions_buffer[:batch_size],
            return_logits=self._needs_draft_logits(),
            return_confidence=self._needs_confidence(),
            store_main_kv=False,
            slot_index=self._active_slot_index,
        )

    def _resolve_slots(
        self,
        req_ids: list[str] | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Map this step's requests (in batch-row order) to stable KV slots.

        Returns None when the resolved permutation is the identity
        [0, 1, ..., batch_size - 1] so callers take the original byte-identical
        fast-path. Otherwise returns a length-`batch_size` long tensor whose
        i-th entry is the persistent main_kv_cache row for batch row i.
        """
        if req_ids is None:
            return None
        if len(req_ids) != batch_size:
            # Defensive: if the runner's row-ordered req-id list does not match
            # the tensor batch dim, fall back to the original behavior rather
            # than risk mismapping KV.
            return None

        live = set(req_ids)
        # Reclaim slots from requests that are no longer running.
        for stale in [r for r in self._req_id_to_slot if r not in live]:
            self._free_slots.append(self._req_id_to_slot.pop(stale))
        # Lowest-first reuse keeps the identity permutation stable for the
        # common single-stream / append-only case.
        self._free_slots = sorted(set(self._free_slots))

        slots: list[int] = []
        for req_id in req_ids:
            slot = self._req_id_to_slot.get(req_id)
            if slot is None:
                slot = self._free_slots.pop(0)
                self._req_id_to_slot[req_id] = slot
            slots.append(slot)

        if slots == list(range(batch_size)):
            return None
        self._draft_slot_index_buffer[:batch_size].copy_(
            torch.tensor(slots, dtype=torch.long, device=self.device)
        )
        return self._draft_slot_index_buffer[:batch_size]

    def _run_draft_for_current_context(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A permuted slot index must bypass the captured graph (recorded with the
        # identity index_select and fixed padded shapes); run the draft eager.
        if self._draft_graph_runner is not None and self._active_slot_index is None:
            return self._draft_graph_runner()
        return self._run_draft_from_buffers()

    def _batch_size(self, next_token_ids: torch.Tensor) -> int:
        return int(next_token_ids.shape[0])

    def _view_by_request(
        self,
        values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if values.shape[0] % batch_size != 0:
            raise ValueError(
                "DSpark currently requires uniform flattened per-request inputs; "
                f"got {values.shape[0]} rows for batch_size={batch_size}."
            )
        seq_len = values.shape[0] // batch_size
        return values.view(batch_size, seq_len, values.shape[-1])

    def _positions_by_request(
        self,
        positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if positions.ndim != 1:
            positions = positions.reshape(-1)
        if positions.shape[0] % batch_size != 0:
            raise ValueError(
                "DSpark currently requires uniform flattened positions; "
                f"got {positions.shape[0]} rows for batch_size={batch_size}."
            )
        return positions.view(batch_size, positions.shape[0] // batch_size)

    def _query_start_loc_cpu(
        self,
        common_attn_metadata: CommonAttentionMetadata | None,
    ) -> torch.Tensor | None:
        if common_attn_metadata is None:
            return None
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        if qsl_cpu is None:
            qsl = common_attn_metadata.query_start_loc
            if qsl is None:
                return None
            qsl_cpu = qsl.detach().cpu()
        return qsl_cpu

    def _maybe_ragged_query_start_loc(
        self,
        common_attn_metadata: CommonAttentionMetadata | None,
        batch_size: int,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return query_start_loc (on device) iff per-request rows are ragged.

        Raggedness (unequal per-request segment lengths) only happens with
        chunked prefill mixing prefill + decode in one step. For a uniform
        batch we return None so the caller keeps the rectangular fast-path.
        """
        qsl_cpu = self._query_start_loc_cpu(common_attn_metadata)
        if qsl_cpu is None:
            return None
        if qsl_cpu.numel() != batch_size + 1:
            return None
        starts = qsl_cpu.to(torch.long).tolist()
        # The flat hidden tensor must cover exactly the segmented rows.
        if starts[-1] != target_hidden_states.shape[0]:
            return None
        lengths = [starts[i + 1] - starts[i] for i in range(batch_size)]
        if len(set(lengths)) <= 1:
            # Uniform -> rectangular fast-path (byte-identical).
            return None
        return qsl_cpu.to(device=self.device, dtype=torch.long)

    def _prepare_context_ragged(
        self,
        target_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        batch_size: int,
        rejected_for_gpu_mask: torch.Tensor | None,
    ):
        """Ragged context-prep using query_start_loc segment offsets.

        Returns flat hidden/positions (passed straight to the ragged
        prefill_main) plus per-request last-row hidden/positions for the draft.
        No rectangular [B, seq, H] view is taken, so per-request row counts may
        differ. Runs eager.
        """
        flat_hidden = target_hidden_states.reshape(
            -1, target_hidden_states.shape[-1]
        )
        flat_positions = target_positions.reshape(-1)
        starts = query_start_loc.to(torch.long)
        # last row of each request segment, optionally backed off by the
        # request's rejected-suffix count.
        seg_starts = starts[:batch_size]
        seg_ends = starts[1 : batch_size + 1]
        last_offsets = (seg_ends - seg_starts - 1).clamp(min=0)
        if rejected_for_gpu_mask is not None:
            rejected = rejected_for_gpu_mask.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            ).view(batch_size)
            last_offsets = (last_offsets - rejected).clamp(min=0)
        anchor_idx = (seg_starts + last_offsets).clamp(
            min=0, max=flat_hidden.shape[0] - 1
        )
        last_hidden = flat_hidden.index_select(0, anchor_idx).contiguous()
        last_positions = flat_positions.index_select(0, anchor_idx).contiguous()
        # hidden_by_req / positions_by_req carry the flat tensors here; the
        # ragged prefill_main consumes them together with query_start_loc.
        return (
            flat_hidden,
            flat_positions,
            last_hidden,
            last_positions,
            rejected_for_gpu_mask,
            query_start_loc,
        )

    def _trim_rejected_target_context(
        self,
        target_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop rejected verification suffixes before updating DSpark context.

        Padded speculative decoding keeps rejected tokens in the target forward
        as padding and expects proposers to ignore them. DSpark stores target
        hidden states in an internal context cache, so rejected suffix states
        must be removed before `prefill_main()`.
        """
        if num_rejected_tokens_gpu is None or common_attn_metadata is None:
            return target_hidden_states, target_positions

        rejected = num_rejected_tokens_gpu.detach().cpu().tolist()
        if not any(int(value) for value in rejected):
            return target_hidden_states, target_positions

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None:
            query_start_loc_cpu = common_attn_metadata.query_start_loc.detach().cpu()
        query_starts = query_start_loc_cpu.tolist()
        if len(query_starts) != len(rejected) + 1:
            raise ValueError(
                "DSpark rejected-context trimming requires query_start_loc "
                "to have batch_size + 1 entries; got "
                f"{len(query_starts)} starts for {len(rejected)} requests."
            )

        hidden_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        effective_lengths: list[int] = []
        flat_positions = target_positions.reshape(-1)
        for req_index, num_rejected in enumerate(rejected):
            start = int(query_starts[req_index])
            end = int(query_starts[req_index + 1]) - int(num_rejected)
            if end <= start:
                raise ValueError(
                    "DSpark rejected-context trimming removed every token for "
                    f"request {req_index}: start={start}, end={end}."
                )
            hidden_chunks.append(target_hidden_states[start:end])
            position_chunks.append(flat_positions[start:end])
            effective_lengths.append(end - start)

        if len(set(effective_lengths)) != 1:
            raise ValueError(
                "DSpark currently requires uniform effective per-request "
                "target context lengths after rejection trimming; got "
                f"{effective_lengths}."
            )

        return torch.cat(hidden_chunks, dim=0), torch.cat(position_chunks, dim=0)

    def _warmup_drafts(self, batch_size: int) -> torch.Tensor:
        self._last_draft_lengths = [self.num_speculative_tokens] * batch_size
        return make_dspark_warmup_draft_token_ids(
            batch_size=batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
            noise_token_id=self.noise_token_id,
            device=self.device,
        )

    def _draft_lengths_from_confidence(
        self,
        confidence_rows: list[list[float]],
    ) -> list[int]:
        return list(self._schedule_from_confidence(confidence_rows).lengths)

    def _schedule_from_confidence(
        self,
        confidence_rows: list[list[float]],
    ):
        confidence_rows = [
            row[: self.num_speculative_tokens] for row in confidence_rows
        ]
        forced_length = getattr(self, "_forced_draft_length", None)
        if forced_length is not None:
            lengths = [
                min(int(forced_length), self.num_speculative_tokens, len(row))
                for row in confidence_rows
            ]
            return score_prefix_lengths(
                confidence_rows,
                lengths,
                steps_per_second=self._steps_per_second,
            )

        scheduler = self._effective_confidence_scheduler()

        if scheduler == "hardware":
            return hardware_aware_prefix_schedule(
                confidence_rows,
                steps_per_second=self._steps_per_second,
                early_stop=getattr(self, "_hardware_scheduler_early_stop", True),
            )

        if scheduler == "off" or self.confidence_threshold <= 0.0:
            lengths = [
                min(self.num_speculative_tokens, len(row))
                for row in confidence_rows
            ]
        else:
            lengths = [
                confidence_threshold_prefix_length(
                    row,
                    self.confidence_threshold,
                )
                for row in confidence_rows
            ]
        return score_prefix_lengths(
            confidence_rows,
            lengths,
            steps_per_second=self._steps_per_second,
        )

    def _observe_confidence(self, confidence: torch.Tensor) -> list[int]:
        confidence_rows = confidence.detach().float().cpu().tolist()
        schedule = self._schedule_from_confidence(confidence_rows)
        self.diagnostics.observe(confidence_rows, schedule)
        return list(schedule.lengths)

    def _should_observe_confidence(self) -> bool:
        return (
            self._effective_confidence_scheduler() != "off"
            or getattr(self, "_collect_confidence_diagnostics", False)
        )

    def _needs_confidence(self) -> bool:
        return (
            self._should_observe_confidence()
            or getattr(self, "_collect_position0_diagnostics", False)
        )

    def _needs_draft_logits(self) -> bool:
        return self._export_draft_probs

    def take_last_draft_lengths(self) -> list[int] | None:
        lengths = self._last_draft_lengths
        self._last_draft_lengths = None
        return lengths

    def take_last_draft_probs(self) -> torch.Tensor | None:
        draft_probs = self._last_draft_probs
        self._last_draft_probs = None
        return draft_probs

    def take_last_confidence(self) -> torch.Tensor | None:
        confidence = self._last_confidence
        self._last_confidence = None
        return confidence

    def _maybe_store_draft_probs(
        self,
        draft_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
    ) -> None:
        self._last_draft_probs = None
        if not getattr(self, "_export_draft_probs", False):
            return
        if draft_logits.numel() == 0:
            logger.warning_once(
                "DSpark draft probability export requested but draft logits "
                "were not returned by the draft model."
            )
            return
        if not sampling_metadata.all_greedy:
            logger.warning_once(
                "VLLM_DSPARK_EXPORT_DRAFT_PROBS is currently limited to "
                "greedy DSpark requests because DSpark non-greedy drafting "
                "must sample the Markov-corrected block left-to-right."
            )
            return
        self._last_draft_probs = draft_logits[
            :batch_size, : self.num_speculative_tokens
        ].softmax(dim=-1, dtype=torch.float32)

    @override
    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
        num_speculative_tokens: int | None = None,
        req_ids: list[str] | None = None,
    ) -> torch.Tensor:
        # num_speculative_tokens is fixed at the DSpark block size; the jasl
        # gpu_model_runner passes it for other proposers — accept and ignore.
        del (
            target_token_ids,
            token_indices_to_sample,
            mm_embed_inputs,
            slot_mappings,
            num_speculative_tokens,
        )
        self._last_draft_probs = None
        self._last_confidence = None
        total_started = time.perf_counter()
        batch_size = self._batch_size(next_token_ids)
        # Bug 1: resolve a stable per-request KV slot for each batch row. None on
        # the identity permutation (single stream / no condense) -> original
        # byte-identical path. A permuted index forces eager draft below so the
        # captured cudagraph (fixed-shape, identity-only) is never replayed with
        # a permuted index_select.
        slot_index = self._resolve_slots(req_ids, batch_size)
        self._active_slot_index = slot_index

        def prepare_context():
            nonlocal target_hidden_states, target_positions
            rejected_for_gpu_mask = None
            gpu_mask_on = getattr(self, "_gpu_rejected_context_mask", False)
            if gpu_mask_on and num_rejected_tokens_gpu is not None:
                rejected_for_gpu_mask = num_rejected_tokens_gpu
            elif not gpu_mask_on:
                target_hidden_states, target_positions = (
                    self._trim_rejected_target_context(
                        target_hidden_states,
                        target_positions,
                        common_attn_metadata,
                        num_rejected_tokens_gpu,
                    )
                )

            # Bug 2: under the GPU rejected-context mask, detect ragged
            # per-request rows (chunked-prefill mixed prefill+decode step) from
            # query_start_loc and take a flat/ragged path. The uniform batch
            # keeps the original rectangular _view_by_request fast-path
            # (byte-identical). The ragged path runs eager in prefill_main.
            query_start_loc = None
            if gpu_mask_on:
                query_start_loc = self._maybe_ragged_query_start_loc(
                    common_attn_metadata, batch_size, target_hidden_states
                )

            if query_start_loc is not None:
                return self._prepare_context_ragged(
                    target_hidden_states,
                    target_positions,
                    query_start_loc,
                    batch_size,
                    rejected_for_gpu_mask,
                )

            hidden_by_req = self._view_by_request(target_hidden_states, batch_size)
            positions_by_req = self._positions_by_request(target_positions, batch_size)

            if rejected_for_gpu_mask is not None:
                rejected = rejected_for_gpu_mask.to(
                    device=hidden_by_req.device,
                    dtype=torch.long,
                    non_blocking=True,
                ).view(batch_size)
                last_indices = (
                    hidden_by_req.shape[1] - rejected - 1
                ).clamp(min=0)
                last_hidden = hidden_by_req.gather(
                    1,
                    last_indices.view(batch_size, 1, 1).expand(
                        -1,
                        -1,
                        hidden_by_req.shape[-1],
                    ),
                ).squeeze(1).contiguous()
                last_positions = positions_by_req.gather(
                    1,
                    last_indices.view(batch_size, 1),
                ).squeeze(1).contiguous()
            else:
                last_hidden = hidden_by_req[:, -1].contiguous()
                last_positions = positions_by_req[:, -1].contiguous()
            return (
                hidden_by_req,
                positions_by_req,
                last_hidden,
                last_positions,
                rejected_for_gpu_mask,
                None,
            )

        (
            hidden_by_req,
            positions_by_req,
            last_hidden,
            last_positions,
            rejected_for_gpu_mask,
            ragged_query_start_loc,
        ) = self._timed_stage("context_prepare", prepare_context)

        self._timed_stage(
            "prefill_main",
            lambda: self.model.prefill_main(
                hidden_by_req,
                positions_by_req,
                num_rejected_tokens=rejected_for_gpu_mask,
                slot_index=slot_index,
                query_start_loc=ragged_query_start_loc,
            ),
        )
        if not self._prefilled:
            self._prefilled = True
            return self._warmup_drafts(batch_size)

        def prepare_graph():
            (
                cudagraph_runtime_mode,
                padded_batch_size,
                num_tokens_across_dp,
                batch_descriptor,
            ) = self._determine_graph_batch(
                batch_size,
                # A permuted slot index must run eager: the captured graph was
                # recorded with the identity index_select and fixed (padded)
                # shapes. Disabling cudagraphs here also makes padded_batch_size
                # == batch_size so the length-batch_size slot index lines up with
                # the draft buffers (no padded rows to mis-slot).
                use_cudagraphs=slot_index is None,
            )
            self._prepare_draft_buffers(
                input_ids=next_token_ids,
                hidden_states=last_hidden,
                positions=last_positions,
                padded_batch_size=padded_batch_size,
            )
            return (
                cudagraph_runtime_mode,
                padded_batch_size,
                num_tokens_across_dp,
                batch_descriptor,
            )

        (
            cudagraph_runtime_mode,
            padded_batch_size,
            num_tokens_across_dp,
            batch_descriptor,
        ) = self._timed_stage("graph_prepare", prepare_graph)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=padded_batch_size * self.num_speculative_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            draft_token_ids, draft_logits, confidence = (
                self._timed_stage("draft", self._run_draft_for_current_context)
            )

        def postprocess():
            self._maybe_store_draft_probs(draft_logits, sampling_metadata, batch_size)
            confidence_for_batch = None
            if confidence is not None and confidence.numel() > 0:
                confidence_for_batch = confidence[
                    :batch_size, : self.num_speculative_tokens
                ]
                if getattr(self, "_collect_position0_diagnostics", False):
                    self._last_confidence = confidence_for_batch.detach().clone()
            forced_length = getattr(self, "_forced_draft_length", None)
            if forced_length is not None:
                self._last_draft_lengths = [int(forced_length)] * batch_size
            elif (
                confidence_for_batch is not None
                and self._should_observe_confidence()
            ):
                self._last_draft_lengths = self._observe_confidence(
                    confidence_for_batch
                )
            else:
                self._last_draft_lengths = [self.num_speculative_tokens] * batch_size
            return draft_token_ids[:batch_size, : self.num_speculative_tokens].to(
                torch.int32
            )

        result = self._timed_stage("postprocess", postprocess)
        self._record_stage_timing(
            "total",
            (time.perf_counter() - total_started) * 1000.0,
        )
        self._maybe_log_stage_timing()
        return result

    def get_diagnostics_snapshot(self) -> Any:
        return self.diagnostics.snapshot()
