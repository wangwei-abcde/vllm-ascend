# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

if TYPE_CHECKING:
    from typing import Any

    from vllm.v1.core.sched.output import SamplingMetadata


class LwdMTPProposer(AscendEagleProposer):
    """Adapt externally supplied LWD embeddings to the MTP SP layout."""

    def _run_merged_draft(
        self,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor | None,
        target_positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        multi_steps_attn_metadata: list[Any],
        num_tokens: int,
        is_prefill: bool | None = None,
        # Base contract since the probabilistic-draft change; the LWD path
        # does not consume it but must accept and forward it.
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        """Shard external embeddings in the active draft context before forward."""
        context = get_forward_context()
        # flash_comm_v1_enabled is only set by the SP fast path; without SP
        # (the LWD 1+8 TP layout) it is absent and the sharding is skipped.
        if (
            inputs_embeds is not None
            and getattr(context, "flash_comm_v1_enabled", False)
            and not self.is_multimodal_model
        ):
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_input_tokens:
                    raise ValueError("LWD MTP SP expects full-token inputs_embeds before slicing")
                # Embeddings are already replicated; reduce-scatter would sum
                # their values again, so only pad and slice the token rows.
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
        return super()._run_merged_draft(
            num_input_tokens,
            batch_size,
            token_indices_to_sample,
            target_positions,
            inputs_embeds,
            multi_steps_attn_metadata,
            num_tokens,
            is_prefill,
            sampling_metadata,
        )
