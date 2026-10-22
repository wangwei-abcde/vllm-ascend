# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Rotating pinned staging buffers shared by the LWD edge/cloud host transfers.

Consolidates the hand-rolled rotation loops that used to live in
``LwdEdgeWorker._lwd_embed_ids_h2d`` and ``LwdCloudModelRunner
._lwd_h2d_stage_i64`` (pinned + device mirror pairs). Rotation depth 4
covers batch_queue depth 2 plus one step of engine lag, so a later step
cannot overwrite a buffer whose transfer is still queued.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class LwdPinnedStaging:
    """Pinned staging slots rotated per use, with an optional device mirror.

    ``stage_to_device`` stages host data into the next pinned slot and
    issues a ``non_blocking`` H2D into the mirrored device slot, returning
    a ``[:n]`` device view; ``next_pinned`` rotates and returns the full
    pinned slot for the caller to fill and slice. Capacity grows (never
    shrinks) to ``max(n, min_capacity)`` on first use or overflow,
    rebuilding every slot and restarting rotation at slot 0.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        depth: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        self._dtype = dtype
        self._min_capacity = min_capacity
        self._depth = depth
        self._device = device
        self._pinned_slots: list[torch.Tensor] | None = None
        self._device_slots: list[torch.Tensor] | None = None
        self._index = 0

    def _ensure(self, n: int) -> None:
        if self._pinned_slots is not None and self._pinned_slots[0].numel() >= n:
            return
        capacity = max(n, self._min_capacity)
        self._pinned_slots = [torch.empty(capacity, dtype=self._dtype, pin_memory=True) for _ in range(self._depth)]
        self._device_slots = (
            None
            if self._device is None
            else [torch.empty(capacity, dtype=self._dtype, device=self._device) for _ in range(self._depth)]
        )
        self._index = 0

    def _rotate(self) -> int:
        index = self._index
        self._index = (index + 1) % self._depth
        return index

    def stage_to_device(self, src: Sequence[int] | np.ndarray, n: int) -> torch.Tensor:
        """Stage ``src`` and issue an async H2D; return the ``[:n]`` device view.

        Same-stream ordering guarantees the consumer kernel reads this
        step's values (H2D enqueued before the consumer), matching the
        retired inline staging loops.
        """
        self._ensure(n)
        index = self._rotate()
        stage = self._pinned_slots[index]
        stage.numpy()[:n] = src
        device = self._device_slots[index]
        device[:n].copy_(stage[:n], non_blocking=True)
        return device[:n]

    def next_pinned(self, n: int) -> torch.Tensor:
        """Rotate and return the next full pinned slot (caller fills/slices)."""
        self._ensure(n)
        return self._pinned_slots[self._rotate()]
