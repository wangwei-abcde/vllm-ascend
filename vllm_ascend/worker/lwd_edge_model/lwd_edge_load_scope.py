#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Embed-only model loading toolkit for the LWD edge (stdlib only).

The LWD edge runs embedding + lm_head only (no decoder layers, no KV
cache). Two pure helpers cover the whole load-side story:

- :func:`edge_embed_only_load`: the standard model loader decides how
  many decoder layers to build from the hf config's
  ``num_hidden_layers``; the edge load is wrapped in a scope that
  temporarily zeroes it, so the loader builds zero decoder layers
  natively while upstream files stay untouched. The true depth is
  restored on exit, including on exceptions, so every consumer after
  the load -- Hash routing table validation, logging -- sees the real
  value, matching the mm branch's "config keeps N, model has zero
  layers" edge semantics.

- :func:`is_layer_weight_out_of_range`: the checkpoint always ships
  every layer, so the loader filters weights of decoder layers this
  process does not hold by layer index (``start_layer <= idx <
  end_layer``); non-layer weights (embeddings, norms, heads) are never
  filtered.

Stdlib-only by design so the load logic stays testable on any
interpreter.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_LAYER_INDEX_RE = re.compile(r"\blayers\.(\d+)\.")


def is_layer_weight_out_of_range(name: str, start_layer: int, end_layer: int) -> bool:
    """True when the weight belongs to a decoder layer this process does not hold."""
    match = _LAYER_INDEX_RE.search(name)
    if match is None:
        return False
    return not (start_layer <= int(match.group(1)) < end_layer)


@contextmanager
def edge_embed_only_load(config: Any) -> Iterator[None]:
    """Suppress decoder-layer construction for the duration of the load.

    Resolves the text config (VL wrappers nest it) so both plain and
    multimodal hf configs are covered; a missing or already-zero field
    makes the scope a no-op passthrough.
    """
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    real_layers = getattr(text_config, "num_hidden_layers", None)
    if real_layers:
        text_config.num_hidden_layers = 0
    try:
        yield
    finally:
        if real_layers:
            text_config.num_hidden_layers = real_layers
