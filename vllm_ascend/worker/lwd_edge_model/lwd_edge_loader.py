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
"""Embed-only model loader for the LWD edge (registry-swapped loader).

The edge process builds embed + lm_head only: the loader zeroes the
text config's layer count for the construction (see lwd_edge_load_scope)
and filters checkpoint weights of decoder layers this process does not
hold out of the weight stream, so the shared model files need no
LWD-specific guards (the hand-rolled load_weights of e.g. DeepSeek V4
recognizes foreign layers via PPMissingLayer placeholder names, which an
empty layer list cannot provide).

Dispatch goes through vLLM's own loader registry: for the duration of
the load the registry entry for the deployment's own load format is
swapped to this loader (:func:`lwd_edge_loader_format`). The
``load_config.load_format`` string itself is never changed --
``DefaultModelLoader._prepare_weights`` validates it against vLLM's
known formats -- and the cloud and every non-LWD process keep the stock
``DefaultModelLoader`` untouched.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager

import torch
from torch import nn
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader import (
    get_model_loader,
    register_model_loader,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm_ascend.worker.lwd_edge_model.lwd_edge_load_scope import (
    edge_embed_only_load,
    is_layer_weight_out_of_range,
)


class LwdEdgeEmbedOnlyLoader(DefaultModelLoader):
    """DefaultModelLoader with zero decoder layers and foreign layer
    weights filtered from the stream; see the module docstring."""

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        with edge_embed_only_load(model_config.hf_config):
            return super().load_model(vllm_config, model_config, prefix)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        inner = getattr(model, "model", model)
        start_layer = getattr(inner, "start_layer", None)
        end_layer = getattr(inner, "end_layer", None)
        if start_layer is None or end_layer is None:
            raise ValueError("LWD edge loader requires start_layer/end_layer on the model")
        for name, weight in super().get_all_weights(model_config, model):
            if is_layer_weight_out_of_range(name, start_layer, end_layer):
                continue
            yield name, weight


@contextmanager
def lwd_edge_loader_format(load_config: LoadConfig) -> Iterator[None]:
    """Route the deployment's own load format to the embed-only loader
    for the duration of the context (edge process only).

    The registry entry is swapped, never ``load_config.load_format``:
    ``DefaultModelLoader._prepare_weights`` validates that string
    against vLLM's known formats and rejects custom names.
    """
    source_format = load_config.load_format
    original_cls = type(get_model_loader(load_config))
    register_model_loader(source_format)(LwdEdgeEmbedOnlyLoader)
    try:
        yield
    finally:
        register_model_loader(source_format)(original_cls)
