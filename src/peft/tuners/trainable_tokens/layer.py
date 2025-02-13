# Copyright 2025-present the HuggingFace Inc. team.
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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class TrainableTokensLayer(nn.Module, BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        token_indices: List[int],
        **kwargs,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer
        self._active_adapter = adapter_name
        self.token_indices = token_indices
        self.kwargs = kwargs

        # we store the delta weight on particular tokens
        self.trainable_tokens_delta_tokens = nn.ParameterDict({})
        self.sparse_delta_tokens = {}

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # we set parameters on layer sizing
        self.num_trainable_embeddings = len(token_indices)  # this is similar to `num_embeddings`
        self.embedding_dim = base_layer.embedding_dim  # token from the embedding
        self.num_total_embeddings = base_layer.num_embeddings  # total number of tokens in the vocabulary

        # we set the number of trainable tokens
        self.update_layer(adapter_name)

    def update_layer(self, adapter_name):
        # we initialize the delta embedding weights and store them in a trainable parameter
        values = torch.rand(
            (self.num_trainable_embeddings * self.base_layer.weight.shape[-1],)
        )  # we initialize the values from a normal distribution N(0, 1), as in https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

        # cause safetensors doesn't support sparse tensors, we need to store the values in a dense tensor and then convert it to a sparse tensor when called
        self.trainable_tokens_delta_tokens[adapter_name] = nn.Parameter(values, requires_grad=True)

        # we created the indices of the sparse tensor
        r = torch.Tensor(self.token_indices).long()
        c = torch.arange(self.embedding_dim)
        indices = torch.stack(
            [r.repeat(self.embedding_dim), c.repeat(self.num_trainable_embeddings, 1).t().reshape(-1)], dim=1
        ).T

        # we create the sparse tensor from our `delta_tokens` and `indices
        self.sparse_delta_tokens[adapter_name] = torch.sparse_coo_tensor(
            indices=indices,
            values=self.trainable_tokens_delta_tokens[adapter_name],
            size=(self.num_total_embeddings, self.embedding_dim),
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)

        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            orig_weights = self.base_layer.weight.data
            merged = orig_weights + self.sparse_delta_tokens[active_adapter]

            if safe_merge and not torch.isfinite(merged).all():
                raise ValueError(
                    f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                )
            else:
                self.base_layer.weight.data = merged

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            self.base_layer.weight.data -= self.sparse_delta_tokens[active_adapter]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            deltas = None
            for active_adapter in self.active_adapters:
                delta = self.sparse_delta_tokens[active_adapter]
                if deltas is None:
                    deltas = delta
                else:
                    deltas += delta

            W = self.base_layer.weight

            result = F.embedding(
                input=x,
                weight=W + deltas,
                padding_idx=self.base_layer.padding_idx,
                max_norm=self.base_layer.max_norm,
                norm_type=self.base_layer.norm_type,
                scale_grad_by_freq=self.base_layer.scale_grad_by_freq,
                sparse=self.base_layer.sparse,
            )

        return result
