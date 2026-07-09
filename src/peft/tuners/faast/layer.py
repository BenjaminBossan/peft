# Copyright 2026-present the HuggingFace Inc. team.
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
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from .._buffer_dict import BufferDict
from .config import FaastConfig


class FaastLayer(nn.Module, BaseTunerLayer):
    """
    Common machinery of FAAST layers: streaming accumulation of the regression statistics and the closed-form solve.

    The fast weights map keys of dimension `d_key` to values of dimension `d_val`. Instead of materializing all
    keys/values and solving `W = pinv(K) @ V` like the reference implementation, the Gram matrix `G = K^T K` and the
    cross term `C = K^T V` are accumulated across learn passes and the (spectrally filtered) pseudoinverse solution is
    obtained from the eigendecomposition of `G`. This is mathematically equivalent, supports streaming over support
    batches without the inexact incremental averaging of the paper (Eq. 14), and only requires `O(d^2)` memory.

    Subclasses define where the key/value pairs come from (see their `begin_learn`) and how the prediction enters the
    forward output.
    """

    adapter_layer_names = ("faast_W", "faast_A", "faast_B")
    other_param_names = (
        "faast_num_tokens",
        "r",
        "filter_alpha",
        "memory_weight",
        "kv_source",
        "preserve_output_scale",
    )

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: FaastConfig, d_key: int, d_val: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.d_key = d_key
        self.d_val = d_val

        # Full-rank fast weights (used when config.r is None), stored transposed like nn.Linear, i.e. (d_val, d_key).
        self.faast_W = nn.ParameterDict({})
        # Low-rank factors (used when config.r is set): prediction = (h @ A) @ B.
        self.faast_A = nn.ParameterDict({})
        self.faast_B = nn.ParameterDict({})
        # Number of key/value pairs compiled into the fast weights; 0 means the adapter is a no-op.
        self.faast_num_tokens = BufferDict({}, persistent=True)

        self.r = {}
        self.filter_alpha = {}
        self.memory_weight = {}
        self.kv_source = {}
        self.preserve_output_scale = {}

        # The fast weights are computed in closed form, never through gradient descent.
        self.frozen_peft_weight_names: dict[str, tuple[str, ...]] = {}

        # Accumulators for the closed-form solve; plain dicts so that they don't end up in the state dict.
        self._faast_gram: dict[str, torch.Tensor] = {}
        self._faast_cross: dict[str, torch.Tensor] = {}
        self._faast_val_sqnorm: dict[str, torch.Tensor] = {}
        # Fraction of the target energy explained by the fast weights on the learning data; a diagnostic for how much
        # the least-squares prediction is attenuated relative to the true targets (regression toward the mean).
        # Filled by the solve.
        self.faast_fit_ratio: dict[str, float] = {}
        # Context set by FaastModel.begin_learn for the duration of the learning forward passes.
        self._learn_context: Optional[dict[str, Any]] = None

        self._disable_adapters = False
        self.merged_adapters = []
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def update_layer(self, adapter_name: str, config: FaastConfig) -> None:
        self.r[adapter_name] = config.r
        self.filter_alpha[adapter_name] = config.filter_alpha
        self.memory_weight[adapter_name] = config.memory_weight
        self.kv_source[adapter_name] = config.kv_source
        self.preserve_output_scale[adapter_name] = config.preserve_output_scale
        self.frozen_peft_weight_names[adapter_name] = ("faast_W", "faast_A", "faast_B")

        device = next(self.base_layer.parameters()).device
        # The fast weights are kept in float32; they result from a float64 solve and adding them in half precision
        # would needlessly lose the small spectral components.
        if config.r is None:
            self.faast_W[adapter_name] = nn.Parameter(
                torch.zeros(self.d_val, self.d_key, device=device, dtype=torch.float32), requires_grad=False
            )
        else:
            self.faast_A[adapter_name] = nn.Parameter(
                torch.zeros(self.d_key, config.r, device=device, dtype=torch.float32), requires_grad=False
            )
            self.faast_B[adapter_name] = nn.Parameter(
                torch.zeros(config.r, self.d_val, device=device, dtype=torch.float32), requires_grad=False
            )
        self.faast_num_tokens[adapter_name] = torch.tensor(0, device=device, dtype=torch.long)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def reset_fast_weights(self, adapter_name: Optional[str] = None) -> None:
        adapter_names = [adapter_name] if adapter_name is not None else list(self.faast_num_tokens.keys())
        for name in adapter_names:
            self._faast_gram.pop(name, None)
            self._faast_cross.pop(name, None)
            self._faast_val_sqnorm.pop(name, None)
            self.faast_fit_ratio.pop(name, None)
            self.faast_num_tokens[name] = torch.zeros_like(self.faast_num_tokens[name])
            with torch.no_grad():
                if name in self.faast_W:
                    self.faast_W[name].zero_()
                if name in self.faast_A:
                    self.faast_A[name].zero_()
                    self.faast_B[name].zero_()

    def begin_learn(self, **learn_context: Any) -> None:
        self._learn_context = learn_context

    def end_learn(self, solve: bool = True) -> None:
        self._learn_context = None
        if solve:
            self.solve_fast_weights()

    def solve_fast_weights(self) -> None:
        for adapter_name in self.active_adapters:
            if adapter_name in self._faast_gram:
                self._solve(adapter_name)

    def _accumulate_pairs(self, adapter_name: str, keys: torch.Tensor, vals: torch.Tensor) -> None:
        """Accumulate the regression statistics for flattened (num_pairs, d_key)/(num_pairs, d_val) tensors."""
        if keys.numel() == 0:
            return
        keys = keys.to(torch.float32)
        vals = vals.to(torch.float32)
        if adapter_name not in self._faast_gram:
            self._faast_gram[adapter_name] = torch.zeros(
                self.d_key, self.d_key, device=keys.device, dtype=torch.float32
            )
            self._faast_cross[adapter_name] = torch.zeros(
                self.d_key, self.d_val, device=keys.device, dtype=torch.float32
            )
            self._faast_val_sqnorm[adapter_name] = torch.zeros((), device=keys.device, dtype=torch.float32)
        self._faast_gram[adapter_name] += keys.T @ keys
        self._faast_cross[adapter_name] += keys.T @ vals
        self._faast_val_sqnorm[adapter_name] += (vals**2).sum()
        self.faast_num_tokens[adapter_name] = self.faast_num_tokens[adapter_name] + keys.size(0)

    def _solve(self, adapter_name: str) -> None:
        """Compute the spectrally filtered least-squares solution from the accumulated statistics.

        With K the matrix of keys and V the matrix of values, the paper computes `W* = pinv(K) V`, dropping singular
        values of K below `sigma_max * epsilon`. Using the SVD `K = U S R^T`, we have `G = K^T K = R S^2 R^T` and
        `pinv(K) V = R S^-2 R^T (K^T V)`, so the same solution is recovered from the eigendecomposition of G.
        """
        gram = self._faast_gram[adapter_name].to(torch.float64)
        cross = self._faast_cross[adapter_name].to(torch.float64)
        num_tokens = int(self.faast_num_tokens[adapter_name])

        evals, evecs = torch.linalg.eigh(gram)  # ascending order
        sigma = evals.clamp(min=0.0).sqrt()
        epsilon = 1.0 / (num_tokens ** self.filter_alpha[adapter_name])
        keep = sigma > sigma.max() * epsilon
        r = self.r[adapter_name]
        if r is not None:
            # eigenvalues are ascending, so the top-r directions are the last r entries
            keep[:-r] = False
        evecs = evecs[:, keep]
        inv_evals = 1.0 / evals[keep]
        mid = inv_evals.unsqueeze(1) * (evecs.T @ cross)

        # fraction of the target energy explained by the solution on the learning data:
        # sum_i ||W k_i||^2 = tr(W^T G W) = sum_j evals_j * ||mid_j||^2 since evecs^T G evecs = diag(evals)
        explained = (evals[keep].unsqueeze(1) * mid**2).sum()
        total = self._faast_val_sqnorm[adapter_name].to(torch.float64)
        self.faast_fit_ratio[adapter_name] = float(explained / total) if total > 0 else 0.0

        with torch.no_grad():
            if r is None:
                w_mat = evecs @ mid
                self.faast_W[adapter_name].copy_(w_mat.T.to(torch.float32))
            else:
                # if fewer than r directions survive the filter, the remaining factor columns/rows stay zero so that
                # the parameter shapes are independent of the support data
                num_kept = evecs.size(1)
                self.faast_A[adapter_name].zero_()
                self.faast_B[adapter_name].zero_()
                self.faast_A[adapter_name][:, :num_kept].copy_(evecs.to(torch.float32))
                self.faast_B[adapter_name][:num_kept].copy_(mid.to(torch.float32))

    def _predict(self, adapter_name: str, hidden_states: torch.Tensor) -> torch.Tensor:
        if adapter_name in self.faast_W:
            weight = self.faast_W[adapter_name]
            return nn.functional.linear(hidden_states, weight.to(hidden_states.dtype))
        a = self.faast_A[adapter_name].to(hidden_states.dtype)
        b = self.faast_B[adapter_name].to(hidden_states.dtype)
        return (hidden_states @ a) @ b

    def _interpolate(self, adapter_name: str, base_output: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        weight = self.memory_weight[adapter_name]
        return (1 - weight) * base_output + weight * prediction

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("FAAST layers cannot be merged into the base model.")

    def unmerge(self) -> None:
        raise NotImplementedError("FAAST layers cannot be merged into the base model.")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "faast." + rep


class FaastDecoderLayer(FaastLayer):
    """
    FAAST layer that wraps a whole decoder layer of a causal language model (not a single linear layer).

    In learn mode, the wrapped layer's output hidden states at position t are collected as keys and the input
    embeddings at position t+1 as values ("output memory" of the reference implementation). In inference mode, the
    layer output is interpolated towards the fast-weight prediction:
    `h <- (1 - memory_weight) * h + memory_weight * (h @ W^T)`.
    """

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: FaastConfig, hidden_size: int) -> None:
        super().__init__(base_layer, adapter_name, config, d_key=hidden_size, d_val=hidden_size)

    def update_layer(self, adapter_name: str, config: FaastConfig) -> None:
        if config.preserve_output_scale:
            raise ValueError(
                "`preserve_output_scale` is only supported when targeting linear output heads; decoder layer outputs "
                "pass through the model's final norm, which makes their scale irrelevant."
            )
        super().update_layer(adapter_name, config=config)

    def _accumulate(self, hidden_states: torch.Tensor) -> None:
        ctx = self._learn_context
        if "inputs_embeds" not in ctx:
            raise ValueError(
                "FAAST layers wrapping decoder layers require `inputs_embeds` in the learn context; use "
                "`FaastModel.learn` for causal language models."
            )
        inputs_embeds = ctx["inputs_embeds"].to(hidden_states.device)
        attention_mask = ctx.get("attention_mask")
        answer_mask = ctx.get("answer_mask")

        keys = hidden_states[:, :-1, :]
        vals = inputs_embeds[:, 1:, :]
        # a (key_t, val_{t+1}) pair is only valid if both positions are real tokens
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)
            pair_mask = (attention_mask[:, 1:] > 0) & (attention_mask[:, :-1] > 0)
        else:
            pair_mask = torch.ones(keys.shape[:2], dtype=torch.bool, device=hidden_states.device)

        for adapter_name in self.active_adapters:
            if adapter_name not in self.memory_weight:
                continue
            mask = pair_mask
            if self.kv_source[adapter_name] == "answer":
                if answer_mask is None:
                    raise ValueError(
                        f"FAAST adapter {adapter_name} is configured with kv_source='answer' but no `answer_mask` "
                        "was passed to `learn`."
                    )
                mask = mask & (answer_mask.to(hidden_states.device)[:, 1:] > 0)

            flat_mask = mask.reshape(-1)
            k = keys.reshape(-1, keys.size(-1))[flat_mask]
            v = vals.reshape(-1, vals.size(-1))[flat_mask]
            self._accumulate_pairs(adapter_name, k, v)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Any:
        output = self.base_layer(hidden_states, *args, **kwargs)
        if self.disable_adapters or not self.active_adapters:
            return output

        # depending on the transformers version, decoder layers return a tensor or a tuple with the hidden states as
        # the first element
        is_tuple = isinstance(output, tuple)
        new_hidden_states = output[0] if is_tuple else output

        if self._learn_context is not None:
            self._accumulate(new_hidden_states)
            return output

        modified = False
        for adapter_name in self.active_adapters:
            if adapter_name not in self.memory_weight:
                continue
            if int(self.faast_num_tokens[adapter_name]) == 0:
                continue
            prediction = self._predict(adapter_name, new_hidden_states)
            new_hidden_states = self._interpolate(adapter_name, new_hidden_states, prediction)
            modified = True

        if not modified:
            return output
        if is_tuple:
            return (new_hidden_states,) + output[1:]
        return new_hidden_states


class FaastLinear(FaastLayer):
    """
    FAAST layer that wraps an `nn.Linear` output head, e.g. the final projection of a diffusion transformer.

    Unlike [`FaastDecoderLayer`], the regression targets are not derived from the inputs but must be supplied
    externally while learning: `begin_learn(values=...)` with `values` of shape `(*input_batch_dims, out_features)`,
    aligned position by position with the layer input (e.g. the ground-truth flow matching targets of a diffusion
    model). In inference mode, the base layer output is interpolated towards the fast-weight prediction:
    `out <- (1 - memory_weight) * base_layer(x) + memory_weight * (x @ W^T)`.
    """

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: FaastConfig) -> None:
        super().__init__(base_layer, adapter_name, config, d_key=base_layer.in_features, d_val=base_layer.out_features)

    @torch.no_grad()
    def _accumulate(self, x: torch.Tensor) -> None:
        ctx = self._learn_context
        if "values" not in ctx:
            raise ValueError(
                "FAAST layers wrapping linear layers require `values` in the learn context, i.e. "
                "`begin_learn(values=...)` with the regression targets for each input position."
            )
        values = ctx["values"].to(x.device)
        if values.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"The learn context `values` must align with the layer input positions, got values of shape "
                f"{tuple(values.shape)} for input of shape {tuple(x.shape)}."
            )
        keys = x.reshape(-1, x.size(-1))
        vals = values.reshape(-1, values.size(-1))
        for adapter_name in self.active_adapters:
            if adapter_name not in self.memory_weight:
                continue
            self._accumulate_pairs(adapter_name, keys, vals)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = self.base_layer(x, *args, **kwargs)
        if self.disable_adapters or not self.active_adapters:
            return output

        if self._learn_context is not None:
            self._accumulate(x)
            return output

        for adapter_name in self.active_adapters:
            if adapter_name not in self.memory_weight:
                continue
            if int(self.faast_num_tokens[adapter_name]) == 0:
                continue
            prediction = self._predict(adapter_name, x)
            new_output = self._interpolate(adapter_name, output, prediction)
            if self.preserve_output_scale[adapter_name]:
                # the least-squares prediction is systematically shorter than the true targets (it is missing the
                # residual energy), which would scale down the output; keep the base prediction's per-token norm and
                # only adopt the direction of the interpolated output
                base_norm = output.norm(dim=-1, keepdim=True)
                new_norm = new_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                new_output = new_output * (base_norm / new_norm)
            output = new_output
        return output
