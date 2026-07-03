# Copyright 2024-present the HuggingFace Inc. team.
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

from copy import deepcopy
from functools import wraps
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


ENABLE_DORA_CACHING = False
"""Whether to enable DoRA caching, which makes it faster at inference but requires more memory"""


def cache_decorator(cache_key: str):
    """Caching decorator for DoRA

    Caching is only enabled if ENABLE_DORA_CACHING is set to True (default: False), when in eval mode, and when the
    adapter_name is passed (e.g. not during layer initialization).

    """

    def cache_value(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # if adapter_name is not passed, no caching
            adapter_name = kwargs.get("adapter_name")
            if (not ENABLE_DORA_CACHING) or self.training or (adapter_name is None):
                self._cache_clear()
                return func(self, *args, **kwargs)

            cache_key_adapter = f"{cache_key}-{adapter_name}"
            output = self._cache_get(cache_key_adapter, None)
            if output is not None:
                return output

            output = func(self, *args, **kwargs)
            self._cache_store(cache_key_adapter, output)
            return output

        return wrapper

    return cache_value


class DoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out
        self._dora_cache: dict[str, Any] = {}  # small ad hoc cache; values are not part of the state_dict

    def _cache_store(self, key: str, value: Any) -> None:
        # cache intermediate values, e.g. weight norm of DoRA
        self._dora_cache[key] = value

    def _cache_get(self, key: str, default: Optional[Any]) -> Optional[Any]:
        # retrieve from ad hoc cache
        return self._dora_cache.get(key, default)

    def _cache_clear(self) -> None:
        self._dora_cache.clear()

    def train(self, mode: bool = True):
        if mode:
            self._cache_clear()
        super().train(mode=mode)
        return self

    @cache_decorator("weight-norm")
    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    @cache_decorator("lora-weight")
    def get_lora_weight(self, lora_A, lora_B, adapter_name: Optional[str] = None):
        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=lora_A.weight.dtype)
        lora_weight = lora_B(lora_A(x_eye)).T
        return lora_weight

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == "Linear4bit":
                # We have to create a copy of the base layer, otherwise, FSDP will throw an error. 8bit does not work
                # yet because Int8Params cannot be correctly deep-copied (attributes vanish)
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)
            if weight.data.ndim >= 3:  # For handling LoRAs applied to Conv layers.
                r = lora_A.shape[0]
                lora_weight = torch.mm(lora_B.view([-1, r]), lora_A.view([r, -1]))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B @ lora_A

            if dtype_is_fp16:
                lora_weight = lora_weight.half()
            weight_norm = self.get_weight_norm(
                weight=weight.to(lora_A.device), lora_weight=lora_weight, scaling=scaling
            )

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name="default"):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = self.get_lora_weight(lora_A=lora_A, lora_B=lora_B, adapter_name=adapter_name)
        lora_weight = lora_weight.to(x.dtype)

        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        weight_norm = self.get_weight_norm(
            weight=weight, lora_weight=lora_weight.detach(), scaling=scaling, adapter_name=adapter_name
        )
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        lora_result = lora_B(lora_A(x))

        bias = None
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, transpose(weight, self.fan_in_fan_out))

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraEmbeddingLayer(DoraLinearLayer):
    @cache_decorator("lora-weight")
    def get_lora_weight(self, lora_A, lora_B, adapter_name: Optional[str] = None):
        return (lora_A @ lora_B).T

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn, adapter_name="default"):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = self.get_lora_weight(lora_A=lora_A, lora_B=lora_B, adapter_name=adapter_name)
        magnitude = self.weight
        weight = base_layer.weight
        weight_norm = self.get_weight_norm(
            weight=weight, lora_weight=lora_weight.detach(), scaling=scaling, adapter_name=adapter_name
        )
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = mag_norm_scale * (embed_fn(x, lora_A) @ lora_B) * scaling
        return mag_norm_scale, result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class _DoraConvNdLayer(DoraLinearLayer):
    @cache_decorator("weight-norm")
    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4/5D weight tensors of Conv2D/3D
        dim = tuple(range(1, weight.dim()))
        weight_norm = weight.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        return weight_norm

    @cache_decorator("lora-weight")
    def get_lora_weight(self, lora_A, lora_B, adapter_name: Optional[str] = None) -> torch.Tensor:
        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        r = lora_A.weight.shape[0]
        lora_weight = torch.mm(lora_B.weight.view([-1, r]), lora_A.weight.view([r, -1]))
        return lora_weight

    def forward(
        self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name: str = "default"
    ) -> torch.Tensor:
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = base_layer.weight
        lora_weight = self.get_lora_weight(lora_A=lora_A, lora_B=lora_B, adapter_name=adapter_name).reshape(
            weight.shape
        )
        magnitude = self.weight
        weight_norm = self.get_weight_norm(
            weight=weight, lora_weight=lora_weight.detach(), scaling=scaling, adapter_name=adapter_name
        )
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm

        if base_result is None:
            base_result = self.conv_fn(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        else:
            bias = base_layer.bias
            if bias is not None:
                # reshape bias to (1, -1, 1, ...)
                bias_shape = (1, -1) + (1,) * (base_result.dim() - 2)
                base_result = base_result - bias.view(*bias_shape)

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraConv1dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv1d


class DoraConv2dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv2d


class DoraConv3dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d






class DoraKernelLinearLayer(nn.Module):
    """DoRA layer using the low-rank decomposition of the weight norm.

    See module docstring for details.
    """

    def __init__(self, fan_in_fan_out: bool):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    @staticmethod
    def get_weight_norm_lowrank(
        weight: torch.Tensor,
        lora_A_weight: torch.Tensor,
        lora_B_weight: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        """Compute ``||W + s*BA||_c`` without materializing BA.

        Args:
            weight: base weight, shape ``(out, in)``.
            lora_A_weight: shape ``(r, in)``.
            lora_B_weight: shape ``(out, r)``.
            scaling: LoRA scaling factor ``alpha / r``.

        Returns:
            ``weight_norm`` of shape ``(out,)`` — the L2 norm of each row of ``W + s*BA``.
        """
        # ||W||²_c  -- shape (out,)
        w_norm_sq = (weight * weight).sum(dim=1)

        # <W, BA>_c = (W @ Aᵀ * B).sum(dim=1)
        wa = weight @ lora_A_weight.t()  # (out, r)
        cross = (wa * lora_B_weight).sum(dim=1)  # (out,)

        # ||BA||²_c = (B @ (A@Aᵀ) * B).sum(dim=1)
        aa = lora_A_weight @ lora_A_weight.t()  # (r, r)
        ba_sq = (lora_B_weight @ aa * lora_B_weight).sum(dim=1)  # (out,)

        norm_sq = w_norm_sq + 2.0 * scaling * cross + scaling * scaling * ba_sq
        # clamp to avoid sqrt of tiny negative due to float rounding
        norm_sq = norm_sq.clamp(min=0.0)
        return torch.sqrt(norm_sq)

    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None):
        """Compute ``||W + s*lora_weight||_c`` for merge/unmerge compatibility.

        During merge, ``lora_weight`` is the full delta weight (already materialized as ``B@A``). We compute the
        column-wise norm directly.
        """
        # same as DoraLinearVariant.get_weight_norm
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == "Linear4bit":
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)

            if dtype_is_fp16:
                lora_A = lora_A.half()
                lora_B = lora_B.half()

            # Compute weight norm using low-rank decomposition (float32 for stability)
            weight_f = weight.to(lora_A.device).to(torch.float32)
            a_f = lora_A.to(torch.float32)
            b_f = lora_B.to(torch.float32)
            weight_norm = self.get_weight_norm_lowrank(weight_f, a_f, b_f, float(scaling))
            weight_norm = weight_norm.to(weight.dtype)

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name="default"):
        """DoRA forward using low-rank weight norm.

        Requires ``base_result`` to be provided (the base layer output without bias). If ``base_result`` is ``None``
        (training mode with dropout), we fall back to the standard DoRA path that materializes BA.
        """
        magnitude = self.weight  # (out,)

        if base_result is None:
            # Training mode with dropout: fall back to full materialization.
            return self._forward_full(x, lora_A, lora_B, scaling, base_layer, magnitude)

        # Low-rank kernel path (eval mode or no dropout)
        lora_A_weight = lora_A.weight  # (r, in)
        lora_B_weight = lora_B.weight  # (out, r)

        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)

        # Compute weight norm without materializing BA (float32 for stability)
        with torch.no_grad():
            weight_f = weight.to(torch.float32)
            a_f = lora_A_weight.to(torch.float32)
            b_f = lora_B_weight.to(torch.float32)
            weight_norm = self.get_weight_norm_lowrank(weight_f, a_f, b_f, float(scaling))
        weight_norm = weight_norm.detach().to(x.dtype)

        mag_norm_scale = (magnitude / weight_norm).view(1, -1)  # (1, out)

        lora_result = lora_B(lora_A(x))  # (batch, out)

        # Remove bias from base_result if present, so the DoRA scaling only
        # applies to the directional component
        bias = base_layer.bias
        if bias is not None:
            base_result = base_result - bias

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
        return result_dora

    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        # same as DoraLinearLayer.get_weight_norm
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def _forward_full(self, x, lora_A, lora_B, scaling, base_layer, magnitude):
        """Fallback: materializes full BA (same as DoraLinearLayer.forward)."""
        lora_weight = self.get_lora_weight(lora_A=lora_A, lora_B=lora_B, adapter_name=adapter_name)
        lora_weight = lora_weight.to(x.dtype)

        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        weight_norm = self.get_weight_norm_lowrank(
            weight.to(torch.float32),
            lora_A.weight.to(torch.float32),
            lora_B.weight.to(torch.float32),
            float(scaling),
        )
        weight_norm = weight_norm.detach().to(x.dtype)
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        base_result = F.linear(x, weight)

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora_kernel." + rep
