# Copyright 2023-present the HuggingFace Inc. team.
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

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TRANSFORMERS_MODEL_CONFIG


class _BaseAdaptedAttention(nn.Module):
    """Base module, which defines adaption prompts for multiple model types."""

    def __init__(self, model_type: str, adapter_len: int, model, target_dtype=torch.float32):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        if isinstance(model, _BaseAdaptedAttention):
            raise ValueError("Unable to stack multiple adaption prompts")
        super().__init__()
        self.model_type = model_type
        self.model = model
        self.adapter_len = adapter_len
        # Assume all parameters of the attention model we are wrapping are on the same device.

        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)

        if hasattr(self.model, "hidden_size"):
            # TODO: remove this clause after 2026-01-01
            hidden_size = self.model.hidden_size
        else:  # changed in https://github.com/huggingface/transformers/pull/35235
            hidden_size = self.model.config.hidden_size

        if hasattr(self.model, "num_heads"):
            # TODO: remove this clause after 2026-01-01
            self.num_heads = self.model.num_heads
        else:  # changed in https://github.com/huggingface/transformers/pull/35235
            self.num_heads = self.model.config.num_attention_heads

        self.adaption_prompt = nn.Parameter(
            torch.empty(1, adapter_len, hidden_size, device=device, dtype=target_dtype).normal_()
        )
        # Initialize the gate to 0 as this is "zero-init".
        self.adaption_gate = nn.Parameter(torch.zeros(1, device=device, dtype=target_dtype))


class AdaptedAttentionGPT(_BaseAdaptedAttention):
    """This module wraps a GPT2Attention module and injects adaption prompts"""

    def __init__(self, model_type, adapter_len, model):
        target_dtype = (
            model.c_proj.weight.dtype if model.c_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        )
        super().__init__(model_type, adapter_len, model, target_dtype=target_dtype)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        layer_past: Optional[tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        attn_outputs = self.model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        """
        Forward pass for the adapter which wraps the GPT2Attention module
        """

        attn_output = attn_outputs[0]
        add_outputs = attn_outputs[1:]

        c_attn_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer

        bsz = attn_output.shape[0]
        q_len = attn_output.shape[1]
        embed_dim = attn_output.shape[2]

        _, key, value = getattr(self.model, c_attn_layer)(self.adaption_prompt).split(embed_dim, dim=2)

        adapter_k = (
            key.view(1, self.adapter_len, self.num_heads, self.model.head_dim).repeat(bsz, 1, 1, 1).transpose(1, 2)
        )
        adapter_v = (
            value.view(1, self.adapter_len, self.num_heads, self.model.head_dim).repeat(bsz, 1, 1, 1).transpose(1, 2)
        )
        # recompute query state since it is not returned by GPT2 forward
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        query_states = compute_query_states(
            self.model, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
        )

        previous_dtype = query_states.dtype

        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(
            self.model.head_dim
        )
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

        # Add adaption prompt output to original output.
        hidden_state = attn_output + adapter_output

        # Restore original dtype.
        hidden_state = hidden_state.to(previous_dtype)

        # add additional attention outputs (attention and cross attention)
        output = (hidden_state,) + add_outputs
        return output


class AdaptedAttention(_BaseAdaptedAttention):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type, adapter_len, model):
        target_dtype = (
            model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        )
        super().__init__(model_type, adapter_len, model, target_dtype=target_dtype)

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        output, *_ = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        factor = (
            self.model.k_proj.in_features // self.model.k_proj.out_features
        )  # Mistral has different input and output dimension for k_proj and v_proj layers

        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)

        if hasattr(self.model, "num_heads"):
            # TODO: remove this clause after 2026-01-01
            num_heads = self.model.num_heads
        else:  # changed in https://github.com/huggingface/transformers/pull/35235
            num_heads = self.model.config.num_attention_heads
        # (bsz, num_key_value_heads, adapter_len, head_dim)
        adapter_k = (
            key.view(1, self.adapter_len, (num_heads // factor), self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        adapter_v = (
            value.view(1, self.adapter_len, (num_heads // factor), self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        # Below is taken from https://github.com/huggingface/transformers/blob/e547458c43dfdbbb8f6a7757237e234c44e20a8f/src/transformers/models/mistral/modeling_mistral.py#L181
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_k = torch.repeat_interleave(adapter_k, repeats=factor, dim=1)
        adapter_v = torch.repeat_interleave(adapter_v, repeats=factor, dim=1)
        # Recompute query states.
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        # (bsz, num_heads, q_len, head_dim)
        query_states = compute_query_states(model=self.model, **kwargs)

        previous_dtype = query_states.dtype

        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(
            self.model.head_dim
        )
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

        # (bsz, q_len, hidden_size)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Add adaption prompt output to original output.
        output = output + adapter_output

        # Restore original dtype.
        output = output.to(previous_dtype)
        return output, *_
