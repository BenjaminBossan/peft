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

from typing import Optional

import torch
from torch import nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner

from .config import FaastConfig
from .layer import FaastDecoderLayer, FaastLayer, FaastLinear


class FaastModel(BaseTuner):
    """
    Creates a FAAST (Forward-Only Associative Learning via Closed-Form Fast Weights) model from a pretrained
    transformers model.

    The method is described in https://arxiv.org/abs/2605.04651. Unlike gradient-based PEFT methods, the FAAST fast
    weights are not trained with an optimizer. Instead, call [`FaastModel.learn`] with batches of (tokenized) support
    examples; each call updates the fast weights with the closed-form solution over all examples seen so far.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`FaastConfig`]): The configuration of the FAAST model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from peft import FaastConfig, get_peft_model

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        >>> peft_config = FaastConfig(task_type="CAUSAL_LM", inference_mode=True)
        >>> model = get_peft_model(model, peft_config)
        >>> inputs = tokenizer(["some support example ..."], return_tensors="pt")
        >>> model.base_model.learn(**inputs)
        >>> # now generate as usual, the fast weights are active
        ```
    """

    prefix: str = "faast_"
    tuner_layer_cls = FaastLayer
    target_module_mapping = {}

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            num_layers = model_config.get("num_hidden_layers")
            if num_layers is None:
                raise ValueError(
                    "Cannot determine the last decoder layer for this model, please specify `target_modules` in "
                    "`peft_config`."
                )
            # by default, only the last decoder layer hosts the fast weights ("output memory")
            peft_config.target_modules = rf".*\.layers\.{num_layers - 1}$"
        return peft_config

    def _create_and_replace(
        self,
        peft_config: FaastConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if isinstance(target, FaastLayer):
            target.update_layer(adapter_name, config=peft_config)
        elif isinstance(target, nn.Linear):
            # linear output heads (e.g. the final projection of a diffusion transformer); the regression targets
            # must be supplied externally via `begin_learn(values=...)`
            new_module = FaastLinear(target, adapter_name, config=peft_config)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            # assumed to be a decoder layer of a causal language model
            hidden_size = self._get_hidden_size()
            new_module = FaastDecoderLayer(target, adapter_name, config=peft_config, hidden_size=hidden_size)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _get_hidden_size(self) -> int:
        # the keys live in the residual stream and the values in the input embedding space, which have the same
        # dimension for the decoder-only models supported here
        embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise ValueError("Could not determine the hidden size from the model's input embeddings.")
        return embedding.weight.shape[1]

    def _faast_layers(self) -> list[FaastLayer]:
        return [module for module in self.model.modules() if isinstance(module, FaastLayer)]

    def begin_learn(self, **learn_context) -> None:
        """
        Put all FAAST layers into learn mode for externally driven forward passes.

        Until [`FaastModel.end_learn`] is called, every forward pass through the model accumulates key/value
        statistics instead of applying the fast weights. The required context depends on the layer type: layers
        wrapping linear output heads ([`FaastLinear`]) need `values` (the regression target for each input position
        of the wrapped layer, e.g. the flow matching targets of a diffusion model). For causal language models, use
        the [`FaastModel.learn`] convenience method instead.
        """
        layers = self._faast_layers()
        if not layers:
            raise ValueError("No FAAST layers found in the model.")
        for layer in layers:
            layer.begin_learn(**learn_context)

    def end_learn(self, solve: bool = True) -> None:
        """
        Leave learn mode. If `solve` is True, also compute the fast weights from the accumulated statistics; pass
        `solve=False` when learning continues (e.g. with the next batch) and call [`FaastModel.solve_fast_weights`]
        once at the end, as the solve is not free.
        """
        for layer in self._faast_layers():
            layer.end_learn(solve=solve)

    def solve_fast_weights(self) -> None:
        """Compute the fast weights from the statistics accumulated so far (can be called repeatedly)."""
        for layer in self._faast_layers():
            layer.solve_fast_weights()

    @torch.no_grad()
    def learn(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Compile a batch of support examples into the fast weights with a single forward pass (no backpropagation).

        Can be called multiple times; the fast weights always correspond to the exact closed-form solution over all
        examples learned since the last call to [`FaastModel.reset_fast_weights`].

        Args:
            input_ids (`torch.Tensor`):
                Tokenized support examples of shape (batch_size, seq_len).
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask of the support examples; padding positions do not contribute key/value pairs.
            answer_mask (`torch.Tensor`, *optional*):
                Mask of the same shape as `input_ids` that is 1 for tokens belonging to the answer. Only used (and
                required) by adapters configured with `kv_source="answer"`; keys are then restricted to positions
                whose next token is an answer token.
            kwargs (`dict`):
                Additional arguments passed to the model's forward.
        """
        layers = self._faast_layers()
        if not layers:
            raise ValueError("No FAAST layers found in the model.")

        device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        for layer in layers:
            layer.begin_learn(inputs_embeds=inputs_embeds, attention_mask=attention_mask, answer_mask=answer_mask)

        was_training = self.model.training
        self.model.eval()
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
        finally:
            for layer in layers:
                layer.end_learn()
            if was_training:
                self.model.train()

    def reset_fast_weights(self, adapter_name: Optional[str] = None) -> None:
        """Reset the fast weights (and the accumulated statistics) so that the adapter is a no-op again."""
        for layer in self._faast_layers():
            layer.reset_fast_weights(adapter_name)
