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

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class FaastConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FaastModel`].

    FAAST (Forward-Only Associative Learning via Closed-Form Fast Weights for Test-Time Supervised Adaptation,
    https://arxiv.org/abs/2605.04651) compiles labeled support examples into "fast weights" in a single forward pass,
    without backpropagation. The fast weight matrix is the closed-form least-squares solution mapping the hidden
    states of the targeted decoder layer at position t (keys) to the input embedding of the token at position t+1
    (values). At inference, the layer output is interpolated towards the fast-weight prediction, nudging the final
    hidden states towards the embedding of the most associated next token (which raises its logit via the tied LM
    head).

    Note: This first implementation only covers the training-free "output memory" of the paper's reference
    implementation, i.e. it should be applied to the last decoder layer (the default when `target_modules` is not
    set). The middle-layer memories require a backprop-trained readout projection and are not implemented.

    Besides decoder layers of causal language models, FAAST can also target `nn.Linear` output heads, e.g. the final
    projection (`proj_out`) of a diffusion transformer. In that case the regression targets are not derived from the
    inputs but must be supplied externally while learning, see `FaastModel.begin_learn`.

    Args:
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply FAAST to: either decoder layer modules of a causal language model or
            `nn.Linear` output heads. If not specified, the last decoder layer is targeted (requires the model config
            to have `num_hidden_layers`). Note that this method was only tested with the Llama and FLUX.2
            architectures, targeting modules of other architectures may not work as expected.
        r (`int`, *optional*):
            If set, only the `r` largest singular directions of the key matrix are used in the closed-form solve and
            the fast weights are stored as two low-rank factors (LoRA-like), reducing memory and compute. If `None`,
            the full solution is used, subject to spectral filtering (see `filter_alpha`).
        filter_alpha (`float`):
            Spectral filtering strength. Singular values of the key matrix below `sigma_max * epsilon` with
            `epsilon = 1 / N**filter_alpha` (N = number of key/value pairs) are discarded in the pseudoinverse.
            Defaults to 1.0 as in the paper.
        memory_weight (`float`):
            Interpolation weight of the fast-weight prediction: the layer output becomes
            `(1 - memory_weight) * hidden_states + memory_weight * prediction`. Defaults to 0.9 following the
            reference implementation. Values > 1 are allowed and extrapolate beyond the prediction, i.e. they amplify
            the learned correction `prediction - hidden_states` (guidance-style), typically at the cost of output
            quality.
        preserve_output_scale (`bool`):
            Only supported when targeting linear output heads. If True, the adapted output is rescaled per token to
            the norm of the base layer's prediction, so the fast weights only change the direction of the output,
            never its magnitude. This counteracts the systematic shrinkage of the least-squares prediction relative
            to the true targets (the unexplained residual energy is missing from it), which otherwise scales down the
            output, increasingly so for larger `memory_weight`. Defaults to False.
        kv_source (`Literal["all", "answer"]`):
            Which token positions of the support examples contribute key/value pairs. `"all"` uses every position
            (like the paper's language modeling setup), `"answer"` only uses positions whose *next* token is part of
            the answer, which requires passing `answer_mask` to `FaastModel.learn`. Only relevant when targeting
            decoder layers of a causal language model.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the decoder layer modules to replace with FAAST. If not "
                "specified, the last decoder layer is targeted."
            )
        },
    )
    r: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, keep only the r largest singular directions in the closed-form solve and store the fast "
                "weights as two low-rank factors."
            )
        },
    )
    filter_alpha: float = field(
        default=1.0,
        metadata={
            "help": (
                "Spectral filtering strength: singular values below sigma_max / N**filter_alpha are discarded in the "
                "pseudoinverse."
            )
        },
    )
    memory_weight: float = field(
        default=0.9,
        metadata={
            "help": ("Interpolation weight of the fast-weight prediction in the output of the targeted decoder layer.")
        },
    )
    preserve_output_scale: bool = field(
        default=False,
        metadata={
            "help": (
                "Rescale the adapted output per token to the norm of the base layer's prediction, so that the fast "
                "weights only change the direction of the output. Only supported when targeting linear output heads."
            )
        },
    )
    kv_source: Literal["all", "answer"] = field(
        default="all",
        metadata={
            "help": (
                "Which token positions of the support examples contribute key/value pairs: 'all' positions or only "
                "positions whose next token belongs to the 'answer'."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.FAAST
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        if (self.r is not None) and (self.r <= 0):
            raise ValueError(f"`r` must be a positive integer or None, got {self.r}.")
        if self.memory_weight <= 0.0:
            raise ValueError(f"`memory_weight` must be positive, got {self.memory_weight}.")
        if self.filter_alpha < 0:
            raise ValueError(f"`filter_alpha` must be non-negative, got {self.filter_alpha}.")
        if self.kv_source not in ("all", "answer"):
            raise ValueError(f"`kv_source` must be 'all' or 'answer', got {self.kv_source}.")
