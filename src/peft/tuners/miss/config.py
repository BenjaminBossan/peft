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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MissConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`MiSSModel`].

    Args:
        r (`int`):
            The rank of MiSS across different layers. It is best to set 'r' to an even number; otherwise, the default
            initialization method will not work. The rank of MiSS corresponds to a low-rank decomposition along the
            in_features dimension.
        miss_dropout (`float`):
            The dropout probability for MiSS layers.
        mini_r (`int`):
            The rank of MiSS corresponds to a low-rank decomposition along the out_features dimension. When you set
            `init_weights=mini`, you need to set `mini_r`. Please make sure that `out_features` is divisible by
            `mini_r`.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen, excluding
            the output layer. If this is not specified, modules will be chosen according to the model architecture. If
            the architecture is not known, an error will be raised -- in this case, you should specify the target
            modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        init_weights (bool | Literal["bat", "mini"]):
            Different initializations correspond to different MiSS variants. By default(balance), the most efficient
            and general method in MiSS will be used. 'bat': In this mode, you can enable nonlinear updates across
            different shards. 'mini': In this mode, you can set a smaller rank to use fewer trainable parameters, but
            it is recommended to keep `out_features % mini_r == 0`.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(
        default=64,
        metadata={
            "help": "The rank of MiSS corresponds to a low-rank decomposition along the in_features dimension.",
            "note": "It is best to set 'r' to an even number; otherwise, the default initialization method will not work.",
        },
    )
    miss_dropout: float = field(default=0.0, metadata={"help": "MiSS dropout"})
    mini_r: int = field(
        default=1,
        metadata={
            "help": "The rank of MiSS corresponds to a low-rank decomposition along the out_features dimension.",
            "note": "It is recommended that mini_r be divisible by out_features. When mini_r == out_features, the mini method is equivalent to the default efficient MiSS.",
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with MiSS.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from MiSS."},
    )
    init_weights: bool | Literal["bat", "mini"] = field(
        default=True,
        metadata={
            "help": (
                "True -> MiSS balance; `bat` -> Bat; `mini` -> smaller rank and efficiency"
                "Whether to initialize the weights of the MiSS layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type for MiSS. Can be 'none', 'all' or 'MiSS_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from MiSS layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MISS
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
