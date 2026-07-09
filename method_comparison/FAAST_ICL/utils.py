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

"""
All utilities not related to data handling.

Mostly copied from the MetaMathQA benchmark, with the training-related parts removed and the train config replaced by
a run config with FAAST/ICL specific options.
"""

import enum
import json
import os
import platform
import subprocess
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any, Literal, Optional

import datasets
import huggingface_hub
import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

import peft
from peft import PeftConfig, get_peft_model
from peft.utils import SAFETENSORS_WEIGHTS_NAME, infer_device


device = infer_device()

if device not in ["cuda", "xpu"]:
    raise RuntimeError("CUDA or XPU is not available, currently only CUDA or XPU is supported")

ACCELERATOR_MEMORY_INIT_THRESHOLD = 500 * 2**20  # 500MB
FILE_NAME_DEFAULT_RUN_PARAMS = os.path.join(os.path.dirname(__file__), "default_run_params.json")
FILE_NAME_RUN_PARAMS = "run_params.json"  # specific params for this experiment
# main results
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
# testing results
RESULT_PATH_TEST = os.path.join(os.path.dirname(__file__), "temporary_results")
# cancelled results
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")
hf_api = huggingface_hub.HfApi()


@dataclass
class RunConfig:
    """All configuration parameters associated with running an experiment

    Args:
        model_id: The model identifier
        mode: The type of adaptation, one of "faast" (fast weights), "icl" (in-context learning), "base" (no
            adaptation, zero-shot baseline)
        num_support_examples: The number of labeled support examples the method may use (ignored for mode="base")
        support_seed: The random seed used to sample the support examples
        dtype: The data type to use for the model
        max_seq_length: The maximum sequence length (for mode="icl", ensure that this is large enough to fit the
            in-context examples plus the query)
        batch_size_learn: The batch size for FAAST learning (forward passes over the support examples)
        batch_size_eval: The batch size for eval/test
        query_template: The template for the query
        seed: The random seed
        autocast_adapter_dtype: Whether to cast adapter dtype to float32, same argument as in PEFT
        generation_kwargs: Arguments passed to transformers GenerationConfig (used in evaluation)
        attn_implementation: The attention implementation to use (if any), see transformers docs
    """

    model_id: str
    mode: Literal["faast", "icl", "base"]
    num_support_examples: int
    support_seed: int
    dtype: Literal["float32", "float16", "bfloat16"]
    max_seq_length: int
    batch_size_learn: int
    batch_size_eval: int
    query_template: str
    seed: int
    autocast_adapter_dtype: bool
    generation_kwargs: dict[str, Any]
    attn_implementation: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str):
            raise TypeError(f"Invalid model_id: {self.model_id}")
        if self.mode not in ["faast", "icl", "base"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.max_seq_length < 0:
            raise ValueError(f"Invalid max_seq_length: {self.max_seq_length}")
        if self.num_support_examples < 0:
            raise ValueError(f"Invalid num_support_examples: {self.num_support_examples}")
        if self.batch_size_learn <= 0:
            raise ValueError(f"Invalid batch_size_learn: {self.batch_size_learn}")
        if self.batch_size_eval <= 0:
            raise ValueError(f"Invalid eval batch_size: {self.batch_size_eval}")
        if "{query}" not in self.query_template:
            raise ValueError("Invalid query_template, must contain '{query}'")


def validate_experiment_path(path: str) -> str:
    # the experiment path should take the form of ./experiments/<mode>/<experiment-name>
    # e.g. ./experiments/faast/llama-3.2-3B-default
    # it should contain:
    # - adapter_config.json (only for mode="faast")
    # - optional: run_params.json
    if not os.path.exists(FILE_NAME_DEFAULT_RUN_PARAMS):
        raise FileNotFoundError(
            f"Missing default run params file '{FILE_NAME_DEFAULT_RUN_PARAMS}' in the ./experiments directory"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")

    # check path structure
    path_parts = path.rstrip(os.path.sep).split(os.path.sep)
    if (len(path_parts) != 3) or (path_parts[-3] != "experiments"):
        raise ValueError(
            f"Path {path} does not have the correct structure, should be ./experiments/<mode>/<experiment-name>"
        )

    experiment_name = os.path.join(*path_parts[-2:])
    return experiment_name


def get_run_config(path: str) -> RunConfig:
    # first, load the default params, then update with experiment-specific params
    with open(FILE_NAME_DEFAULT_RUN_PARAMS) as f:
        default_config_kwargs = json.load(f)

    config_kwargs = {}
    if os.path.exists(path):
        with open(path) as f:
            config_kwargs = json.load(f)

    config_kwargs = {**default_config_kwargs, **config_kwargs}
    return RunConfig(**config_kwargs)


def init_accelerator() -> int:
    torch_accelerator_module = getattr(torch, device, torch.cuda)
    torch.manual_seed(0)
    torch_accelerator_module.reset_peak_memory_stats()
    torch_accelerator_module.manual_seed_all(0)
    # might not be necessary, but just to be sure
    nn.Linear(1, 1).to(device)

    accelerator_memory_init = torch_accelerator_module.max_memory_reserved()
    if accelerator_memory_init > ACCELERATOR_MEMORY_INIT_THRESHOLD:
        raise RuntimeError(
            f"{device} memory usage at start is too high: {accelerator_memory_init // 2**20}MB, please ensure that no other "
            f"processes are running on {device}."
        )

    torch_accelerator_module.reset_peak_memory_stats()
    accelerator_memory_init = torch_accelerator_module.max_memory_reserved()
    return accelerator_memory_init


def get_tokenizer(*, model_id: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = max_seq_length
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_base_model(
    *,
    model_id: str,
    dtype: Literal["float32", "float16", "bfloat16"],
    attn_implementation: Optional[str],
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device,
        "attn_implementation": attn_implementation,
    }
    if dtype == "bfloat16":
        kwargs["dtype"] = torch.bfloat16
    elif dtype == "float16":
        kwargs["dtype"] = torch.float16
    elif dtype != "float32":
        raise ValueError(f"Invalid dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    return model


def get_model(
    *,
    model_id: str,
    dtype: Literal["float32", "float16", "bfloat16"],
    attn_implementation: Optional[str],
    peft_config: Optional[PeftConfig],
    autocast_adapter_dtype: bool,
) -> nn.Module:
    base_model = get_base_model(model_id=model_id, dtype=dtype, attn_implementation=attn_implementation)
    if peft_config is None:
        model = base_model
    else:
        model = get_peft_model(base_model, peft_config, autocast_adapter_dtype=autocast_adapter_dtype)
    return model


def get_file_size(
    model: nn.Module, *, peft_config: Optional[PeftConfig], clean: bool, print_fn: Callable[..., None]
) -> int:
    file_size = 0
    if peft_config is not None:
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True, delete=clean) as tmp_dir:
                model.save_pretrained(tmp_dir)
                stat = os.stat(os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
                file_size = stat.st_size
                if not clean:
                    print_fn(f"Saved PEFT checkpoint to {tmp_dir}")
        except Exception as exc:
            print(f"Failed to save PEFT checkpoint due to the following error: {exc}")
    return file_size


##################
# ANSWER PARSING #
##################


def parse_answer(text: str) -> Optional[str]:
    """
    A label/prediction can look like this:

    Question: If the magnitude of vector v is equal to 4, what is the dot product of vector v with itself?. Think step
    by step
    Answer: The dot product of a vector with itself is equal to the square of its magnitude. So, the dot product of
    vector v with itself is equal to $4^2 = \boxed{16}$.The answer is: 16

    We want to extract '16' from this string.

    """
    # This implementation is based on sampling meta-llama/Llama-3.1-8B-Instruct. It may not work for other models.
    candidate_delimiters = [
        # MetaMath:
        "The answer is: ",
        "The answer is ",
        "The final answer is: ",
        "The final answer is ",
        # GSM8K:
        "#### ",
    ]
    text = text.strip()
    text = text.rstrip(".!?")
    for delimiter in candidate_delimiters:
        if delimiter in text:
            break
    else:  # no match
        return None

    text = text.rpartition(delimiter)[-1].strip()
    # if a new paragraph follows after the final answer, we want to remove it
    text = text.split("\n", 1)[0]
    # note: we can just remove % here since the GSM8K dataset just omits it, i.e. 50% -> 50, no need to divide by 100
    text = text.strip(" .!?$%")
    return text


def convert_to_decimal(s: Optional[str]) -> Optional[Decimal]:
    """
    Converts a string representing a number to a Decimal.

    The string may be:
      - A simple number (e.g., "13", "65.33")
      - A fraction (e.g., "20/14")
    """
    if s is None:
        return None

    try:
        s = s.strip()
        # Check if the string represents a fraction.
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                return None
            numerator = Decimal(parts[0].strip())
            denominator = Decimal(parts[1].strip())
            if denominator == 0:
                return None
            value = numerator / denominator
        else:
            # Parse as a regular decimal or integer string.
            value = Decimal(s)
        return value
    except (DivisionByZero, InvalidOperation, ValueError):
        return None


def get_accuracy(*, predictions: list[str], responses: list[str]) -> float:
    if len(predictions) != len(responses):
        raise ValueError(f"Prediction length mismatch: {len(predictions)} != {len(responses)}")

    y_true: list[str | float | None] = []
    y_pred: list[str | float | None] = []

    for prediction, response in zip(predictions, responses):
        parsed_prediction = parse_answer(prediction)
        parsed_response = parse_answer(response)
        if parsed_response is None:
            raise ValueError(f"Error encountered while trying to parse response: {response}")

        decimal_prediction = convert_to_decimal(parsed_prediction)
        decimal_answer = convert_to_decimal(parsed_response)
        if decimal_prediction is not None:
            y_pred.append(float(decimal_prediction))
        elif parsed_prediction is not None:
            y_pred.append(parsed_prediction)
        else:
            y_pred.append(None)

        # we convert decimals to float so that stuff like this works:
        # float(convert_to_decimal('20/35')) == float(convert_to_decimal('0.5714285714285714'))
        if decimal_answer is not None:
            y_true.append(float(decimal_answer))
        elif parsed_prediction is not None:
            y_true.append(parsed_response)
        else:
            y_true.append(None)

    correct: list[bool] = []
    for true, pred in zip(y_true, y_pred):
        if (true is not None) and (pred is not None):
            correct.append(true == pred)
        else:
            correct.append(False)

    accuracy = sum(correct) / len(correct)
    return accuracy


###########
# LOGGING #
###########


def get_base_model_info(model_id: str) -> Optional[huggingface_hub.ModelInfo]:
    try:
        return hf_api.model_info(model_id)
    except Exception as exc:
        warnings.warn(f"Could not retrieve model info, failed with error {exc}")
        return None


def get_dataset_info(dataset_id: str) -> Optional[huggingface_hub.DatasetInfo]:
    try:
        return hf_api.dataset_info(dataset_id)
    except Exception as exc:
        warnings.warn(f"Could not retrieve dataset info, failed with error {exc}")
        return None


def get_git_hash(module) -> Optional[str]:
    if "site-packages" in module.__path__[0]:
        return None

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(module.__file__)).decode().strip()


def get_package_info() -> dict[str, Optional[str]]:
    """Get the package versions and commit hashes of transformers, peft, datasets, and torch"""
    package_info = {
        "transformers-version": transformers.__version__,
        "transformers-commit-hash": get_git_hash(transformers),
        "peft-version": peft.__version__,
        "peft-commit-hash": get_git_hash(peft),
        "datasets-version": datasets.__version__,
        "datasets-commit-hash": get_git_hash(datasets),
        "torch-version": torch.__version__,
        "torch-commit-hash": get_git_hash(torch),
    }
    return package_info


def get_system_info() -> dict[str, str]:
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "accelerator": getattr(torch, device, torch.cuda).get_device_name(0),
    }
    return system_info


@dataclass
class MetaInfo:
    package_info: dict[str, Optional[str]]
    system_info: dict[str, str]
    pytorch_info: str


def get_meta_info() -> MetaInfo:
    meta_info = MetaInfo(
        package_info=get_package_info(),
        system_info=get_system_info(),
        pytorch_info=torch.__config__.show(),
    )
    return meta_info


def get_peft_branch() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=os.path.dirname(peft.__file__))
        .decode()
        .strip()
    )


class RunStatus(enum.Enum):
    FAILED = "failed"
    SUCCESS = "success"
    CANCELED = "canceled"


@dataclass
class RunResult:
    status: RunStatus
    metrics: dict[str, Any]
    error_msg: str


def log_to_console(log_data: dict[str, Any], print_fn: Callable[..., None]) -> None:
    metrics = log_data["run_info"]["metrics"]
    for key in ["valid accuracy", "test accuracy", "adaptation time", "eval time", "file size"]:
        if key in metrics:
            print_fn(f"{key}: {metrics[key]}")
    accelerator_memory_max = log_data["run_info"]["accelerator_memory_max"]
    print_fn(f"accelerator memory max: {accelerator_memory_max // 2**20}MB")


def log_to_file(
    *, log_data: dict, save_dir: str, experiment_name: str, timestamp: str, print_fn: Callable[..., None]
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if save_dir.endswith(RESULT_PATH):
        file_name = f"{experiment_name.replace(os.path.sep, '--')}.json"
    else:
        # For cancelled and temporary runs, we want to include the timestamp, as these runs are not tracked in git, thus
        # we need unique names to avoid losing history.
        file_name = f"{experiment_name.replace(os.path.sep, '--')}--{timestamp.replace(':', '-')}.json"
    file_name = os.path.join(save_dir, file_name)
    with open(file_name, "w") as f:
        json.dump(log_data, f, indent=2)
    print_fn(f"Saved log to: {file_name}")


def log_results(
    *,
    experiment_name: str,
    run_result: RunResult,
    accelerator_memory_init: int,
    time_total: float,
    model_info: Optional[huggingface_hub.ModelInfo],
    datasets_info: dict[str, Optional[huggingface_hub.DatasetInfo]],
    start_date: str,
    run_config: RunConfig,
    peft_config: Optional[PeftConfig],
    print_fn: Callable[..., None],
) -> None:
    # collect results
    torch_accelerator_module = getattr(torch, device, torch.cuda)
    accelerator_memory_final = torch_accelerator_module.max_memory_reserved()

    meta_info = get_meta_info()
    if model_info is not None:
        model_sha = model_info.sha
        model_created_at = model_info.created_at.isoformat()
    else:
        model_sha = None
        model_created_at = None

    dataset_info_log = {}
    for key, dataset_info in datasets_info.items():
        if dataset_info is not None:
            dataset_sha = dataset_info.sha
            dataset_created_at = dataset_info.created_at.isoformat()
        else:
            dataset_sha = None
            dataset_created_at = None
        dataset_info_log[key] = {"sha": dataset_sha, "created_at": dataset_created_at}

    peft_branch = get_peft_branch()

    if run_result.status == RunStatus.CANCELED:
        save_dir = RESULT_PATH_CANCELLED
        print_fn("Experiment run was categorized as canceled")
    elif peft_branch != "main":
        save_dir = RESULT_PATH_TEST
        print_fn(f"Experiment run was categorized as a test run on branch {peft_branch}")
    elif run_result.status == RunStatus.SUCCESS:
        save_dir = RESULT_PATH
        print_fn("Experiment run was categorized as successful run")
    else:
        save_dir = tempfile.mkdtemp()
        print_fn(f"Experiment could not be categorized, writing results to {save_dir}. Please open an issue on PEFT.")

    if peft_config is None:
        peft_config_dict: Optional[dict[str, Any]] = None
    else:
        peft_config_dict = peft_config.to_dict()
        for key, value in peft_config_dict.items():
            if isinstance(value, set):
                peft_config_dict[key] = list(value)

    log_data = {
        "run_info": {
            "created_at": start_date,
            "total_time": time_total,
            "experiment_name": experiment_name,
            "peft_branch": peft_branch,
            "run_config": asdict(run_config),
            "peft_config": peft_config_dict,
            "error_msg": run_result.error_msg,
            "status": run_result.status.value,
            "metrics": run_result.metrics,
            "accelerator_memory_max": accelerator_memory_final - accelerator_memory_init,
        },
        "meta_info": {
            "model_info": {"sha": model_sha, "created_at": model_created_at},
            "dataset_info": dataset_info_log,
            **asdict(meta_info),
        },
    }

    log_to_console(log_data, print_fn=print)  # use normal print to be able to redirect if so desired
    log_to_file(
        log_data=log_data, save_dir=save_dir, experiment_name=experiment_name, timestamp=start_date, print_fn=print_fn
    )
