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
Main entry point to run the experiments. Contains general setup and the adaptation/evaluation code.

This benchmark compares test-time adaptation methods that consume a small set of labeled support examples without
gradient training:

- "faast": the support examples are compiled into FAAST fast weights (closed-form, forward-only)
- "icl":   the support examples are prepended to each query as in-context examples
- "base":  no adaptation, zero-shot baseline

The evaluation code is copied from the MetaMathQA benchmark, with one notable difference: only the generated
continuation is decoded, not the prompt. This is required for in-context learning, since the prompt itself contains
answer strings ("The answer is: ...") that would otherwise be picked up by the answer parser whenever the model
generates no parsable answer of its own.
"""

import argparse
import datetime as dt
import gc
import os
import random
import sys
import textwrap
import time
from typing import Optional

import torch
from data import (
    build_icl_prompt_prefix,
    get_gsm8k_valid_test_datasets,
    get_metamath,
    get_valid_test_questions,
    get_wiki_small,
    sample_support_examples,
    tokenize_support_with_answer_mask,
)
from tqdm import tqdm
from transformers import GenerationConfig, set_seed
from utils import (
    FILE_NAME_RUN_PARAMS,
    RunConfig,
    RunResult,
    RunStatus,
    get_accuracy,
    get_base_model_info,
    get_dataset_info,
    get_file_size,
    get_model,
    get_peft_branch,
    get_run_config,
    get_tokenizer,
    init_accelerator,
    log_results,
    validate_experiment_path,
)

from peft import PeftConfig
from peft.utils import CONFIG_NAME, infer_device


def get_generation_config(*, seq_len, generate_kwargs) -> GenerationConfig:
    # filter out None values so that we don't depend on setting correct defaults in the config
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
    if ("max_length" in generate_kwargs) and ("max_new_tokens" in generate_kwargs):
        # transformers does not support setting both max_length and max_new_tokens, but what we want in this case is to
        # take the smaller of the two values
        new_max_length = min(generate_kwargs["max_new_tokens"] + seq_len, generate_kwargs["max_length"])
        del generate_kwargs["max_new_tokens"]
        generate_kwargs["max_length"] = new_max_length
    generation_config = GenerationConfig(**generate_kwargs)
    return generation_config


def evaluate(model, tokenizer, ds, batch_size, generate_kwargs, use_tqdm: bool = False) -> tuple[list[str], list[str]]:
    generate_kwargs = generate_kwargs.copy()
    generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        predictions = []
        responses = []
        pbar = range(0, len(ds), batch_size)
        if use_tqdm:
            pbar = tqdm(pbar)
        for j in pbar:
            sliced = ds[j : j + batch_size]
            responses += sliced.pop("response")
            batch = tokenizer.pad(sliced, return_tensors="pt", padding_side="left").to(model.device)
            seq_len = batch["input_ids"].shape[1]
            generation_config = get_generation_config(seq_len=seq_len, generate_kwargs=generate_kwargs)
            outputs = model.generate(**batch, generation_config=generation_config)
            # only decode the continuation; the prompt may contain in-context examples whose answers must not be
            # picked up by the answer parser
            outputs = outputs[:, seq_len:]
            predictions += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions, responses


@torch.inference_mode()
def calculate_mean_per_token_loss(model, tokenizer, rows: list[str], batch_size: int, max_length: int) -> float:
    """Calculate the mean loss per token on the given dataset.

    Useful to determine general model performance before and after adaptation to get an estimate of the magnitude of
    'forgetting'. Note that for Wikipedia data, since the information density is quite high, the loss can be
    surprisingly large.

    """
    losses: list[float] = []
    for j in range(0, len(rows), batch_size):
        sliced = rows[j : j + batch_size]
        batch = tokenizer(sliced, truncation=True, max_length=max_length)
        batch = tokenizer.pad(batch, return_tensors="pt", padding_side="left").to(model.device)
        outputs = model(**batch)
        logits = outputs.logits
        for logit, target, mask in zip(logits, batch["input_ids"], batch["attention_mask"]):
            # calculate loss per token so that the mean is not skewed by sequence length of sample, padding from left
            num_tokens = mask.sum()
            token_losses = torch.nn.functional.cross_entropy(
                logit[-num_tokens:], target[-num_tokens:], reduction="none"
            )
            losses.extend(loss.item() for loss in token_losses)
    return torch.tensor(losses).mean().item()


def learn_fast_weights(*, model, tokenizer, support, query_template: str, batch_size: int) -> int:
    """Compile the support examples into the FAAST fast weights, batch by batch. Returns the total token count."""
    rows = tokenize_support_with_answer_mask(support, tokenizer, query_template)
    # sort by length to minimize padding
    rows = sorted(rows, key=lambda row: len(row["input_ids"]))
    total_tokens = 0

    def pad(values, pad_value, max_len):
        return [v + [pad_value] * (max_len - len(v)) for v in values]

    for j in range(0, len(rows), batch_size):
        batch_rows = rows[j : j + batch_size]
        max_len = max(len(row["input_ids"]) for row in batch_rows)
        pad_id = tokenizer.pad_token_id

        input_ids = torch.tensor(pad([row["input_ids"] for row in batch_rows], pad_id, max_len))
        attention_mask = torch.tensor(pad([row["attention_mask"] for row in batch_rows], 0, max_len))
        answer_mask = torch.tensor(pad([row["answer_mask"] for row in batch_rows], 0, max_len))
        device = model.device
        model.base_model.learn(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            answer_mask=answer_mask.to(device),
        )
        total_tokens += int(attention_mask.sum())
    return total_tokens


def run(*, model, tokenizer, run_config: RunConfig) -> RunResult:
    device_type = infer_device()
    torch_accelerator_module = getattr(torch, device_type, torch.cuda)
    metrics: dict = {}
    status = RunStatus.FAILED
    error_msg = ""
    model.eval()

    try:
        # sample the support examples first so that a failure surfaces early
        support = []
        icl_prompt_prefix = ""
        if run_config.mode != "base":
            metamath = get_metamath(print_verbose)
            forbidden_questions = get_valid_test_questions()
            support = sample_support_examples(
                metamath=metamath,
                num_examples=run_config.num_support_examples,
                seed=run_config.support_seed,
                forbidden_questions=forbidden_questions,
                print_fn=print_verbose,
            )
        if run_config.mode == "icl":
            icl_prompt_prefix = build_icl_prompt_prefix(support, run_config.query_template)

        ds_valid, ds_test = get_gsm8k_valid_test_datasets(
            tokenizer=tokenizer,
            query_template=run_config.query_template,
            icl_prompt_prefix=icl_prompt_prefix,
            print_fn=print_verbose,
        )

        # forgetting metric: only relevant for FAAST, since it is the only mode that changes the model
        if run_config.mode == "faast":
            rows_wiki = get_wiki_small()
            wiki_loss_before = calculate_mean_per_token_loss(
                model=model, tokenizer=tokenizer, rows=rows_wiki, batch_size=4, max_length=768
            )

        # adaptation
        tic_adapt = time.perf_counter()
        if run_config.mode == "faast":
            num_learn_tokens = learn_fast_weights(
                model=model,
                tokenizer=tokenizer,
                support=support,
                query_template=run_config.query_template,
                batch_size=run_config.batch_size_learn,
            )
            metrics["adaptation num tokens"] = num_learn_tokens
        adaptation_time = time.perf_counter() - tic_adapt
        metrics["adaptation time"] = adaptation_time
        metrics["num support examples"] = len(support)
        gc.collect()
        torch_accelerator_module.empty_cache()

        # evaluation
        tic_eval = time.perf_counter()
        predictions, responses = evaluate(
            model=model,
            tokenizer=tokenizer,
            ds=ds_valid,
            batch_size=run_config.batch_size_eval,
            generate_kwargs=run_config.generation_kwargs,
        )
        metrics["valid accuracy"] = get_accuracy(predictions=predictions, responses=responses)

        example = random.choice(predictions)
        example = textwrap.shorten(example, width=750)
        example = textwrap.indent(example, "    ")
        print_verbose(f"\nExample prediction:\n{example}\n")
        print_verbose(f"Valid accuracy: {metrics['valid accuracy']:.3f}")

        predictions, responses = evaluate(
            model=model,
            tokenizer=tokenizer,
            ds=ds_test,
            batch_size=run_config.batch_size_eval,
            generate_kwargs=run_config.generation_kwargs,
            use_tqdm=len(ds_test) > 100,
        )
        metrics["test accuracy"] = get_accuracy(predictions=predictions, responses=responses)
        metrics["eval time"] = time.perf_counter() - tic_eval
        print_verbose(f"Test accuracy: {metrics['test accuracy']:.3f}")

        if run_config.mode == "faast":
            wiki_loss_after = calculate_mean_per_token_loss(
                model=model, tokenizer=tokenizer, rows=rows_wiki, batch_size=4, max_length=768
            )
            metrics["forgetting"] = wiki_loss_after - wiki_loss_before

        status = RunStatus.SUCCESS
    except KeyboardInterrupt:
        print_verbose("canceled run")
        status = RunStatus.CANCELED
        error_msg = "manually canceled"
    except torch.OutOfMemoryError as exc:
        print_verbose("out of memory error encountered")
        status = RunStatus.CANCELED
        error_msg = str(exc)
    except Exception as exc:
        print_verbose(f"encountered an error: {exc}")
        status = RunStatus.CANCELED
        error_msg = str(exc)

    return RunResult(status=status, metrics=metrics, error_msg=error_msg)


def main(*, path_experiment: str, experiment_name: str, clean: bool) -> None:
    tic_total = time.perf_counter()
    start_date = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

    peft_branch = get_peft_branch()
    if peft_branch == "main":
        print_verbose("===== This experiment is categorized as a MAIN run because the PEFT branch is 'main' ======")
    else:
        print_verbose(
            f"===== This experiment is categorized as a TEST run because the PEFT branch is '{peft_branch}' ======"
        )

    # load configs
    peft_config: Optional[PeftConfig] = None
    if os.path.exists(os.path.join(path_experiment, CONFIG_NAME)):
        peft_config = PeftConfig.from_pretrained(path_experiment)
    path_run_config = os.path.join(path_experiment, FILE_NAME_RUN_PARAMS)
    run_config = get_run_config(path_run_config)
    if (run_config.mode == "faast") and (peft_config is None):
        raise ValueError("Experiments with mode='faast' require an adapter_config.json")
    if (run_config.mode != "faast") and (peft_config is not None):
        raise ValueError(f"Experiments with mode='{run_config.mode}' must not have an adapter_config.json")
    set_seed(run_config.seed)

    # initialize objects
    accelerator_memory_init = init_accelerator()
    tokenizer = get_tokenizer(model_id=run_config.model_id, max_seq_length=run_config.max_seq_length)

    model_info = get_base_model_info(run_config.model_id)
    metamath_info = get_dataset_info("meta-math/MetaMathQA")
    gsm8k_info = get_dataset_info("openai/gsm8k")
    model = get_model(
        model_id=run_config.model_id,
        dtype=run_config.dtype,
        attn_implementation=run_config.attn_implementation,
        peft_config=peft_config,
        autocast_adapter_dtype=run_config.autocast_adapter_dtype,
    )
    print_verbose(model)

    run_result = run(model=model, tokenizer=tokenizer, run_config=run_config)

    if run_result.status == RunStatus.FAILED:
        print_verbose("Run failed, not logging results")
        sys.exit(1)

    file_size = get_file_size(model, peft_config=peft_config, clean=clean, print_fn=print_verbose)
    run_result.metrics["file size"] = file_size

    time_total = time.perf_counter() - tic_total
    # log results: print and save to file
    log_results(
        experiment_name=experiment_name,
        run_result=run_result,
        accelerator_memory_init=accelerator_memory_init,
        time_total=time_total,
        model_info=model_info,
        datasets_info={"metamath": metamath_info, "gsm8k": gsm8k_info},
        start_date=start_date,
        run_config=run_config,
        peft_config=peft_config,
        print_fn=print_verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("path_experiment", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete artifacts after run finishes (logs are still saved)",
    )
    args = parser.parse_args()

    experiment_name = validate_experiment_path(args.path_experiment)

    if args.verbose:

        def print_verbose(*args, **kwargs) -> None:
            kwargs["file"] = sys.stderr
            print(*args, **kwargs)
    else:

        def print_verbose(*args, **kwargs) -> None:
            pass

    main(
        path_experiment=args.path_experiment,
        experiment_name=experiment_name,
        clean=args.clean,
    )
