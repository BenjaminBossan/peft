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
All utilities related to data handling.

Mostly copied from the MetaMathQA benchmark. The main additions are the support set sampling (with a guard against
leaking validation/test questions) and the tokenization helpers for FAAST learning and for in-context learning.
"""

from collections.abc import Callable
from functools import partial

import datasets
import numpy as np
from datasets import Dataset, load_dataset


# with a token limit of 768 for query + response, we have to exclude all texts with length > 1304; this leaves 93.8% of
# the dataset
CHAR_LIMIT = 1300
# train/valid/test split -- note that evaluation takes quite long, so don't choose too large sizes for the valid set,
# since it's run multiple times during training; test is only run once at the end and thus can be larger
TEST_SIZE = 100
VALID_SIZE = 10


def get_filtered_dataset(*, ds: datasets.Dataset, print_fn: Callable[..., None]) -> Dataset:
    """Return the filtered dataset, with long queries removed.

    We determined that 99% of queries have 529 or fewer characters. Characters roughly correspond to tokens, so this is
    a good proxy. We cannot use tokens directly, as that depends on the tokenizer, which can be different for each
    model, but we want the same filter for each model.

    """
    char_lengths = [len(f"{q} {r}") for q, r in zip(ds["query"], ds["response"])]
    idx_filtered = [i for i, length in enumerate(char_lengths) if length <= CHAR_LIMIT]
    print_fn(f"Filtered dataset: {100 * len(idx_filtered) / len(ds):.1f}% of the original dataset")
    return ds.select(idx_filtered)


def _normalize_question(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def get_metamath(print_fn: Callable[..., None]) -> Dataset:
    metamath = load_dataset("meta-math/MetaMathQA")["train"]
    metamath = get_filtered_dataset(ds=metamath, print_fn=print_fn)
    return metamath


def get_gsm8k_valid_test_datasets(
    *, tokenizer, query_template: str, icl_prompt_prefix: str, print_fn: Callable[..., None]
) -> tuple[Dataset, Dataset]:
    """Return the tokenized GSM8K validation and test sets.

    Same splits as in the MetaMathQA benchmark: the validation set is a fixed random sample of the GSM8K train set,
    the test set is the whole GSM8K test set. `icl_prompt_prefix` is prepended to each query (pass an empty string for
    no in-context examples).
    """
    gsm8k = load_dataset("openai/gsm8k", "main")
    gsm8k = gsm8k.rename_columns({"question": "query", "answer": "response"})
    gsm8k_train = gsm8k["train"]
    gsm8k_test = gsm8k["test"]

    np.random.seed(0)
    indices = np.arange(len(gsm8k_train))
    np.random.shuffle(indices)
    idx_valid = indices[:VALID_SIZE]
    idx_test = np.arange(TEST_SIZE)

    ds_valid = gsm8k_train.select(idx_valid)
    ds_test = gsm8k_test.select(idx_test)

    print_fn(f"Valid size: {len(ds_valid)}")
    print_fn(f"Test size: {len(ds_test)}")

    tokenize_ = partial(
        tokenize_wo_answer, tokenizer=tokenizer, template=query_template, prompt_prefix=icl_prompt_prefix
    )
    ds_valid = ds_valid.map(tokenize_, batched=True).remove_columns(["query"])
    ds_test = ds_test.map(tokenize_, batched=True).remove_columns(["query"])

    return ds_valid, ds_test


def get_valid_test_questions() -> set[str]:
    """Return the normalized questions of the validation and test sets, used to prevent leakage of eval questions
    into the support set."""
    gsm8k = load_dataset("openai/gsm8k", "main")
    questions = list(gsm8k["train"]["question"]) + list(gsm8k["test"]["question"])
    return {_normalize_question(q) for q in questions}


def sample_support_examples(
    *, metamath: Dataset, num_examples: int, seed: int, forbidden_questions: set[str], print_fn: Callable[..., None]
) -> list[dict[str, str]]:
    """Sample support examples from the (filtered) MetaMathQA train set.

    MetaMathQA is built by augmenting the GSM8K and MATH train sets, so the GSM8K *test* questions can never leak
    into the support set. The validation set, however, is drawn from the GSM8K *train* set, whose (rephrased)
    questions do occur in MetaMathQA. To be safe, all candidates whose original question matches a validation or test
    question are skipped.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(metamath))
    support: list[dict[str, str]] = []
    num_skipped = 0
    original_questions = metamath["original_question"]
    for idx in indices:
        idx = int(idx)
        if _normalize_question(original_questions[idx]) in forbidden_questions:
            num_skipped += 1
            continue
        row = metamath[idx]
        support.append({"query": row["query"], "response": row["response"]})
        if len(support) == num_examples:
            break
    else:
        raise ValueError(f"Could not sample {num_examples} support examples from the dataset.")

    print_fn(f"Sampled {len(support)} support examples, skipped {num_skipped} that occur in the valid/test sets")
    return support


def build_icl_prompt_prefix(support: list[dict[str, str]], template: str) -> str:
    """Concatenate the support examples into a few-shot prompt prefix.

    Each example is formatted exactly like the training samples of the MetaMathQA benchmark (template + response,
    without separator). Note that the response is never passed through `str.format`, since MetaMathQA responses can
    contain curly braces (e.g. from LaTeX).
    """
    shots = [template.format(query=example["query"]) + example["response"] for example in support]
    return "\n\n".join(shots) + "\n\n"


def tokenize_support_with_answer_mask(
    support: list[dict[str, str]], tokenizer, template: str
) -> list[dict[str, list[int]]]:
    """Tokenize the support examples for FAAST learning.

    In addition to the usual input_ids/attention_mask, an answer_mask is returned that is 1 for the response tokens
    (including the final EOS), which is required for `FaastConfig(kv_source="answer")`. An EOS token is appended so
    that the fast weights also associate the end of an answer with stopping.
    """
    rows = []
    for example in support:
        prompt = template.format(query=example["query"])
        prompt_ids = tokenizer(prompt)["input_ids"]
        input_ids = tokenizer(prompt + example["response"])["input_ids"]
        input_ids = input_ids + [tokenizer.eos_token_id]
        input_ids = input_ids[: tokenizer.model_max_length]
        answer_mask = [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))
        answer_mask = answer_mask[: len(input_ids)]
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "answer_mask": answer_mask,
            }
        )
    return rows


def tokenize_wo_answer(samples, tokenizer, template, prompt_prefix=""):
    queries = [prompt_prefix + template.format(query=sample) for sample in samples["query"]]
    tokenized = tokenizer(queries)
    tokenized["input_ids"] = [input_ids[: tokenizer.model_max_length] for input_ids in tokenized["input_ids"]]
    tokenized["attention_mask"] = [
        input_ids[: tokenizer.model_max_length] for input_ids in tokenized["attention_mask"]
    ]
    return tokenized


def get_wiki_small(num_samples: int = 100) -> list[str]:
    # This way of loading the dataset avoid having to download whole shards
    ds = load_dataset("HuggingFaceFW/finewiki", split="train", streaming=True)
    dataset_head = ds.take(num_samples)
    rows = [row["text"] for row in dataset_head]
    return rows
