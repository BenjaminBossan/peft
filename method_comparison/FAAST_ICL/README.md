# FAAST vs in-context learning on GSM8K

## Goal

This benchmark compares test-time adaptation methods that consume a small set of labeled support examples *without
gradient training*:

- **faast**: The support examples are compiled into FAAST fast weights via the closed-form, forward-only solve (see
  the [FAAST paper](https://arxiv.org/abs/2605.04651) and `peft.FaastConfig`).
- **icl**: The support examples are prepended to each query as few-shot in-context examples.
- **base**: No adaptation, zero-shot baseline.

The setup is derived from the MetaMathQA benchmark (`method_comparison/MetaMathQA`) and re-uses its data filtering,
prompt template, generation settings, and answer parsing, so the accuracies are roughly comparable.

## Dataset

The support examples are sampled from the [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) train
set. Validation/testing is done on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) ("main"), using the same
splits as the MetaMathQA benchmark (validation: fixed random sample of the GSM8K train set, test: the whole GSM8K
test set).

To make the comparison fair, both `faast` and `icl` consume the same `num_support_examples` support examples, sampled
with the same `support_seed`.

### Leakage prevention

MetaMathQA is constructed by augmenting the GSM8K/MATH *train* sets, so the GSM8K test questions can never leak into
the support set. The validation questions, however, are drawn from the GSM8K train set, whose rephrased variants do
occur in MetaMathQA. Support sampling therefore skips every candidate whose `original_question` matches a validation
or test question (see `data.sample_support_examples`).

## Running

Create an experiment under `experiments/<mode>/<experiment-name>`, e.g. `experiments/faast/llama-3.2-3B-default/`,
containing:

- `run_params.json`: overrides for the defaults in `default_run_params.json`; must at least set `"mode"`.
- `adapter_config.json`: the FAAST PEFT config; required for (and only allowed for) `mode="faast"`.

Then either run the whole suite with `make`, or a single experiment with:

```sh
python run.py -v experiments/faast/llama-3.2-3B-default
```

Notes:

- For `mode="icl"`, choose a `max_seq_length` large enough to fit the in-context examples plus the query (each
  support example is up to ~1300 characters). Also prefer setting only `max_new_tokens` (not `max_length`) in
  `generation_kwargs` and a smaller `batch_size_eval`, as the prompts are much longer.
- Unlike the MetaMathQA benchmark, only the generated continuation is decoded and passed to the answer parser. This
  is required for ICL, since the prompt itself contains `The answer is: ...` strings.
- The `forgetting` metric (increase of the mean per-token loss on Wikipedia text) is only computed for `faast`, the
  only mode that modifies the model.

Results are written to `results/` (on the main PEFT branch) or `temporary_results/` (other branches), analogous to
the MetaMathQA benchmark.
