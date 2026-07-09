# FAAST vs gradient fine-tuning on a DreamBooth-style image generation task

## Goal

This benchmark compares FAAST (forward-only, closed-form fast weights; see the
[FAAST paper](https://arxiv.org/abs/2605.04651) and `peft.FaastConfig`) against gradient-trained PEFT methods like
LoRA on the DreamBooth-style task of [`method_comparison/image-gen`](../image-gen). It is a copy of that benchmark
with one addition: experiments whose `adapter_config.json` is a FAAST config skip the optimizer entirely.

## How FAAST is applied to FLUX.2

FAAST wraps the final output projection of the transformer (`proj_out`), which conveniently only sees the image
tokens (timestep-modulated by the preceding AdaLN). During "training" (which consists of forward passes only), the
same data pipeline as for gradient training is used — identical latents, prompts, noise and timestep sampling — and
the FAAST layer accumulates the statistics of the least-squares problem mapping its input hidden states to the packed
flow matching targets (`noise - latents`). The closed-form solve happens once before each evaluation. At inference,
the velocity prediction is interpolated towards the fast-weight prediction:

```
output = (1 - memory_weight) * proj_out(h) + memory_weight * (h @ W^T)
```

`max_steps` doubles as the FAAST learn budget (number of forward-only batches), so LoRA and FAAST see the same data
by default. The `memory_weight` interpolation strength is the main hyperparameter, hence the sweep in
`experiments/faast/`. Values > 1 extrapolate beyond the fast-weight prediction, amplifying the learned correction.
The run log prints the number of learned pairs and the fraction of the target energy explained by the fast weights
on the learning data (a diagnostic for how attenuated the least-squares prediction is; see `preserve_output_scale`).

## Running

Same as the image-gen benchmark:

```sh
python run.py -v experiments/faast/flux2-klein-mw0.3/
make          # run all pending experiments
make list     # list them
```

Metrics (DINOv2 similarity, drift, runtime, memory, checkpoint size), result handling, and sample image generation
are identical to `method_comparison/image-gen`, so numbers are directly comparable. Note that for FAAST, the logged
train loss is the flow matching loss of the *frozen base model* (the fast weights are inactive during the learning
forward passes); it is only a reference value and not expected to decrease.
