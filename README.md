# efficient-cnn-quantize-demo

> **OSS / learning demo only.** This is a personal open-source teaching project by [vk-ai](https://github.com/vk-ai). It is **not** employer production software, is **not** affiliated with any employer (including Lowe's or any other company), and must not be described as production edge-vision or quantization infrastructure.

CPU-friendly **tiny efficient CNN** (depthwise-separable block) on **fully synthetic** blob images, plus **int8 fake-quant** (affine scale + zero-point) and **magnitude pruning**, with a before/after eval harness (loss, accuracy, param/FLOP/size stats). No Hugging Face downloads, no GPU, numpy only.

Driven by **Plumerai-style efficient / edge vision learning fundamentals** — small models, cheap ops, and compression toys you can read end-to-end.

## Why

Edge and efficient-vision stacks are great, but for learning and CI you often want:

- **No multi-GB model downloads**
- **Seconds on CPU**, not minutes on GPU
- Explicit **quantize / prune** knobs and a pytest gate on size + accuracy

This repo is that slice.

## What’s inside

| Piece | Role |
|---|---|
| `src/efficient_cnn/layers.py` | numpy conv / depthwise / pointwise / GAP / CE |
| `src/efficient_cnn/model.py` | Tiny stem → DW+PW → GAP → linear CNN |
| `src/efficient_cnn/data.py` | Synthetic quadrant-blob images (no network) |
| `src/efficient_cnn/train.py` | Minibatch SGD with analytical backprop |
| `src/efficient_cnn/quantize.py` | PTQ int8 fake-quant + minmax/percentile observers + skip-list + `per_tensor`/`per_channel` granularity + per-layer error table |
| `src/efficient_cnn/distill.py` | Teacher→student KD (temperature softmax + CE mix) |
| `src/efficient_cnn/qat.py` | Fake-quant QAT: prepare → few steps → convert (vs PTQ) |
| `src/efficient_cnn/prune.py` | Global-per-tensor magnitude pruning |
| `src/efficient_cnn/mixed_precision.py` | Round 5: sensitivity-driven per-layer float/int8/int4/int2 + suggested skip list + Pareto table |
| `src/efficient_cnn/eval.py` | Float vs quant vs prune vs **prune→int8 compose** harness |
| `configs/default.yaml` | Width, bits, sparsity, train knobs |
| `evals/runner.py` | CI-friendly eval CLI + JSON report |
| `tests/` | Unit + train + harness (pytest, seconds on CPU) |
| `ci/github-actions.yml` | GitHub Actions workflow (copy to `.github/workflows/ci.yml` to enable CI) |

## Quickstart

```bash
git clone https://github.com/vk-ai/efficient-cnn-quantize-demo.git
cd efficient-cnn-quantize-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
python evals/runner.py
python -m efficient_cnn
```

Example output:

```text
Efficient CNN quantize/prune eval
  baseline  loss=0.xx  acc=0.xx  params=…  flops≈…  nbytes_fp64=…
  int8 fq   loss=0.xx  acc=0.xx  nbytes≈…  size_ratio=0.xx  acc_drop=±0.xx
  prune     loss=0.xx  acc=0.xx  nonzero=…  pruned=…  acc_drop=±0.xx
  compose   order=prune→quantize  loss=0.xx  acc=0.xx  nonzero=…  nbytes≈…  size_ratio=0.xx
```

## Efficient block (toy)

Stem `3×3` conv → ReLU → **depthwise** `3×3` → ReLU → **pointwise** `1×1` → ReLU → global average pool → linear classifier. Same idea as MobileNet-style separable blocks used in edge vision — here at 8×8 grayscale blobs so everything stays CPU-instant.

## Quantization & pruning

**Int8 fake-quant (PTQ toy):** per-tensor affine

```text
q = clip(round(w / scale) + zp, -128, 127)
ŵ = (q − zp) · scale
```

Weights are fake-quantized in float for the forward pass; size stats compare fp64 payload vs packed int8 weights (+ fp biases).

**Magnitude prune:** zero the lowest-|w| fraction per weight tensor (`configs/default.yaml` → `prune.sparsity`).

**Compose (prune → int8):** set `compose.order: [prune, quantize]` to run the fourth eval row. Order is configurable for teaching; default matches the common prune-then-quantize recipe.


**Calibration observers (teaching stub):** `quantize.observer: minmax | percentile` chooses how weight ranges are measured before affine fake-quant. `percentile` clips to the p-th abs quantile (`quantize.percentile`, default 99) — a tiny stand-in for histogram/percentile calibrators in TensorRT/ModelOpt, **not** those products.

**Selective quant / skip list:** `quantize.skip: [fc]` (substring match on weight names) leaves sensitive layers in float — the usual “skip the head” pattern. Eval adds rows `int8_minmax`, `int8_percentile`, `int8_skip_head` alongside the existing `prune_then_int8` compose row.


## Knowledge distillation + QAT (round 3)

**KD:** train a wider teacher, distill into a thinner student with

```text
L = α · T² · KL(softmax(z_t/T) ∥ softmax(z_s/T)) + (1−α) · CE(z_s, y)
```

Config: `distill.enabled`, `temperature`, `alpha`, `student_width`  
([PyTorch KD tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)).

**QAT:** fake-quant weights during a few SGD steps, then convert to the same int8 path as PTQ. Report row `qat_vs_ptq` contrasts float / PTQ / QAT  
([torchao QAT](https://docs.pytorch.org/ao/stable/workflows/qat.html)).

Recommended ladder (document only): train → (optional KD) → prune → (optional QAT) → PTQ/int8. Still **fake-quant**, not TensorRT.

## Eval harness

`run_before_after(cfg)` trains the float model, then scores:

1. **baseline** — loss, accuracy, params, rough FLOPs, fp64 nbytes  
2. **int8 fake-quant** — loss/acc + packed size ratio  
3. **magnitude prune** — loss/acc + nonzero count  
4. **compose (`prune_then_int8`)** — magnitude prune → fake-quant on remaining weights (`compose.order` in YAML)

**Why order matters:** community recipes often prefer prune→INT8 (then optionally distill) over isolated ops; sparsity changes the weight distribution that PTQ sees. This demo only shows the toy compose path — still **fake-quant**, not TensorRT/ONNX Runtime kernels or QAT. See e.g. [r/computervision on prune/distill/quantize order](https://www.reddit.com/r/computervision/comments/1i84qw7/prune_distill_quantize_whats_the_best_order/).

```bash
pytest tests/ -q
```

## Per-channel weight quantization (round 4)

Config switch `quantize.granularity: per_tensor | per_channel` (default `per_tensor`, which is the unchanged behaviour). `per_channel` gives every **output channel** (axis 0) its own affine `(scale, zero_point)`, using the same observer and formula as above. The switch applies to the configured PTQ row, the compose row, and QAT.

```python
from efficient_cnn.quantize import quantize_model, layer_error_table, format_layer_error_table
qm, stats = quantize_model(model, bits=8, granularity="per_channel")
stats.channel_scale["dw_w"]        # 8 scales, one per depthwise filter
stats.nbytes_qparams               # scale/zp overhead: 5 B per group (fp32 scale + int8 zp)
print(format_layer_error_table(layer_error_table(model, (8, 4), x=test_X[:64])))
```

**Why the depthwise layer:** a depthwise filter sees one channel only, so ranges can differ a lot between filters (BN folding makes this worse in real MobileNets). One scale per tensor then spends most of the levels on the widest channel and crushes the rest ([Sheng et al., arXiv 1803.08607](https://ar5iv.labs.arxiv.org/html/1803.08607)). Per-channel weight scales are the standard fix ([Wu et al. / NVIDIA, arXiv 2004.09602](https://arxiv.org/pdf/2004.09602), [NVIDIA blog](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/)). It is also the first thing people hit when "quantization killed my accuracy" ([r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/comments/1ta3k82/quantization_killed_my_models_accuracy/)).

`python evals/runner.py` now also prints:

- **Granularity ablation**: `int8_per_tensor`, `int8_per_channel`, and optional `int4_*` (`quantize.int4_rows`).
- **Per-layer error table** (`quantize.error_table`): weight MSE per_tensor vs per_channel, their ratio, the per-channel **range spread** (max/min range), and logit MSE when *only that layer* is quantized. The depthwise row is marked `◀ depthwise`.
- **Depthwise outlier stress** (`quantize.dw_outlier_factor`, default 64). Depthwise channel 0 is scaled ×k and the matching pointwise input column ÷k. ReLU is positively homogeneous, so the float logits are **identical**, but the depthwise tensor now has one very wide channel.

```text
layer   bits  ch  spread  mse_tensor mse_channel  ratio logit_mse_t logit_mse_c
dw_w       8   8    2.95   8.348e-06   3.371e-06    2.5   8.536e-05   6.419e-06  ◀ depthwise
dw_w       4   8    2.95   2.363e-03   8.227e-04    2.9   6.739e-03   5.522e-03  ◀ depthwise
  dw_outlier x64 (float_acc=1.000, same logits)  int8_per_tensor=1.000  int8_per_channel=1.000  int4_per_tensor=0.281  int4_per_channel=1.000
```

**Honest reading:** on the trained 8×8 toy, int8 is already lossless at either granularity. Per-channel roughly halves the weight MSE on every layer, and it is not always lower on each individual tensor (rounding and zero-point effects). The depthwise layer is *not* dramatically worse than the others here, because its channel spread is only about 3×. The gap becomes visible once the ranges diverge: in the outlier stress, per-tensor int4 falls to chance (0.28) while per-channel stays at 1.00. This is still **fake-quant in float**. There are no int8 kernels, no activation quantization, and it is not TensorRT, torchao, or ONNX Runtime.

## Auto skip list / mixed precision from measured sensitivity (round 5)

Round 4's error table *measures* per-layer sensitivity. Round 5 turns it into a **decision**: pick `float` / `int8` / `int4` / `int2` per weight tensor under an accuracy budget. This is a numpy-sized version of what [NVIDIA ModelOpt AutoQuantize](https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html), [AIMET automatic mixed precision](https://quic.github.io/aimet-pages/releases/latest/techniques/mixed_precision/amp.html), and the [HF sensitivity-aware mixed-precision quantizer](https://huggingface.co/blog/badaoui/sensitivity-aware-mixed-precision-quantizer-v1) do. It also answers the "which layers should I keep in higher precision?" question from [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1v8nj6o/i_built_a_tool_to_actually_test_which_weights/).

1. **Measure:** quantize *only one layer* at each rung and record calibration accuracy and logit MSE. This is the same logit MSE as `layer_error_table`.
2. **Greedy descent:** start all-float. Take the step with the smallest logit MSE per byte saved, and evaluate the *whole* mixed model. Keep the step if `acc_drop ≤ quantize.auto_mixed_precision.max_acc_drop`. Otherwise that layer stays where it is.
3. **Report:** the per-layer assignment, the suggested `quantize.skip` (layers left float), the skip list you would need for a *uniform* int8 or int4 run, and a Pareto table (bytes vs test accuracy, `*` = non-dominated).

Decisions use the first `calib_samples` **training** examples. The test split only scores the result.

```bash
python evals/runner.py                                    # report now includes "Auto mixed precision" blocks
python evals/runner.py --mixed-precision --max-acc-drop 0.0 --ladder float 8 4 2   # → evals/mixed_precision.json
python evals/runner.py --mixed-precision --granularity per_channel
```

```text
Auto mixed precision — dw_outlier (greedy on calib, budget acc_drop ≤ 0.010, per_tensor, ladder float/int8/int4/int2)
  assignment: stem_w=int4  dw_w=int8  pw_w=int4  fc_w=int2
  calib acc 1.000 (float 1.000)  test acc 1.000 (float 1.000)  bytes 392/2144 (size_ratio 0.183)
  suggested quantize.skip (mixed): []   uniform int8 → skip []  uniform int4 → skip ['dw_w']
  rejected: stem_w: int4 → int2 (calib acc 0.758); pw_w: int4 → int2 (calib acc 0.758); dw_w: int8 → int4 (calib acc 0.281)
  uniform int4                                     364  0.170    0.281
  int4 skip=['dw_w']                               899  0.419    1.000
```

On the clean model the search ends at `stem/pw/fc=int4, dw=int2` (346 B, test acc 1.000). That is smaller than uniform int4 (364 B) at the same accuracy, while uniform int2 drops to 0.719. On the function-preserving **depthwise-outlier** model, it refuses per-tensor int4 for `dw_w` (accuracy would fall to 0.281) and keeps it at int8. For a uniform int4 pipeline, it suggests `quantize.skip: [dw_w]`. The config is under `quantize.auto_mixed_precision`: `enabled`, `max_acc_drop`, `ladder`, `granularity`, `calib_samples`, `dw_outlier`.

**Honesty:**
- Greedy is a heuristic, not an optimal search. ModelOpt solves a constrained optimisation, and AIMET uses Pareto-front phases.
- On an 8×8 toy with 4 layers, accuracy is coarse (128 samples).
- Sizes use the toy's fp64 baseline and *packed* intN counts. Everything is still weight-only fake-quant in float: no int4 or int2 kernels, no activation quantization, and none of ModelOpt, AIMET or TensorRT.

## Design notes

- Prefer **fully synthetic** data so CI never needs network or caches.
- **numpy only** — no torch/HF required.
- This is a **learning demo**, not a claim about production edge deployment, employer systems, or Plumerai products.

## License

MIT — see [LICENSE](LICENSE).
