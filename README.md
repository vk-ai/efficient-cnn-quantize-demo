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
| `src/efficient_cnn/quantize.py` | PTQ int8 fake-quant + minmax/percentile observers + skip-list |
| `src/efficient_cnn/prune.py` | Global-per-tensor magnitude pruning |
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

## Design notes

- Prefer **fully synthetic** data so CI never needs network or caches.
- **numpy only** — no torch/HF required.
- This is a **learning demo**, not a claim about production edge deployment, employer systems, or Plumerai products.

## License

MIT — see [LICENSE](LICENSE).
