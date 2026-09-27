"""Teacher → student knowledge distillation (temperature softmax + CE mix).

Loss = α · T² · KL(softmax(z_t/T) ∥ softmax(z_s/T)) + (1−α) · CE(z_s, y)

PyTorch tutorial analogue (numpy toy):
https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
"""

from __future__ import annotations

from typing import Any

import numpy as np

from efficient_cnn.data import ImageDataset
from efficient_cnn.layers import softmax
from efficient_cnn.model import TinyEfficientCNN
from efficient_cnn.train import _train_step, train


def _kl_soft(teacher_logits: np.ndarray, student_logits: np.ndarray, T: float) -> float:
    """KL(p_t || p_s) with temperature; returns mean over batch (nats)."""
    pt = softmax(teacher_logits / T)
    # log student probs at T
    zs = student_logits / T
    zs = zs - zs.max(axis=1, keepdims=True)
    log_ps = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True) + 1e-12)
    # KL = sum_k pt * (log pt - log ps)
    log_pt = np.log(pt + 1e-12)
    return float(np.mean(np.sum(pt * (log_pt - log_ps), axis=1)))


def _ce(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax(logits)
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def distill_train_step(
    student: TinyEfficientCNN,
    teacher: TinyEfficientCNN,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> float:
    """
    One KD step: soft KL to teacher + hard CE, then SGD on student via CE backprop.

    For the toy we mix losses for the reported scalar, but reuse analytical CE
    backprop (``_train_step``) as a stable teaching update — soft targets mainly
    shape the reported KD loss. A fuller soft-label backprop is left as an
    exercise / peft-free stub.
    """
    T = float(temperature)
    a = float(alpha)
    # Teacher frozen forward
    with_np = teacher.forward(x)
    # CE path updates student (hard labels) — primary gradient for the toy
    ce_loss = _train_step(student, x, y, lr * (1.0 - a + 1e-12))
    # Soft loss (monitoring / report); optional tiny nudge via CE on soft labels
    z_s = student.forward(x)
    soft = _kl_soft(with_np, z_s, T)
    # Soft-label CE nudge: treat teacher hard-ish soft targets
    pt = softmax(with_np / T)
    soft_y = pt.argmax(axis=1)
    if a > 0:
        _train_step(student, x, soft_y, lr * a * (T * T) * 0.1)
    return float(a * (T * T) * soft + (1.0 - a) * ce_loss)


def train_student_with_kd(
    teacher: TinyEfficientCNN,
    student: TinyEfficientCNN,
    train_ds: ImageDataset,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    temperature: float = 2.0,
    alpha: float = 0.7,
    seed: int = 0,
) -> list[float]:
    """Distill teacher → student; returns per-epoch mean KD losses."""
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    history: list[float] = []
    for _ in range(epochs):
        idx = rng.permutation(n)
        losses: list[float] = []
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            losses.append(
                distill_train_step(
                    student,
                    teacher,
                    train_ds.X[batch],
                    train_ds.y[batch],
                    lr=lr,
                    temperature=temperature,
                    alpha=alpha,
                )
            )
        history.append(float(np.mean(losses)))
    return history


def build_teacher_student(
    *,
    in_channels: int,
    teacher_width: int,
    student_width: int,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[TinyEfficientCNN, TinyEfficientCNN]:
    """Wider teacher + thinner student (same depthwise-separable topology)."""
    teacher = TinyEfficientCNN(in_channels, teacher_width, n_classes, rng)
    student = TinyEfficientCNN(in_channels, student_width, n_classes, rng)
    return teacher, student


def run_kd_demo(
    cfg: dict[str, Any],
    train_ds: ImageDataset,
    test_ds: ImageDataset,
) -> dict[str, Any]:
    """Train teacher, distill into thinner student, compare student alone vs KD."""
    seed = int(cfg["seed"])
    d = cfg["data"]
    m = cfg["model"]
    t = cfg["train"]
    kd = cfg.get("distill") or {}
    temperature = float(kd.get("temperature", 2.0))
    alpha = float(kd.get("alpha", 0.7))
    student_width = int(kd.get("student_width", max(4, int(m["width"]) // 2)))
    teacher_epochs = int(kd.get("teacher_epochs", t["epochs"]))
    student_epochs = int(kd.get("epochs", max(5, t["epochs"] // 2)))

    rng = np.random.default_rng(seed)
    teacher, student_kd = build_teacher_student(
        in_channels=int(d["channels"]),
        teacher_width=int(m["width"]),
        student_width=student_width,
        n_classes=int(d["n_classes"]),
        rng=rng,
    )
    # Baseline student (no KD) — same init seed offset
    rng2 = np.random.default_rng(seed + 1)
    student_solo = TinyEfficientCNN(
        int(d["channels"]), student_width, int(d["n_classes"]), rng2
    )

    train(
        teacher,
        train_ds,
        epochs=teacher_epochs,
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        seed=seed,
    )
    train(
        student_solo,
        train_ds,
        epochs=student_epochs,
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        seed=seed + 2,
    )
    kd_losses = train_student_with_kd(
        teacher,
        student_kd,
        train_ds,
        epochs=student_epochs,
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        temperature=temperature,
        alpha=alpha,
        seed=seed + 3,
    )

    t_loss, t_acc = teacher.loss_acc(test_ds.X, test_ds.y)
    s_loss, s_acc = student_solo.loss_acc(test_ds.X, test_ds.y)
    k_loss, k_acc = student_kd.loss_acc(test_ds.X, test_ds.y)
    return {
        "temperature": temperature,
        "alpha": alpha,
        "student_width": student_width,
        "teacher_width": int(m["width"]),
        "teacher": {"loss": t_loss, "acc": t_acc},
        "student_solo": {"loss": s_loss, "acc": s_acc},
        "student_kd": {"loss": k_loss, "acc": k_acc},
        "kd_final_loss": kd_losses[-1] if kd_losses else None,
        "notes": (
            "OSS learning stub — KD temperature/alpha mix; not production distillation."
        ),
    }
