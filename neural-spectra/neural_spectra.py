#!/usr/bin/env python3
"""Neural Spectra — standalone NumPy reference implementation.

Author: Paweł Majsterek
Copyright (c) 2026 Paweł Majsterek

Licensed under neural-spectra/LICENSE.md.
Commercial use requires a separate written license.

The program implements:
- synthetic nonlinear binary classification data,
- float64 MLP 64 -> N -> N -> 2,
- Adam training,
- true-label and shuffled-label conditions,
- checkpoint spectral analysis of G = W2.T @ W2,
- finite-size local Poisson / real-Wishart / GOE / GUE controls,
- seed-level paired bootstrap, exact sign-flip tests, and BH-FDR.

It is intentionally self-contained and depends only on NumPy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DATASET_SEED = 2026092801
DEFAULT_INIT_SEEDS = [2026092802 + 10 * i for i in range(10)]
BATCH_SEED = 2026092803
SHUFFLE_SEED = 2026092804
ANALYSIS_SEED = 20260928

DEFAULT_CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50]


def _json_float(x: float) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite value cannot be serialized: {x}")
    return x


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_dataset(
    seed: int = DATASET_SEED,
    n_train: int = 2048,
    n_val: int = 512,
    n_test: int = 1024,
    dim: int = 64,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = n_train + n_val + n_test
    x = rng.normal(0.0, 1.0, size=(total, dim)).astype(np.float64)
    score = x[:, 0] + 0.8 * x[:, 1] * x[:, 2] + 0.5 * np.sin(x[:, 3])
    y = (score > 0.0).astype(np.int64)

    a = n_train
    b = n_train + n_val
    return {
        "x_train": x[:a],
        "y_train": y[:a],
        "x_val": x[a:b],
        "y_val": y[a:b],
        "x_test": x[b:],
        "y_test": y[b:],
    }


def init_params(width: int, seed: int, input_dim: int = 64, output_dim: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def w(shape: tuple[int, int], nin: int) -> np.ndarray:
        return rng.normal(0.0, 1.0 / math.sqrt(nin), size=shape).astype(np.float64)

    return {
        "W1": w((input_dim, width), input_dim),
        "b1": np.zeros(width, dtype=np.float64),
        "W2": w((width, width), width),
        "b2": np.zeros(width, dtype=np.float64),
        "W3": w((width, output_dim), width),
        "b3": np.zeros(output_dim, dtype=np.float64),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    z1 = x @ params["W1"] + params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ params["W2"] + params["b2"]
    h2 = np.tanh(z2)
    logits = h2 @ params["W3"] + params["b3"]
    return logits, (x, h1, h2)


def loss_accuracy(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsum = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - logsum
    loss = -np.mean(log_probs[np.arange(y.size), y])
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    return _json_float(loss), _json_float(acc)


def loss_and_grads(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    logits, (x0, h1, h2) = forward(params, x)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    loss = -np.mean(np.log(np.maximum(probs[np.arange(y.size), y], 1e-300)))

    dlogits = probs
    dlogits[np.arange(y.size), y] -= 1.0
    dlogits /= y.size

    grads: dict[str, np.ndarray] = {}
    grads["W3"] = h2.T @ dlogits
    grads["b3"] = np.sum(dlogits, axis=0)

    dh2 = dlogits @ params["W3"].T
    dz2 = dh2 * (1.0 - h2 * h2)
    grads["W2"] = h1.T @ dz2
    grads["b2"] = np.sum(dz2, axis=0)

    dh1 = dz2 @ params["W2"].T
    dz1 = dh1 * (1.0 - h1 * h1)
    grads["W1"] = x0.T @ dz1
    grads["b1"] = np.sum(dz1, axis=0)

    return _json_float(loss), grads


class Adam:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        if params.keys() != grads.keys():
            raise ValueError("Parameter and gradient keys differ")
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            mh = self.m[k] / b1t
            vh = self.v[k] / b2t
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def spectral_statistics(w2: np.ndarray, bulk_fraction: float = 0.5) -> dict[str, Any]:
    if w2.ndim != 2 or w2.shape[0] != w2.shape[1]:
        raise ValueError("W2 must be a square matrix")

    n = w2.shape[0]
    g = w2.T @ w2
    eig = np.linalg.eigvalsh(g)
    eig = np.maximum(eig, 0.0)

    trace = float(np.sum(eig))
    maxeig = float(eig[-1])
    spec = math.sqrt(maxeig)
    stable_rank = trace / maxeig if maxeig > 0.0 else 0.0

    bulk_n = int(round(n * bulk_fraction))
    bulk_n = min(n, max(4, bulk_n))
    if bulk_n % 2 == 1:
        bulk_n -= 1
    start = (n - bulk_n) // 2
    bulk = eig[start:start + bulk_n]

    gaps = np.diff(bulk)
    a = gaps[:-1]
    b = gaps[1:]
    den = np.maximum(a, b)
    num = np.minimum(a, b)
    valid = den > 0.0
    ratios = np.divide(num[valid], den[valid])

    if ratios.size == 0:
        raise RuntimeError("No valid local gap ratios")

    smallest = float(eig[0])
    condition = maxeig / smallest if smallest > 0.0 else float("inf")

    return {
        "dimension": int(n),
        "eigenvalue_count": int(eig.size),
        "bulk_level_count": int(bulk.size),
        "spacing_count": int(gaps.size),
        "expected_valid_ratio_count": int(max(0, bulk.size - 2)),
        "actual_valid_ratio_count": int(ratios.size),
        "zero_spacing_count": int(np.sum(gaps == 0.0)),
        "trace": _json_float(trace),
        "trace_over_width": _json_float(trace / n),
        "max_eigenvalue": _json_float(maxeig),
        "max_eig_over_trace": _json_float(maxeig / trace),
        "spectral_norm": _json_float(spec),
        "stable_rank": _json_float(stable_rank),
        "stable_rank_over_width": _json_float(stable_rank / n),
        "condition_number": None if not math.isfinite(condition) else _json_float(condition),
        "mean_rtilde": _json_float(np.mean(ratios)),
        "median_rtilde": _json_float(np.median(ratios)),
        "frac_rtilde_lt_0_1": _json_float(np.mean(ratios < 0.1)),
        "frac_rtilde_lt_0_2": _json_float(np.mean(ratios < 0.2)),
    }


def evaluate(params: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in ("train", "val", "test"):
        logits, _ = forward(params, data[f"x_{split}"])
        loss, acc = loss_accuracy(logits, data[f"y_{split}"])
        out[f"{split}_loss"] = loss
        out[f"{split}_accuracy"] = acc
    return out


def train_one(
    width: int,
    init_seed: int,
    data: dict[str, np.ndarray],
    train_labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    checkpoints: list[int] | None = None,
) -> dict[str, Any]:
    params = init_params(width, init_seed)
    adam = Adam(params)
    checkpoints = sorted(set(checkpoints or DEFAULT_CHECKPOINTS))
    checkpoints = [e for e in checkpoints if 0 <= e <= epochs]
    if 0 not in checkpoints:
        checkpoints = [0] + checkpoints
    if epochs not in checkpoints:
        checkpoints.append(epochs)

    x_train = data["x_train"]
    n = x_train.shape[0]
    batch_rng = np.random.default_rng(BATCH_SEED)

    result = {
        "width": int(width),
        "init_seed": int(init_seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "checkpoints": [],
    }

    def snapshot(epoch: int) -> None:
        metrics = evaluate(params, data)
        spec = spectral_statistics(params["W2"])
        result["checkpoints"].append({
            "epoch": int(epoch),
            **metrics,
            "spectral": spec,
        })

    snapshot(0)

    checkpoint_set = set(checkpoints)
    for epoch in range(1, epochs + 1):
        perm = batch_rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            _, grads = loss_and_grads(params, x_train[idx], train_labels[idx])
            adam.step(params, grads)
        if epoch in checkpoint_set:
            snapshot(epoch)

    return result


def run_experiment(
    widths: list[int],
    seeds: list[int],
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    data = make_dataset()
    shuffle_rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = data["y_train"][shuffle_rng.permutation(data["y_train"].size)]

    runs = []
    for width in widths:
        for seed in seeds:
            for condition, labels in (
                ("true", data["y_train"]),
                ("shuffled", shuffled),
            ):
                one = train_one(
                    width=width,
                    init_seed=seed,
                    data=data,
                    train_labels=labels,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                one["condition"] = condition
                runs.append(one)

    return {
        "schema": "neural-spectra-experiment-v1",
        "dataset_seed": DATASET_SEED,
        "batch_seed": BATCH_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "widths": [int(x) for x in widths],
        "init_seeds": [int(x) for x in seeds],
        "runs": runs,
    }


def local_stats_from_levels(levels: np.ndarray) -> dict[str, float | int]:
    levels = np.sort(np.asarray(levels, dtype=np.float64))
    n = levels.size
    bulk_n = n // 2
    if bulk_n % 2 == 1:
        bulk_n -= 1
    start = (n - bulk_n) // 2
    bulk = levels[start:start + bulk_n]
    gaps = np.diff(bulk)
    den = np.maximum(gaps[:-1], gaps[1:])
    num = np.minimum(gaps[:-1], gaps[1:])
    valid = den > 0.0
    ratios = num[valid] / den[valid]
    if ratios.size == 0:
        raise RuntimeError("Control realization has no valid ratios")
    return {
        "mean_rtilde": _json_float(np.mean(ratios)),
        "median_rtilde": _json_float(np.median(ratios)),
        "frac_rtilde_lt_0_1": _json_float(np.mean(ratios < 0.1)),
        "frac_rtilde_lt_0_2": _json_float(np.mean(ratios < 0.2)),
        "valid_ratio_count": int(ratios.size),
        "zero_spacing_count": int(np.sum(gaps == 0.0)),
    }


def sample_control_levels(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "poisson":
        gaps = rng.exponential(1.0, size=n - 1)
        return np.concatenate(([0.0], np.cumsum(gaps)))

    if kind == "real_wishart":
        x = rng.normal(0.0, 1.0 / math.sqrt(n), size=(n, n))
        return np.linalg.eigvalsh(x.T @ x)

    if kind == "goe":
        a = rng.normal(size=(n, n))
        h = (a + a.T) / math.sqrt(2.0 * n)
        return np.linalg.eigvalsh(h)

    if kind == "gue":
        a = rng.normal(size=(n, n))
        b = rng.normal(size=(n, n))
        z = a + 1j * b
        h = (z + z.conj().T) / math.sqrt(4.0 * n)
        return np.linalg.eigvalsh(h)

    raise ValueError(f"Unknown control kind: {kind}")


def summarize_scalars(records: list[dict[str, Any]], metric: str) -> dict[str, float]:
    x = np.asarray([r[metric] for r in records], dtype=np.float64)
    return {
        "mean": _json_float(np.mean(x)),
        "sd": _json_float(np.std(x, ddof=1)),
        "q025": _json_float(np.quantile(x, 0.025)),
        "median": _json_float(np.median(x)),
        "q975": _json_float(np.quantile(x, 0.975)),
    }


def run_controls(
    widths: list[int],
    realizations: int,
    master_seed: int = 2026092810,
) -> dict[str, Any]:
    kinds = ["poisson", "real_wishart", "goe", "gue"]
    out: dict[str, Any] = {
        "schema": "neural-spectra-local-controls-v1",
        "master_seed": int(master_seed),
        "realizations_per_ensemble": int(realizations),
        "controls": {},
    }

    seq = np.random.SeedSequence(master_seed)
    child = seq.spawn(len(widths) * len(kinds))
    ci = 0

    for n in widths:
        width_out: dict[str, Any] = {}
        for kind in kinds:
            rng = np.random.default_rng(child[ci])
            ci += 1
            records = []
            for _ in range(realizations):
                levels = sample_control_levels(kind, n, rng)
                records.append(local_stats_from_levels(levels))
            width_out[kind] = {
                "count": len(records),
                "summary": {
                    metric: summarize_scalars(records, metric)
                    for metric in (
                        "mean_rtilde",
                        "median_rtilde",
                        "frac_rtilde_lt_0_1",
                        "frac_rtilde_lt_0_2",
                    )
                },
                "records": records,
            }
        out["controls"][str(n)] = width_out
    return out


def exact_sign_flip_pvalue(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("d must be a non-empty 1D array")
    obs = abs(float(np.mean(d)))
    hits = 0
    total = 1 << d.size
    for mask in range(total):
        signs = np.ones(d.size, dtype=np.float64)
        for i in range(d.size):
            if (mask >> i) & 1:
                signs[i] = -1.0
        stat = abs(float(np.mean(signs * d)))
        if stat >= obs - 1e-15:
            hits += 1
    return _json_float(hits / total)


def paired_bootstrap_ci(
    d: np.ndarray,
    reps: int = 20000,
    seed: int = ANALYSIS_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    means = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        means[r] = np.mean(d[idx])
    return (
        _json_float(np.quantile(means, alpha / 2.0)),
        _json_float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def paired_effect(d: np.ndarray) -> float | None:
    d = np.asarray(d, dtype=np.float64)
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return None
    return _json_float(np.mean(d) / sd)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return [_json_float(x) for x in out]


def checkpoint_lookup(run: dict[str, Any], epoch: int) -> dict[str, Any]:
    for c in run["checkpoints"]:
        if int(c["epoch"]) == int(epoch):
            return c
    raise KeyError(f"Missing epoch {epoch}")


def paired_test_record(name: str, a: np.ndarray, b: np.ndarray, bootstrap_seed: int) -> dict[str, Any]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    lo, hi = paired_bootstrap_ci(d, seed=bootstrap_seed)
    return {
        "name": name,
        "n_pairs": int(d.size),
        "mean_difference": _json_float(np.mean(d)),
        "ci95": [lo, hi],
        "p_exact_sign_flip": exact_sign_flip_pvalue(d),
        "cohen_dz": paired_effect(d),
        "positive": int(np.sum(d > 0)),
        "negative": int(np.sum(d < 0)),
        "zero": int(np.sum(d == 0)),
    }


def run_formal_analysis(experiment: dict[str, Any]) -> dict[str, Any]:
    runs = experiment["runs"]
    widths = sorted({int(r["width"]) for r in runs})
    seeds = sorted({int(r["init_seed"]) for r in runs})
    final_epoch = max(int(r["epochs"]) for r in runs)

    by = {(int(r["width"]), str(r["condition"]), int(r["init_seed"])): r for r in runs}

    global_metrics = [
        "stable_rank_over_width",
        "max_eig_over_trace",
        "trace_over_width",
        "spectral_norm",
    ]
    local_metrics = [
        "mean_rtilde",
        "median_rtilde",
        "frac_rtilde_lt_0_1",
        "frac_rtilde_lt_0_2",
    ]

    stage_a = []
    counter = 0
    for width in widths:
        for condition in ("true", "shuffled"):
            for metric in global_metrics:
                a, b = [], []
                for seed in seeds:
                    run = by[(width, condition, seed)]
                    a.append(checkpoint_lookup(run, final_epoch)["spectral"][metric])
                    b.append(checkpoint_lookup(run, 0)["spectral"][metric])
                stage_a.append(
                    paired_test_record(
                        f"A:{width}:{condition}:{metric}:final-minus-init",
                        np.asarray(a), np.asarray(b),
                        ANALYSIS_SEED + counter,
                    )
                )
                counter += 1
    q = benjamini_hochberg([x["p_exact_sign_flip"] for x in stage_a])
    for rec, qi in zip(stage_a, q):
        rec["q_bh"] = qi

    stage_b_global = []
    stage_b_local = []
    for metric, target in [
        *[(m, stage_b_global) for m in global_metrics],
        *[(m, stage_b_local) for m in local_metrics],
    ]:
        for width in widths:
            true_v, shuf_v = [], []
            for seed in seeds:
                rt = by[(width, "true", seed)]
                rs = by[(width, "shuffled", seed)]
                true_v.append(checkpoint_lookup(rt, final_epoch)["spectral"][metric])
                shuf_v.append(checkpoint_lookup(rs, final_epoch)["spectral"][metric])
            target.append(
                paired_test_record(
                    f"B:{width}:{metric}:true-minus-shuffled",
                    np.asarray(true_v), np.asarray(shuf_v),
                    ANALYSIS_SEED + counter,
                )
            )
            counter += 1

    for family in (stage_b_global, stage_b_local):
        q = benjamini_hochberg([x["p_exact_sign_flip"] for x in family])
        for rec, qi in zip(family, q):
            rec["q_bh"] = qi

    return {
        "schema": "neural-spectra-formal-analysis-v1",
        "analysis_seed": ANALYSIS_SEED,
        "final_epoch": int(final_epoch),
        "network_seed_count": int(len(seeds)),
        "stage_a_global_final_vs_initial": stage_a,
        "stage_b_true_vs_shuffled_global": stage_b_global,
        "stage_b_true_vs_shuffled_local": stage_b_local,
        "limitations": [
            "Inference unit is the network initialization seed.",
            "All network seeds share the fixed dataset and training setup.",
            "Exact sign-flip inference assumes sign exchangeability/symmetry under the null.",
            "This analysis does not establish asymptotic universality.",
        ],
    }


def parse_seed_list(text: str | None, count: int) -> list[int]:
    if text:
        vals = [int(x.strip()) for x in text.split(",") if x.strip()]
        if not vals:
            raise ValueError("Empty --seed-list")
        return vals
    if count <= len(DEFAULT_INIT_SEEDS):
        return DEFAULT_INIT_SEEDS[:count]
    return [2026092802 + 10 * i for i in range(count)]


def cmd_experiment(args: argparse.Namespace) -> None:
    seeds = parse_seed_list(args.seed_list, args.seeds)
    payload = run_experiment(
        widths=args.widths,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    save_json(args.out, payload)
    print(f"Wrote experiment: {args.out}")


def cmd_controls(args: argparse.Namespace) -> None:
    payload = run_controls(
        widths=args.widths,
        realizations=args.realizations,
        master_seed=args.master_seed,
    )
    save_json(args.out, payload)
    print(f"Wrote controls: {args.out}")


def cmd_formal(args: argparse.Namespace) -> None:
    experiment = json.loads(Path(args.experiment).read_text(encoding="utf-8"))
    payload = run_formal_analysis(experiment)
    save_json(args.out, payload)
    print(f"Wrote formal analysis: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Neural Spectra standalone reference implementation"
    )
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("experiment", help="train true/shuffled networks and save spectral checkpoints")
    e.add_argument("--widths", type=int, nargs="+", default=[96, 256, 512])
    e.add_argument("--seeds", type=int, default=10, help="number of default initialization seeds")
    e.add_argument("--seed-list", default=None, help="comma-separated explicit seed list")
    e.add_argument("--epochs", type=int, default=50)
    e.add_argument("--batch-size", type=int, default=128)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_experiment)

    c = sub.add_parser("controls", help="generate finite-size local random-matrix controls")
    c.add_argument("--widths", type=int, nargs="+", default=[96, 256, 512])
    c.add_argument("--realizations", type=int, default=5000)
    c.add_argument("--master-seed", type=int, default=2026092810)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_controls)

    f = sub.add_parser("formal", help="run seed-level formal analysis on experiment JSON")
    f.add_argument("--experiment", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_formal)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
