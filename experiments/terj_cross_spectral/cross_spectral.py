#!/usr/bin/env python3
"""TERJ Cross-Spectral Test — Stage A reference implementation.

Author: Paweł Majsterek

This program compares null-standardized residual feature vectors from
Riemann-zero blocks and neural-network spectral groups.

It does NOT compare raw Riemann and neural spectra directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MC_SEED = 2026092901


def _finite(x: Any, name: str) -> float:
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return y


def load_input(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != "terj-cross-spectral-v1":
        raise ValueError("Unsupported or missing schema")
    features = obj.get("features")
    groups = obj.get("groups")
    if not isinstance(features, list) or not features:
        raise ValueError("features must be a non-empty list")
    if len(set(features)) != len(features):
        raise ValueError("features must be unique")
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("groups must contain at least two entries")
    domains = {g.get("domain") for g in groups}
    if "riemann" not in domains or "neural" not in domains:
        raise ValueError("Both riemann and neural groups are required")

    seen_ids = set()
    for g in groups:
        gid = str(g.get("id", ""))
        if not gid or gid in seen_ids:
            raise ValueError("Each group must have a unique non-empty id")
        seen_ids.add(gid)
        if g.get("domain") not in {"riemann", "neural"}:
            raise ValueError(f"Unsupported domain for {gid}")
        obs = g.get("observed", {})
        null = g.get("null", {})
        for f in features:
            if f not in obs or f not in null:
                raise ValueError(f"Missing feature {f} in group {gid}")
            _finite(obs[f], f"{gid}.{f}.observed")
            mu = _finite(null[f]["mean"], f"{gid}.{f}.null.mean")
            sd = _finite(null[f]["sd"], f"{gid}.{f}.null.sd")
            if sd <= 0:
                raise ValueError(f"Null sd must be > 0 for {gid}.{f}")
    return obj


def residual_matrix(obj: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], np.ndarray]:
    features = list(obj["features"])
    groups = list(obj["groups"])
    z = np.empty((len(groups), len(features)), dtype=np.float64)
    for i, g in enumerate(groups):
        for j, f in enumerate(features):
            x = float(g["observed"][f])
            mu = float(g["null"][f]["mean"])
            sd = float(g["null"][f]["sd"])
            z[i, j] = (x - mu) / sd
    return features, groups, z


def domain_arrays(groups: list[dict[str, Any]], z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ri = np.asarray([z[i] for i, g in enumerate(groups) if g["domain"] == "riemann"])
    ne = np.asarray([z[i] for i, g in enumerate(groups) if g["domain"] == "neural"])
    return ri, ne


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def pairwise_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x[:, None, :] - y[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    ab = pairwise_distances(a, b).mean()
    aa = pairwise_distances(a, a).mean()
    bb = pairwise_distances(b, b).mean()
    return float(max(0.0, 2.0 * ab - aa - bb))


def median_bandwidth(x: np.ndarray) -> float:
    d = pairwise_distances(x, x)
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    d = x[:, None, :] - y[None, :, :]
    d2 = np.sum(d * d, axis=2)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mmd2_biased(a: np.ndarray, b: np.ndarray, sigma: float | None = None) -> tuple[float, float]:
    joined = np.vstack([a, b])
    bw = median_bandwidth(joined) if sigma is None else float(sigma)
    if not math.isfinite(bw) or bw <= 0:
        bw = 1.0
    kaa = gaussian_kernel(a, a, bw)
    kbb = gaussian_kernel(b, b, bw)
    kab = gaussian_kernel(a, b, bw)
    val = float(kaa.mean() + kbb.mean() - 2.0 * kab.mean())
    return max(0.0, val), bw


def feature_centroid_differences(features: list[str], a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.mean(axis=0) - b.mean(axis=0)
    return {f: float(diff[i]) for i, f in enumerate(features)}


def leave_one_feature_out(features: list[str], a: np.ndarray, b: np.ndarray) -> list[dict[str, Any]]:
    out = []
    if len(features) == 1:
        return out
    for j, f in enumerate(features):
        keep = [i for i in range(len(features)) if i != j]
        out.append({
            "removed_feature": f,
            "centroid_distance": centroid_distance(a[:, keep], b[:, keep]),
        })
    return out


def mc_null_centroid_distance(
    n_r: int,
    n_n: int,
    p: int,
    observed: float,
    reps: int = 100_000,
    seed: int = MC_SEED,
) -> dict[str, float]:
    """Calibrate D_RN under independent N(0,1) residual nulls.

    This is deliberately simple: after valid null-standardization, each
    residual coordinate should be approximately centered and unit-scaled.
    More exact empirical-control calibration should replace this when raw
    control vectors are supplied.
    """
    rng = np.random.default_rng(seed)
    sims = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        r = rng.normal(size=(n_r, p))
        n = rng.normal(size=(n_n, p))
        sims[i] = centroid_distance(r, n)

    lower = (np.count_nonzero(sims <= observed) + 1.0) / (reps + 1.0)
    upper = (np.count_nonzero(sims >= observed) + 1.0) / (reps + 1.0)
    two = min(1.0, 2.0 * min(lower, upper))
    return {
        "reps": int(reps),
        "seed": int(seed),
        "null_mean": float(sims.mean()),
        "null_sd": float(sims.std(ddof=1)),
        "null_q025": float(np.quantile(sims, 0.025)),
        "null_q50": float(np.quantile(sims, 0.5)),
        "null_q975": float(np.quantile(sims, 0.975)),
        "p_lower_unusually_small": float(lower),
        "p_upper_unusually_large": float(upper),
        "p_two_sided": float(two),
    }


def nearest_centroid_accuracy(groups: list[dict[str, Any]], z: np.ndarray) -> dict[str, Any]:
    """Exploratory leave-one-out nearest-centroid identification.

    With only two domains and small n this is diagnostic, not a primary test.
    """
    labels = np.asarray([g["domain"] for g in groups], dtype=object)
    predictions = []
    correct = 0
    usable = 0

    for i in range(len(groups)):
        train = np.arange(len(groups)) != i
        centroids = {}
        valid = True
        for domain in ("riemann", "neural"):
            arr = z[train & (labels == domain)]
            if len(arr) == 0:
                valid = False
                break
            centroids[domain] = arr.mean(axis=0)
        if not valid:
            predictions.append({"id": groups[i]["id"], "prediction": None})
            continue

        dist = {d: float(np.linalg.norm(z[i] - c)) for d, c in centroids.items()}
        pred = min(dist, key=dist.get)
        usable += 1
        correct += int(pred == labels[i])
        predictions.append({
            "id": groups[i]["id"],
            "truth": labels[i],
            "prediction": pred,
            "distances": dist,
        })

    return {
        "usable": usable,
        "correct": correct,
        "accuracy": None if usable == 0 else correct / usable,
        "predictions": predictions,
    }


def analyze(obj: dict[str, Any], mc_reps: int) -> dict[str, Any]:
    features, groups, z = residual_matrix(obj)
    ri, ne = domain_arrays(groups, z)

    d = centroid_distance(ri, ne)
    ed = energy_distance(ri, ne)
    mmd, bw = mmd2_biased(ri, ne)

    residual_records = []
    for i, g in enumerate(groups):
        residual_records.append({
            "id": g["id"],
            "domain": g["domain"],
            "z": {f: float(z[i, j]) for j, f in enumerate(features)},
        })

    return {
        "schema": "terj-cross-spectral-stage-a-result-v1",
        "features": features,
        "n_riemann_groups": int(len(ri)),
        "n_neural_groups": int(len(ne)),
        "residuals": residual_records,
        "centroids": {
            "riemann": {f: float(ri.mean(axis=0)[j]) for j, f in enumerate(features)},
            "neural": {f: float(ne.mean(axis=0)[j]) for j, f in enumerate(features)},
        },
        "primary": {
            "centroid_distance": d,
            "normal_approx_null_calibration": mc_null_centroid_distance(
                len(ri), len(ne), len(features), d, reps=mc_reps
            ),
        },
        "secondary": {
            "energy_distance": ed,
            "mmd2_biased": mmd,
            "mmd_gaussian_bandwidth": bw,
            "feature_centroid_differences_riemann_minus_neural":
                feature_centroid_differences(features, ri, ne),
            "leave_one_feature_out": leave_one_feature_out(features, ri, ne),
            "nearest_centroid_leave_one_out": nearest_centroid_accuracy(groups, z),
        },
        "interpretation_limits": [
            "A small cross-domain residual distance is not evidence of physical identity.",
            "The built-in Monte Carlo calibration assumes approximately standard-normal residual coordinates.",
            "Empirical matched-control calibration is preferred for final inference.",
            "Stage A tests static residual similarity only.",
        ],
    }


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="TERJ Cross-Spectral Test — Stage A")
    ap.add_argument("--input", required=True, help="terj-cross-spectral-v1 JSON input")
    ap.add_argument("--out", required=True, help="new output JSON path")
    ap.add_argument("--mc-reps", type=int, default=100_000)
    args = ap.parse_args()

    if args.mc_reps < 1000:
        raise ValueError("--mc-reps must be >= 1000")

    obj = load_input(args.input)
    result = analyze(obj, args.mc_reps)
    save_json(args.out, result)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
