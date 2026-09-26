"""Spectral statistics of a block of zeros (.npz from src.zeros or src.odlyzko).

    python -m src.stats data/z_1_20k.npz --out results/stats_1_20k.json

Prints a short summary; the full result (histograms included) goes to JSON.
"""
import argparse
import json
import os

import numpy as np
from scipy.integrate import quad

BINS = np.linspace(0, 3.5, 36)


def load(path):
    d = np.load(path, allow_pickle=False)
    return int(str(d["base"])), d["x"].astype(float), int(str(d["start_index"]))


def unfold(base, x):
    """Unfolded nearest-neighbour spacings, mean density log(t/2pi)/2pi at midpoints."""
    mid = base + 0.5 * (x[1:] + x[:-1])
    return np.diff(x) * np.log(mid / (2 * np.pi)) / (2 * np.pi)


def repulsion_exponent(s, smax=0.25, nboot=400, seed=0):
    """MLE of a in F(s) ~ (s/smax)^a on s < smax (Poisson a=1, GOE a=2, GUE a=3).

    The estimator is biased at finite smax; always compare with the same
    estimator applied to a simulated null sample of equal size.
    """
    def mle(v):
        v = v[(v > 0) & (v < smax)]
        return (len(v) / np.sum(np.log(smax / v)) if len(v) > 1 else np.nan), len(v)

    a, k = mle(s)
    rng = np.random.default_rng(seed)
    boots = [mle(rng.choice(s, size=len(s)))[0] for _ in range(nboot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(a), int(k), float(lo), float(hi)


def pair_correlation(s, umax=3.0, nbins=30, kmax=15):
    u = np.concatenate([[0.0], np.cumsum(s)])
    edges = np.linspace(0, umax, nbins + 1)
    counts = np.zeros(nbins)
    for k in range(1, kmax + 1):
        d = u[k:] - u[:-k]
        counts += np.histogram(d[d < umax], bins=edges)[0]
    return counts / (len(u) * (edges[1] - edges[0]))


def surmise_cdf(a):
    gue = quad(lambda s: 32 / np.pi**2 * s * s * np.exp(-4 * s * s / np.pi), 0, a)[0]
    return {"poisson": 1 - np.exp(-a), "goe": 1 - np.exp(-np.pi * a * a / 4), "gue": gue}


def summarize(path):
    base, x, start = load(path)
    s = unfold(base, x)
    h, _ = np.histogram(s, bins=BINS, density=True)
    a, k, lo, hi = repulsion_exponent(s)
    i = int(np.argmin(s))
    return {
        "file": path,
        "first_index": str(start),
        "height_from": float(base + x[0]),
        "height_to": float(base + x[-1]),
        "n_spacings": int(len(s)),
        "mean": float(s.mean()),
        "var": float(s.var()),
        "count_s_lt_0.1": int(np.sum(s < 0.1)),
        "count_s_lt_0.2": int(np.sum(s < 0.2)),
        "p01": float(np.mean(s < 0.1)),
        "p02": float(np.mean(s < 0.2)),
        "surmise_p01": surmise_cdf(0.1),
        "min_spacing": float(s[i]),
        "min_spacing_zero_index": str(start + i),
        "repulsion_exponent_a": a,
        "repulsion_exponent_ci95": [lo, hi],
        "repulsion_n_used": k,
        "hist_bins": BINS.tolist(),
        "hist": np.round(h, 4).tolist(),
        "pair_correlation": np.round(pair_correlation(s), 4).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = summarize(args.path)
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f)
    print(f"{r['n_spacings']} spacings, heights {r['height_from']:.6g}..{r['height_to']:.6g}")
    print(f"mean {r['mean']:.4f}  var {r['var']:.4f}  (surmise var: GUE 0.178, GOE 0.273)")
    print(f"s<0.1: {r['count_s_lt_0.1']} ({100 * r['p01']:.3f}%), s<0.2: {r['count_s_lt_0.2']}")
    print(f"min spacing {r['min_spacing']:.4f} at zero #{r['min_spacing_zero_index']}")
    print(f"repulsion exponent a = {r['repulsion_exponent_a']:.2f} "
          f"[{r['repulsion_exponent_ci95'][0]:.2f}, {r['repulsion_exponent_ci95'][1]:.2f}] "
          f"(n={r['repulsion_n_used']}; Poisson 1, GOE 2, GUE 3)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
