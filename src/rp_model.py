"""Scan of the TERJ model H + lambda*G (Rosenzweig-Porter) over lambda,
for complex G (GUE, no time-reversal symmetry) and real G (GOE, like nuclei).

    python -m src.rp_model --out results/rp_scan.json
"""
import argparse
import json
import os

import numpy as np

from src.rmt import rp_sample

LAMS = [0, 0.03, 0.08, 0.15, 0.3, 0.6, 1.2, 3.0, None]
BINS = np.linspace(0, 3.5, 36)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--R", type=int, default=110, help="realisations per lambda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/rp_scan.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    res = {"N": a.N, "R": a.R, "seed": a.seed}
    for kind in ("gue", "goe"):
        rows = []
        for lam in LAMS:
            s, pr = zip(*(rp_sample(a.N, lam, kind, rng) for _ in range(a.R)))
            s, pr = np.concatenate(s), np.concatenate(pr)
            h, _ = np.histogram(s, bins=BINS, density=True)
            row = {"lam": "inf" if lam is None else lam, "p01": float(np.mean(s < 0.1)),
                   "pr_over_N": float(pr.mean()), "n": int(len(s)), "hist": np.round(h, 4).tolist()}
            rows.append(row)
            print(f"{kind} lam={row['lam']}: p01={row['p01']:.4f}  PR/N={row['pr_over_N']:.3f}", flush=True)
        res[kind] = rows
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
