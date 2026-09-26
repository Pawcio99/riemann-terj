"""Explicit formula checks: psi(x) rebuilt from zeros, and the 'spectroscopy'
F(x) = -sum_n cos(gamma_n ln x), whose peaks sit at prime powers.

    python -m src.explicit data/z_1_20k.npz --nzeros 500
"""
import argparse
import json
import os

import numpy as np

from src.stats import load


def prime_powers(xmax):
    sieve = np.ones(int(xmax) + 1, bool)
    sieve[:2] = False
    for p in range(2, int(xmax ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    out = []
    for p in np.nonzero(sieve)[0]:
        q, k = int(p), 1
        while q <= xmax:
            out.append((q, int(p), k))
            q, k = q * int(p), k + 1
    return sorted(out)


def psi_exact(x):
    return sum(np.log(p) for q, p, k in prime_powers(x) if q <= x)


def psi_from_zeros(x, g):
    lx = np.log(x)
    terms = 2 * np.sqrt(x) * (0.5 * np.cos(g * lx) + g * np.sin(g * lx)) / (0.25 + g * g)
    return x - terms.sum() - np.log(2 * np.pi) - 0.5 * np.log(1 - x ** -2)


def spectrum_peaks(g, xmin=1.8, xmax=32.0, npts=20000, rel=0.25):
    u = np.linspace(np.log(xmin), np.log(xmax), npts)
    f = np.zeros(npts)
    for gi in g:
        f -= np.cos(gi * u)
    idx = [i for i in range(1, npts - 1) if f[i] > f[i - 1] and f[i] > f[i + 1] and f[i] > rel * f.max()]
    return [float(np.exp(u[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--nzeros", type=int, default=500)
    ap.add_argument("--out", default="results/explicit.json")
    a = ap.parse_args()

    base, x, start = load(a.path)
    if base != 0 or start != 1:
        raise SystemExit("needs a block that starts at the first zero (base 0, index 1)")
    g = x[: a.nzeros]
    peaks = spectrum_peaks(g)
    pp = [q for q, _, _ in prime_powers(32)]
    matched = sorted({min(pp, key=lambda q: abs(q - v)) for v in peaks if min(abs(q - v) for q in pp) < 0.1})
    spurious = [round(v, 2) for v in peaks if min(abs(q - v) for q in pp) >= 0.1]
    psi = {str(t): [round(psi_exact(t), 4), round(float(psi_from_zeros(t, g)), 4)] for t in (10.5, 30.5, 59.5, 100.5)}
    res = {"nzeros": len(g), "peaks_matched": matched, "spurious_peaks": spurious, "psi_exact_vs_zeros": psi}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"{len(g)} zeros: peaks at prime powers {matched}; spurious {spurious}")
    for t, (e, z) in psi.items():
        print(f"psi({t}) exact {e}  from zeros {z}")


if __name__ == "__main__":
    main()
