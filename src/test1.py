"""Test 1: GOE admixture in the statistics of zeros vs CUE of effective size N_eff(T).

    python -m src.test1 --out results/test1.json [--nrep 200]

N_eff = ln(T/2pi)/sqrt(12*Lambda), Lambda = 1.57314 (Bogomolny et al. 2006,
J. Phys. A 39, 10743, Eq. (19); checked against arXiv math/0602270).
Null: CUE with (non-integer) N_eff realised as a mixture of floor/ceil sizes, N >= 2.
Alternative: Pandey-Mehta crossover S + i*alpha*A (alpha=0 GOE, alpha=1 GUE), binned likelihood.
p-values come from null replicas of the same size and the same estimators.
Two unfoldings: mean density N'(T) ("dens") and division by a moving local mean, W=100 ("loc").
"""
import argparse
import json
import os

import numpy as np

from src import rmt, stats

LAM = 1.57314
EDGES = np.concatenate([np.arange(0, 3.0, 0.1), [3.0, 4.0, np.inf]])
ALPHAS = [0.0, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
W = 100
NB = 10000  # spacings per block


def neff(t):
    return np.log(t / (2 * np.pi)) / np.sqrt(12 * LAM)


def cue_batch(n, m, rng):
    z = (rng.normal(size=(m, n, n)) + 1j * rng.normal(size=(m, n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=1, axis2=2)
    q = q * (d / np.abs(d))[:, None, :]
    th = np.sort(np.angle(np.linalg.eigvals(q)), axis=1)
    nxt = np.concatenate([th[:, 1:], th[:, :1] + 2 * np.pi], axis=1)
    return ((nxt - th) * n / (2 * np.pi)).ravel()


def cue_stream(ne, nsp, rng):
    """nsp spacings from CUE of non-integer effective size ne (mixture of floor/ceil, N>=2)."""
    ne = max(ne, 2.0)
    lo = int(np.floor(ne))
    w_hi = ne - lo
    parts = []
    for n, w in ((lo, 1 - w_hi), (lo + 1, w_hi)):
        k = int(round(nsp * w))
        if k > 0:
            parts.append(cue_batch(n, -(-k // n), rng)[:k])
    s = np.concatenate(parts)
    rng.shuffle(s)
    return s[:nsp]


def local_norm(s, w=W):
    ma = np.convolve(s, np.ones(2 * w + 1) / (2 * w + 1), mode="valid")
    return s[w:-w] / ma


def ks_stat(s, ref):
    """sup |F_s - F_ref|, ref sorted."""
    s = np.sort(s)
    f = np.searchsorted(ref, s, side="right") / len(ref)
    i = np.arange(1, len(s) + 1) / len(s)
    return max(np.max(i - f), np.max(f - (i - 1 / len(s))))


def a_mle(s):
    v = s[(s > 0) & (s < 0.25)]
    return len(v) / np.sum(np.log(0.25 / v))


def alt_table(nm, seed):
    rng = np.random.default_rng(seed)
    out = []
    for al in ALPHAS:
        s = np.concatenate([rmt.crossover_sample(200, al, rng) for _ in range(nm)])
        out.append(np.histogram(s, bins=EDGES)[0] + 1.0)
    p = np.array(out)
    return np.log(p / p.sum(axis=1, keepdims=True))


def lr_stat(s, logp):
    h = np.histogram(s, bins=EDGES)[0]
    ll = logp @ h
    return 2 * (ll.max() - ll[-1]), ALPHAS[int(np.argmax(ll))]



def block_stats(s, ref, logp):
    a = a_mle(s)
    return a, ks_stat(s, ref), lr_stat(s, logp)[0]


def analyse(path, ref_d, ref_l, logp, nrep, seed):
    base, x, _ = stats.load(path)
    s_d = stats.unfold(base, x)
    t = float(base) + float(np.mean(x))
    ne = float(neff(t))
    s_l = local_norm(s_d)
    rng = np.random.default_rng(seed)
    res = {"file": os.path.basename(path), "T": t, "N_eff": ne, "n": len(s_d)}
    for tag, s, ref in (("dens", s_d, ref_d), ("loc", s_l, ref_l)):
        d = block_stats(s, ref, logp)
        nul = []
        for _ in range(nrep):
            c = cue_stream(ne, len(s_d) + (2 * W if tag == "loc" else 0), rng)
            c = local_norm(c)[: len(s)] if tag == "loc" else c
            nul.append(block_stats(c, ref, logp))
        nul = np.array(nul)
        mu, sd = nul.mean(axis=0), nul.std(axis=0)
        res[tag] = {
            "a": d[0], "a_null": float(mu[0]), "a_null_sd": float(sd[0]), "z_a": float((d[0] - mu[0]) / sd[0]),
            "ks": d[1], "p_ks": float((1 + np.sum(nul[:, 1] >= d[1])) / (nrep + 1)),
            "lr": d[2], "p_lr": float((1 + np.sum(nul[:, 2] >= d[2])) / (nrep + 1)),
            "alpha_hat": lr_stat(s, logp)[1],
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--nm", type=int, default=1500, help="crossover matrices per alpha")
    a = ap.parse_args()
    files = a.files or [f"data/z_1e{e}_10k.npz" for e in range(3, 9)] + [f"data/odl_1e{e}.npz" for e in (12, 21, 22)]
    files = [f for f in files if os.path.exists(f)]
    logp = alt_table(a.nm, 1)
    # reference null (pooled) for KS, one per unfolding, at a fixed mid-range N_eff per block below
    out = []
    for i, f in enumerate(files):
        base, x, _ = stats.load(f)
        ne = float(neff(float(base) + float(np.mean(x))))
        rr = np.random.default_rng(1000 + i)
        ref_d = np.sort(cue_stream(ne, 20 * NB, rr))
        ref_l = np.sort(local_norm(cue_stream(ne, 20 * NB, rr)))
        r = analyse(f, ref_d, ref_l, logp, a.nrep, 2000 + i)
        out.append(r)
        d, l = r["dens"], r["loc"]
        print(f"{r['file']:18s} T={r['T']:.3g} Neff={r['N_eff']:.2f} | dens a={d['a']:.2f} null {d['a_null']:.2f}±{d['a_null_sd']:.2f} "
              f"z={d['z_a']:+.1f} pKS={d['p_ks']:.3f} pLR={d['p_lr']:.3f} α̂={d['alpha_hat']} | loc z={l['z_a']:+.1f} "
              f"pKS={l['p_ks']:.3f} pLR={l['p_lr']:.3f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
