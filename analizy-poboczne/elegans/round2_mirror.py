"""Runda druga (eksploracyjna): interpolacja A_eps = S + eps*D miedzy czescia lustrzanie symetryczna
a pelnym konektomem, model zerowy z ta sama inwolucja P, IPR w modelu zerowym.

Prerejestracja: prereg2.md (zapisany przed uruchomieniem). Wszystko opisowe.
A_eps dla eps != 1 NIE jest konektomem; to diagnostyka.

Uruchom (w katalogu dane-elegans/, po spectral_gap.py data):
    python round2_mirror.py [--nnull 500]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from spectral_gap import SEED, WORKERS, C, _style, level_stats, mirror_pairs, sectors

EPS = np.round(np.arange(0, 1.0001, 0.1), 10)


def eps_curve(W, perm):
    PWP = W[np.ix_(perm, perm)]
    S, D = (W + PWP) / 2, (W - PWP) / 2
    return [level_stats(np.linalg.eigvalsh(S + e * D))["r_full"] for e in EPS]


def sector_stats(W, perm):
    We, Wo, _ = sectors(W, perm)
    ee, eo = np.linalg.eigvalsh(We), np.linalg.eigvalsh(Wo)
    return dict(even=level_stats(ee)["r_full"], odd=level_stats(eo)["r_full"],
                merged=level_stats(np.concatenate([ee, eo]))["r_full"])


def ipr_stats(W):
    _, v = np.linalg.eigh(W)
    ipr = (v ** 4).sum(axis=0)
    return dict(ipr_mean=float(ipr.mean()), n_ipr_gt_0p1=int((ipr > 0.1).sum()))


def swap_graph(n, edges, w, seed):
    """Identyczna procedura i kolejnosc losowan jak _null_worker('swap') w spectral_gap.py."""
    rng = np.random.default_rng(seed)
    m = len(edges)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    nx.double_edge_swap(g, nswap=10 * m, max_tries=1000 * m, seed=int(rng.integers(2**31)))
    W = np.zeros((n, n))
    for (i, j), x in zip(g.edges(), rng.permutation(w)):
        W[i, j] = W[j, i] = x
    return W


def _worker(args):
    n, edges, w, perm, seed = args
    W = swap_graph(n, edges, w, seed)
    return dict(curve=eps_curve(W, perm), sectors=sector_stats(W, perm), **ipr_stats(W))


def position(null, obs):
    x = np.asarray(null, dtype=float)
    pct = 100 * ((x < obs).sum() + 0.5 * (x == obs).sum()) / len(x)
    return dict(observed=float(obs), null_mean=float(x.mean()), null_sd=float(x.std(ddof=1)),
                null_q025_q975=[float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))], percentile=float(pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnull", type=int, default=500)
    ap.add_argument("--out", default="round2_mirror.json")
    a = ap.parse_args()
    assert os.path.exists("prereg2.md"), "brak prereg2.md: zatrzymuje sie"
    A = np.load("herm_gap_sym_core.npy")
    labels = json.load(open("herm_gap_sym_core_labels.json"))
    n = A.shape[0]
    perm, pairs, unpaired = mirror_pairs(labels)
    assert np.array_equal(perm[perm], np.arange(n)), "P nie jest inwolucja"

    # krok 1
    curve = eps_curve(A, perm)
    sec = sector_stats(A, perm)
    obs_ipr = ipr_stats(A)

    # kroki 2-3: te same ziarna co pierwsze nnull realizacji modelu swap w rundzie pierwszej
    iu = np.triu_indices(n, 1)
    mask = A[iu] != 0
    edges = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
    w = A[iu][mask]
    seeds = np.random.SeedSequence(SEED + 4).spawn(2 * 2000)[:a.nnull]
    with Pool(WORKERS) as pool:
        res = pool.map(_worker, [(n, edges, w, perm, s) for s in seeds], chunksize=10)
    nc = np.array([r["curve"] for r in res])

    out = dict(
        note="A_eps = S + eps*D dla eps != 1 nie jest konektomem; to diagnostyka. Runda eksploracyjna, wyniki opisowe. "
             "Model zerowy: double_edge_swap (nswap = 10 x m), wagi przetasowane miedzy krawedziami (zrywa korelacje "
             "wag ze stopniami); ziarna jak pierwsze nnull realizacji swap z rundy pierwszej.",
        seed=SEED, nnull=a.nnull, n=n, n_pairs=len(pairs), n_unpaired=len(unpaired), eps=EPS.tolist(),
        step1=dict(r_eps=dict(zip([f"{e:.1f}" for e in EPS], curve)), eps0_sectors=sec,
                   monotone_nondecreasing=bool(np.all(np.diff(curve) >= 0)),
                   n_decreasing_steps=int((np.diff(curve) < 0).sum())),
        step2=dict(r_eps0=position(nc[:, 0], curve[0]), r_eps1=position(nc[:, -1], curve[-1]),
                   r_even=position([r["sectors"]["even"] for r in res], sec["even"]),
                   r_odd=position([r["sectors"]["odd"] for r in res], sec["odd"]),
                   r_merged=position([r["sectors"]["merged"] for r in res], sec["merged"]),
                   null_curve_mean=nc.mean(axis=0).tolist(), null_curve_q025=np.quantile(nc, 0.025, axis=0).tolist(),
                   null_curve_q975=np.quantile(nc, 0.975, axis=0).tolist(),
                   null_monotone_fraction=float(np.mean(np.all(np.diff(nc, axis=1) >= 0, axis=1)))),
        step3=dict(ipr_mean=position([r["ipr_mean"] for r in res], obs_ipr["ipr_mean"]),
                   n_ipr_gt_0p1=position([r["n_ipr_gt_0p1"] for r in res], obs_ipr["n_ipr_gt_0p1"])))
    json.dump(out, open(a.out, "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.fill_between(EPS, out["step2"]["null_curve_q025"], out["step2"]["null_curve_q975"], color=C["c2"], alpha=0.18,
                    linewidth=0, label=f"model zerowy: 95% ({a.nnull} grafów)")
    ax.plot(EPS, out["step2"]["null_curve_mean"], color=C["c2"], lw=2, label="model zerowy: średnia")
    ax.plot(EPS, curve, color=C["c1"], lw=2, marker="o", markersize=5, markeredgecolor="white", label="konektom (rdzeń 266)")
    ax.set_xlabel("eps   (A_eps = S + eps·D; eps ≠ 1 to diagnostyka, nie konektom)")
    ax.set_ylabel("średnia r̃")
    ax.set_title("r̃(eps): od części lustrzanie symetrycznej do pełnej macierzy", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout(); fig.savefig("round2_r_eps.png"); plt.close(fig)

    print("r(eps):", " ".join(f"{e:.1f}:{r:.4f}" for e, r in zip(EPS, curve)))
    print(f"eps=0 sektory: {sec}; monotonicznie: {out['step1']['monotone_nondecreasing']} "
          f"(spadki {out['step1']['n_decreasing_steps']}); null monotoniczne: {out['step2']['null_monotone_fraction']:.3f}")
    for k in ("step2", "step3"):
        for kk, v in out[k].items():
            if isinstance(v, dict):
                print(f"{kk}: obs {v['observed']:.4f}, null {v['null_mean']:.4f} ± {v['null_sd']:.4f}, "
                      f"95% [{v['null_q025_q975'][0]:.4f}, {v['null_q025_q975'][1]:.4f}], percentyl {v['percentile']:.1f}")


if __name__ == "__main__":
    main()
