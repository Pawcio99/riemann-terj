"""Analiza widmowa rdzenia zlaczy szczelinowych C. elegans (herm_gap_sym_core.npy, 266 x 266).

Prerejestracja: prereg.md (zapisany przed uruchomieniem). Statystyka glowna: srednia r~ macierzy
wazonej, wszystkie poziomy po regule degeneracji, porownanie z modelem zerowym (dwustronnie, p<0,05).

Uruchom (w katalogu dane-elegans/; potrzebne numpy, scipy, networkx, matplotlib):
    python spectral_gap.py controls [--nreal 2000]     # krok 1; wynik: spectral_controls.json
    python spectral_gap.py data [--nreal 2000]         # kroki 2-7; wynik: spectral_gap.json + PNG
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # 6 procesow x 1 watek BLAS

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.polynomial import Polynomial

SEED = 20260928
TOL = 1e-9            # regula degeneracji: poziomy blizsze niz TOL zlewane w jeden
WORKERS = 6
# Wartosci odniesienia z pamieci (DO WERYFIKACJI): Poisson 2 ln 2 - 1 = 0.3863 (analitycznie),
# GOE ~0.5307 (Atas i in. 2013, PRL 110, 084101). Kontrole kroku 1 zastepuja je symulacja.
REF_POISSON = 2 * math.log(2) - 1
REF_GOE = 0.5307
CTRL_TOL = 0.005      # kryterium zaliczenia kontroli (a), (b): |srednia - odniesienie| < CTRL_TOL
C = dict(c1="#2a78d6", c2="#eb6834", c3="#1baf7a", ink="#1f1f1e", muted="#6b6a64", grid="#e4e3dc")


# ---------------------------------------------------------------- potok poziomow
def merge_levels(ev, tol=TOL):
    """Sortuje i zlewa poziomy blizsze niz tol (lancuchowo). Zwraca (poziomy, liczba zlanych, krotnosc zera)."""
    ev = np.sort(np.asarray(ev, dtype=float))
    zero_mult = int((np.abs(ev) < tol).sum())
    if len(ev) == 0:
        return ev, 0, zero_mult
    new_cluster = np.concatenate([[True], np.diff(ev) >= tol])
    cid = np.cumsum(new_cluster) - 1
    lev = np.bincount(cid, weights=ev) / np.bincount(cid)
    return lev, int(len(ev) - len(lev)), zero_mult


def central(lev, frac):
    n = len(lev)
    k = int(round(n * (1 - frac) / 2))
    return lev[k:n - k]


def rtilde(lev):
    s = np.diff(lev)
    a, b = s[1:], s[:-1]
    return np.minimum(a, b) / np.maximum(a, b)


def level_stats(ev):
    lev, merged, zm = merge_levels(ev)
    return dict(r_full=float(rtilde(lev).mean()), r_c80=float(rtilde(central(lev, 0.8)).mean()),
                r_c50=float(rtilde(central(lev, 0.5)).mean()), n_levels=int(len(lev)), merged=merged, zero_mult=zm)


def matrix_stats(W):
    """Ten sam potok dla macierzy wazonej, binarnej i laplasjanu L = D - A (wazonego)."""
    B = (W != 0).astype(float)
    Lap = np.diag(W.sum(axis=1)) - W
    return dict(weighted=level_stats(np.linalg.eigvalsh(W)), binary=level_stats(np.linalg.eigvalsh(B)),
                laplacian=level_stats(np.linalg.eigvalsh(Lap)))


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return dict(mean=m, sd=float(x.std(ddof=1)), ci95_of_mean=[m - 1.96 * se, m + 1.96 * se], n=len(x))


def goe(n, rng):
    G = rng.standard_normal((n, n))
    return (G + G.T) / math.sqrt(2 * n)


# ---------------------------------------------------------------- krok 1
def _ctrl_worker(args):
    kind, n, m, seed = args
    rng = np.random.default_rng(seed)
    if kind == "goe":
        return level_stats(np.linalg.eigvalsh(goe(n, rng)))
    if kind == "poisson":
        return level_stats(rng.uniform(0, n, n))
    g = nx.gnm_random_graph(n, m, seed=int(rng.integers(2**31)))
    return level_stats(np.linalg.eigvalsh(nx.to_numpy_array(g)))


def cmd_controls(a):
    A = np.load("herm_gap_sym_core.npy")
    n, m = A.shape[0], int(np.count_nonzero(np.triu(A, 1)))
    seeds = np.random.SeedSequence(SEED).spawn(3 * a.nreal)
    out = dict(seed=SEED, tol=TOL, n=n, m_edges=m, density=2 * m / (n * (n - 1)), nreal=a.nreal,
               reference_from_memory_TO_VERIFY=dict(poisson_2ln2_minus_1=REF_POISSON, goe=REF_GOE), controls={})
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, kind in enumerate(("goe", "poisson", "er_binary")):
            res = pool.map(_ctrl_worker, [(kind, n, m, s) for s in seeds[i * a.nreal:(i + 1) * a.nreal]], chunksize=20)
            out["controls"][kind] = {k: mean_ci([r[k] for r in res]) for k in ("r_full", "r_c80", "r_c50")}
            out["controls"][kind]["merged_mean"] = float(np.mean([r["merged"] for r in res]))
            out["controls"][kind]["zero_mult_mean"] = float(np.mean([r["zero_mult"] for r in res]))
    c = out["controls"]
    g, p, e = c["goe"]["r_full"], c["poisson"]["r_full"], c["er_binary"]["r_full"]
    checks = dict(goe_matches_ref=abs(g["mean"] - REF_GOE) < CTRL_TOL,
                  poisson_matches_ref=abs(p["mean"] - REF_POISSON) < CTRL_TOL,
                  goe_no_merges=c["goe"]["merged_mean"] == 0 and c["poisson"]["merged_mean"] == 0,
                  er_between_poisson_and_goe=p["ci95_of_mean"][0] < e["mean"] < g["ci95_of_mean"][1])
    out["checks"], out["all_pass"], out["seconds"] = checks, all(checks.values()), time.time() - t0
    json.dump(out, open(a.out or "spectral_controls.json", "w"), indent=1)
    for kind in ("goe", "poisson", "er_binary"):
        r = c[kind]["r_full"]
        print(f"{kind:10s} r~ {r['mean']:.4f}  95% CI sredniej [{r['ci95_of_mean'][0]:.4f}, {r['ci95_of_mean'][1]:.4f}]"
              f"  sd {r['sd']:.4f}  zlane {c[kind]['merged_mean']:.2f}  krotnosc 0 {c[kind]['zero_mult_mean']:.2f}")
    print(f"n={n}, m={m}; kontrole: {checks}; ALL_PASS={out['all_pass']}; {out['seconds']:.0f} s")
    return 0 if out["all_pass"] else 1


# ---------------------------------------------------------------- krok 4
def _null_worker(args):
    kind, n, edges, w, perm, seed = args
    rng = np.random.default_rng(seed)
    m = len(edges)
    if kind == "swap":
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        nx.double_edge_swap(g, nswap=10 * m, max_tries=1000 * m, seed=int(rng.integers(2**31)))
    else:
        g = nx.gnm_random_graph(n, m, seed=int(rng.integers(2**31)))
    W = np.zeros((n, n))
    ww = rng.permutation(w)
    for (i, j), x in zip(g.edges(), ww):
        W[i, j] = W[j, i] = x
    st = matrix_stats(W)
    st["mirror_asym"] = float(np.linalg.norm(W[np.ix_(perm, perm)] - W) / np.linalg.norm(W))
    st["n_components"] = nx.number_connected_components(g)
    return st


# ---------------------------------------------------------------- krok 5
def mirror_pairs(labels):
    idx = {l: i for i, l in enumerate(labels)}
    perm, pairs, unpaired = list(range(len(labels))), [], []
    for i, l in enumerate(labels):
        mate = l[:-1] + {"L": "R", "R": "L"}[l[-1]] if l[-1] in "LR" else None
        if mate is not None and mate in idx:
            perm[i] = idx[mate]
            if l[-1] == "L":
                pairs.append((l, mate))
        else:
            unpaired.append(l)
    return np.array(perm), pairs, unpaired


def sectors(A, perm):
    n = len(perm)
    ev_cols, od_cols = [], []
    for i in range(n):
        j = perm[i]
        if j == i:
            e = np.zeros(n); e[i] = 1; ev_cols.append(e)
        elif i < j:
            e = np.zeros(n); e[i] = e[j] = 1 / math.sqrt(2); ev_cols.append(e)
            o = np.zeros(n); o[i], o[j] = 1 / math.sqrt(2), -1 / math.sqrt(2); od_cols.append(o)
    Qe, Qo = np.array(ev_cols).T, np.array(od_cols).T
    return Qe.T @ A @ Qe, Qo.T @ A @ Qo, Qe.T @ A @ Qo


def _superpos_worker(args):
    ne, no, seed = args
    rng = np.random.default_rng(seed)
    ee, eo = np.linalg.eigvalsh(goe(ne, rng)), np.linalg.eigvalsh(goe(no, rng))
    return dict(merged=level_stats(np.concatenate([ee, eo]))["r_full"], even=level_stats(ee)["r_full"],
                odd=level_stats(eo)["r_full"])


# ---------------------------------------------------------------- krok 7
def unfold_spacings(lev, deg):
    p = Polynomial.fit(lev, np.arange(1, len(lev) + 1), deg)
    s = np.diff(p(lev))
    return s / s.mean()


def _ps_worker(args):
    kind, n, seed = args
    rng = np.random.default_rng(seed)
    ev = np.linalg.eigvalsh(goe(n, rng)) if kind == "goe" else rng.uniform(0, n, n)
    lev = merge_levels(ev)[0]
    return {d: unfold_spacings(lev, d) for d in (5, 7, 9)}


# ---------------------------------------------------------------- kroki 2-7
def summarize_null(res, obs, key):
    x = np.array([r[key[0]][key[1]] if len(key) == 2 else r[key[0]] for r in res])
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    p = (1 + int((np.abs(x - mu) >= abs(obs - mu) - 1e-15).sum())) / (len(x) + 1)
    return dict(observed=obs, null_mean=mu, null_sd=sd, null_q025_q975=[float(np.quantile(x, 0.025)),
                float(np.quantile(x, 0.975))], z=(obs - mu) / sd, p_two_sided_empirical=p)


def cmd_data(a):
    prereg = open("prereg.md").read()
    A = np.load("herm_gap_sym_core.npy")
    labels = json.load(open("herm_gap_sym_core_labels.json"))
    n = A.shape[0]
    assert np.array_equal(A, A.T) and not np.diag(A).any() and A.any(axis=1).all()
    out = dict(seed=SEED, tol=TOL, n=n, nreal=a.nreal, prereg_sha_note="prereg.md niezmieniony; patrz prereg.sha256",
               prereg_text=prereg)
    t0 = time.time()

    # krok 2
    evals, evecs = np.linalg.eigh(A)
    lev, merged, zm = merge_levels(evals)
    Bn = (A != 0)
    deg = Bn.sum(axis=1)
    nbr = [frozenset(np.where(Bn[i])[0]) for i in range(n)]
    twins = [(labels[i], labels[j]) for i in range(n) for j in range(i + 1, n) if nbr[i] == nbr[j]]
    wtwins = [(labels[i], labels[j]) for i in range(n) for j in range(i + 1, n) if nbr[i] == nbr[j]
              and np.array_equal(np.delete(A[i], [i, j]), np.delete(A[j], [i, j]))]
    out["step2"] = dict(n_eigs=n, n_levels_after_merge=len(lev), merged=merged, zero_multiplicity=zm,
                        degree_1=[labels[i] for i in np.where(deg == 1)[0]], n_degree_1=int((deg == 1).sum()),
                        twin_pairs_identical_neighbourhood=twins, n_twin_pairs=len(twins),
                        twin_pairs_identical_weights=wtwins, n_twin_pairs_identical_weights=len(wtwins))

    # krok 3
    obs = matrix_stats(A)
    out["step3"] = obs

    # krok 4
    iu = np.triu_indices(n, 1)
    mask = A[iu] != 0
    edges = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
    w = A[iu][mask]
    perm, pairs, unpaired = mirror_pairs(labels)
    seeds = np.random.SeedSequence(SEED + 4).spawn(2 * a.nreal)
    with Pool(WORKERS) as pool:
        null_swap = pool.map(_null_worker, [("swap", n, edges, w, perm, s) for s in seeds[:a.nreal]], chunksize=10)
        null_er = pool.map(_null_worker, [("er", n, edges, w, perm, s) for s in seeds[a.nreal:]], chunksize=10)
    step4 = dict(note="Wagi krawedzi przetasowane losowo miedzy krawedziami: zrywa to korelacje wag ze stopniami. "
                      "Model swap: double_edge_swap, nswap = 10 x liczba krawedzi, zachowuje stopnie. "
                      "Model ER: G(n, m) z ta sama liczba krawedzi i tymi samymi (przetasowanymi) wagami.",
                 m_edges=len(edges), primary=summarize_null(null_swap, obs["weighted"]["r_full"], ("weighted", "r_full")))
    for name, res in (("swap", null_swap), ("er", null_er)):
        step4[name] = {f"{v}_{k}": summarize_null(res, obs[v][k], (v, k))
                       for v in ("weighted", "binary", "laplacian") for k in ("r_full", "r_c80", "r_c50")}
        step4[name]["zero_mult_null_mean"] = float(np.mean([r["weighted"]["zero_mult"] for r in res]))
        step4[name]["merged_null_mean"] = float(np.mean([r["weighted"]["merged"] for r in res]))
        step4[name]["n_components_null_mean"] = float(np.mean([r["n_components"] for r in res]))
    out["step4"] = step4

    # krok 5
    asym = float(np.linalg.norm(A[np.ix_(perm, perm)] - A) / np.linalg.norm(A))
    Ae, Ao, Ceo = sectors(A, perm)
    ev_e, ev_o = np.linalg.eigvalsh(Ae), np.linalg.eigvalsh(Ao)
    sseeds = np.random.SeedSequence(SEED + 5).spawn(a.nreal)
    with Pool(WORKERS) as pool:
        sup = pool.map(_superpos_worker, [(Ae.shape[0], Ao.shape[0], s) for s in sseeds], chunksize=20)
    short = [p for p in pairs if len(p[0]) <= 3]
    out["step5"] = dict(n_pairs=len(pairs), n_unpaired=len(unpaired), unpaired=unpaired, pairs=pairs,
                        short_name_pairs_to_review=short, mirror_asym_frob_rel=asym,
                        mirror_asym_null=summarize_null(null_swap, asym, ("mirror_asym",)),
                        mirror_asym_null_er=summarize_null(null_er, asym, ("mirror_asym",)),
                        even_dim=int(Ae.shape[0]), odd_dim=int(Ao.shape[0]),
                        even_odd_coupling_frob_rel=float(np.linalg.norm(Ceo) * math.sqrt(2) / np.linalg.norm(A)),
                        r_even=level_stats(ev_e), r_odd=level_stats(ev_o),
                        r_even_plus_odd_merged=level_stats(np.concatenate([ev_e, ev_o])),
                        goe_superposition={k: mean_ci([r[k] for r in sup]) for k in ("merged", "even", "odd")})

    # krok 6
    ipr = (evecs ** 4).sum(axis=0)
    out["step6"] = dict(ipr_goe_expected_3_over_n_plus_2=3 / (n + 2), ipr_mean=float(ipr.mean()),
                        ipr_median=float(np.median(ipr)), ipr_max=float(ipr.max()),
                        n_ipr_gt_0p1=int((ipr > 0.1).sum()), n_ipr_gt_0p25=int((ipr > 0.25).sum()),
                        eigenvalues=evals.tolist(), ipr=ipr.tolist())
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.scatter(evals, ipr, s=14, color=C["c1"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(3 / (n + 2), color=C["muted"], lw=1, ls="--")
    ax.text(evals.max(), 3 / (n + 2) * 1.15, "GOE: 3/(N+2)", ha="right", va="bottom", color=C["muted"], fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("wartość własna λ (macierz ważona, rdzeń 266)")
    ax.set_ylabel("IPR = Σ v⁴")
    ax.set_title("Lokalizacja wektorów własnych: rdzeń złączy szczelinowych", loc="left", fontsize=10)
    _style(ax)
    fig.tight_layout(); fig.savefig("spectral_ipr.png"); plt.close(fig)

    # krok 7
    pseeds = np.random.SeedSequence(SEED + 7).spawn(2 * a.nps)
    with Pool(WORKERS) as pool:
        sg = pool.map(_ps_worker, [("goe", n, s) for s in pseeds[:a.nps]], chunksize=10)
        sp = pool.map(_ps_worker, [("poisson", n, s) for s in pseeds[a.nps:]], chunksize=10)
    bins = np.arange(0, 4.01, 0.2)
    step7 = dict(bins=bins.tolist(), n_sim=a.nps, degrees={})
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150, sharey=True)
    for ax, d in zip(axs, (5, 7, 9)):
        s = unfold_spacings(lev, d)
        g = np.concatenate([r[d] for r in sg]); p = np.concatenate([r[d] for r in sp])
        h = {k: np.histogram(v, bins=bins, density=True)[0] for k, v in (("data", s), ("goe", g), ("poisson", p))}
        step7["degrees"][d] = dict(
            n_nonpositive_spacings=int((s <= 0).sum()),
            p_s_lt_0p25={k: float((v < 0.25).mean()) for k, v in (("data", s), ("goe", g), ("poisson", p))},
            var_s={k: float(v.var()) for k, v in (("data", s), ("goe", g), ("poisson", p))},
            hist={k: v.tolist() for k, v in h.items()})
        mid = (bins[:-1] + bins[1:]) / 2
        ax.plot(mid, h["goe"], color=C["c2"], lw=2, label="GOE (symulacja)")
        ax.plot(mid, h["poisson"], color=C["c3"], lw=2, label="Poisson (symulacja)")
        ax.bar(mid, h["data"], width=0.2 - 0.02, color=C["c1"], alpha=0.85, label="rdzeń 266", zorder=0)
        ax.set_title(f"rozwinięcie: wielomian st. {d}", loc="left", fontsize=10)
        ax.set_xlabel("s")
        _style(ax)
    axs[0].set_ylabel("P(s)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("spectral_ps.png"); plt.close(fig)
    out["step7"] = step7

    # rozklad modelu zerowego dla statystyki glownej
    x = np.array([r["weighted"]["r_full"] for r in null_swap])
    xe = np.array([r["weighted"]["r_full"] for r in null_er])
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(x, bins=40, color=C["c1"], alpha=0.8, label="swap (stopnie zachowane)")
    ax.hist(xe, bins=40, color=C["c2"], alpha=0.6, label="Erdős–Rényi (ta sama liczba krawędzi)")
    ax.axvline(obs["weighted"]["r_full"], color=C["ink"], lw=2)
    ax.text(obs["weighted"]["r_full"], ax.get_ylim()[1] * 0.95, " obserwowane", color=C["ink"], fontsize=8, va="top")
    ax.set_xlabel("średnia r̃ (macierz ważona, wszystkie poziomy)")
    ax.set_ylabel("liczba grafów")
    ax.set_title(f"Model zerowy: {a.nreal} realizacji na model", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout(); fig.savefig("spectral_null.png"); plt.close(fig)

    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out or "spectral_gap.json", "w"), indent=1)
    print(f"gotowe w {out['seconds']:.0f} s")
    return 0


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C["muted"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["controls", "data"])
    ap.add_argument("--nreal", type=int, default=2000)
    ap.add_argument("--nps", type=int, default=500, help="realizacje GOE/Poisson dla P(s) w kroku 7")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.exit(cmd_controls(a) if a.cmd == "controls" else cmd_data(a))


if __name__ == "__main__":
    main()
