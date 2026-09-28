"""Analiza widmowa macierzy synaps chemicznych C. elegans (herm_chem_272_noauto.npy, 272 x 272, skierowana).

Statystyka: zespolony stosunek odstepow z_k = (l_NN - l_k) / (l_NNN - l_k) (Sa, Ribeiro, Prosen 2020);
r = |z|, cos theta = Re z / |z|, usrednione po punktach widma.
Przetwarzanie (identyczne dla symulacji i danych): tylko Im l > IMTOL (gorna polplaszczyzna; |Im l| <= IMTOL
to wartosci rzeczywiste), odrzucenie |l| < ZTOL, zlanie punktow blizszych niz MERGE_TOL.
Prerejestracja: prereg3.md + prereg3_aneks.md (statystyka glowna, wybrana w kroku 1 przed danymi). Poza rejestrem TERJ.

Uruchom (w katalogu dane-elegans/; numpy, scipy, networkx, matplotlib):
    python chem_spectral.py controls [--nreal 2000]   # krok 1 -> chem_controls.json
    python chem_spectral.py data [--nreal 2000]       # kroki 2-5 -> chem_spectral.json + PNG
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import re
import sys
import time
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from spectral_gap import C, WORKERS, _style

SEED = 20260928
IMTOL = 1e-8       # |Im l| <= IMTOL: wartosc rzeczywista
ZTOL = 1e-8        # |l| < ZTOL: wartosc zerowa, odrzucana
MERGE_TOL = 1e-9   # punkty blizsze niz MERGE_TOL zlewane w jeden
# Wartosci odniesienia z pamieci, DO WERYFIKACJI (Sa, Ribeiro, Prosen, PRX 10, 021019 (2020)):
# 2D Poisson <r> = 2/3, <cos> = 0; GinUE <r> ~ 0.738, <cos> ~ -0.24. Kalibracja kroku 1 zastepuje je symulacja.
REF = dict(poisson=dict(mean_r=2 / 3, mean_cos=0.0), ginibre=dict(mean_r=0.738, mean_cos=-0.24))
TOL_REF = dict(mean_r=0.015, mean_cos=0.03)  # kryterium kontroli: |srednia - odniesienie| < TOL_REF
STATS = ("mean_r", "mean_cos")


# ---------------------------------------------------------------- potok
def merge_points(p, tol=MERGE_TOL):
    if len(p) < 2:
        return p, 0
    pairs = cKDTree(np.column_stack([p.real, p.imag])).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return p, 0
    parent = np.arange(len(p))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(p))])
    uniq, inv = np.unique(roots, return_inverse=True)
    merged = np.array([p[inv == k].mean() for k in range(len(uniq))])
    return merged, int(len(p) - len(merged))


def csr(ev):
    """Zwraca statystyki zespolonego stosunku odstepow po przetwarzaniu oraz liczniki."""
    ev = np.asarray(ev, dtype=complex)
    n_real = int((np.abs(ev.imag) <= IMTOL).sum())
    n_zero = int((np.abs(ev) < ZTOL).sum())
    up = ev[(ev.imag > IMTOL) & (np.abs(ev) >= ZTOL)]
    up, merged = merge_points(up)
    out = dict(n_real=n_real, n_zero=n_zero, n_upper=int(len(up)), merged=merged, mean_r=float("nan"),
               mean_cos=float("nan"))
    if len(up) >= 3:
        _, idx = cKDTree(np.column_stack([up.real, up.imag])).query(np.column_stack([up.real, up.imag]), k=3)
        z = (up[idx[:, 1]] - up) / (up[idx[:, 2]] - up)
        out["mean_r"] = float(np.abs(z).mean())
        out["mean_cos"] = float((z.real / np.abs(z)).mean())
    return out


def summary(W, vectors=True):
    if vectors:
        ev, V = np.linalg.eig(W)
        ipr = (np.abs(V) ** 4).sum(axis=0)
        s = csr(ev)
        s.update(ipr_mean=float(ipr.mean()), n_ipr_gt_0p1=int((ipr > 0.1).sum()))
        return s, ev
    return csr(np.linalg.eigvals(W)), None


def largest_scc(W):
    n, lab = connected_components(csr_matrix(W != 0), directed=True, connection="strong")
    big = np.argmax(np.bincount(lab))
    return np.where(lab == big)[0]


def dist(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), q025=float(np.quantile(x, 0.025)),
                q975=float(np.quantile(x, 0.975)), n=int(len(x)))


def position(null, obs):
    x = np.asarray(null, dtype=float)
    x = x[np.isfinite(x)]
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    return dict(observed=float(obs), null_mean=mu, null_sd=sd, null_q025_q975=[float(np.quantile(x, 0.025)),
                float(np.quantile(x, 0.975))], z=(obs - mu) / sd if sd > 0 else float("nan"),
                percentile=float(100 * ((x < obs).sum() + 0.5 * (x == obs).sum()) / len(x)), n=int(len(x)))


# ---------------------------------------------------------------- krok 1
def _ctrl_worker(args):
    kind, n, m, w, seed = args
    rng = np.random.default_rng(seed)
    if kind == "ginibre":
        return csr(np.linalg.eigvals(rng.standard_normal((n, n)) / math.sqrt(n)))
    if kind == "poisson2d":
        rad, ang = np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * math.pi, n)
        return csr(rad * np.exp(1j * ang))
    g = nx.gnm_random_graph(n, m, seed=int(rng.integers(2**31)), directed=True)
    W = np.zeros((n, n))
    for (i, j), x in zip(g.edges(), rng.permutation(w)):
        W[i, j] = x
    return csr(np.linalg.eigvals(W))


def cmd_controls(a):
    A = np.load("herm_chem_272_noauto.npy")
    n, m, w = A.shape[0], int(np.count_nonzero(A)), A[A != 0]
    seeds = np.random.SeedSequence(SEED).spawn(3 * a.nreal)
    out = dict(seed=SEED, n=n, m_edges=m, density=m / (n * (n - 1)), nreal=a.nreal, imtol=IMTOL, ztol=ZTOL,
               merge_tol=MERGE_TOL, reference_from_memory_TO_VERIFY=REF, tol_ref=TOL_REF, controls={})
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, kind in enumerate(("ginibre", "poisson2d", "er_directed_weighted")):
            res = pool.map(_ctrl_worker, [(kind, n, m, w, s) for s in seeds[i * a.nreal:(i + 1) * a.nreal]],
                           chunksize=20)
            out["controls"][kind] = {k: dist([r[k] for r in res]) for k in STATS + ("n_upper", "n_real", "merged")}
    c = out["controls"]
    sep = {}
    for k in STATS:
        g, p = c["ginibre"][k], c["poisson2d"][k]
        sep[k] = abs(g["mean"] - p["mean"]) / math.sqrt((g["sd"] ** 2 + p["sd"] ** 2) / 2)
    out["separation_poisson_ginibre_in_single_realization_sd"] = sep
    out["separation_definition"] = "|mean_G - mean_P| / sqrt((sd_G^2 + sd_P^2) / 2), sd pojedynczej realizacji"
    out["primary_choice"] = max(sep, key=lambda k: sep[k])
    checks = {f"{kind}_{k}_matches_ref": abs(c[kind][k]["mean"] - REF[ref][k]) < TOL_REF[k]
              for kind, ref in (("ginibre", "ginibre"), ("poisson2d", "poisson")) for k in STATS}
    checks["no_merges_ginibre_poisson"] = c["ginibre"]["merged"]["mean"] == 0 and c["poisson2d"]["merged"]["mean"] == 0
    checks["er_finite"] = c["er_directed_weighted"]["mean_r"]["n"] == a.nreal
    out["checks"], out["all_pass"], out["seconds"] = checks, all(checks.values()), time.time() - t0
    json.dump(out, open(a.out or "chem_controls.json", "w"), indent=1)
    for kind in c:
        print(f"{kind:22s} " + "  ".join(f"{k} {c[kind][k]['mean']:.4f} (sd {c[kind][k]['sd']:.4f})" for k in STATS)
              + f"  n_upper {c[kind]['n_upper']['mean']:.1f}  n_real {c[kind]['n_real']['mean']:.1f}")
    print(f"separacja P-G [sd]: {', '.join(f'{k} {v:.2f}' for k, v in sep.items())}; wybor: {out['primary_choice']}")
    print(f"kontrole: {checks}; ALL_PASS={out['all_pass']}; {out['seconds']:.0f} s")
    return 0 if out["all_pass"] else 1


# ---------------------------------------------------------------- krok 4
def _null_worker(args):
    n, edges, w, seed = args
    rng = np.random.default_rng(seed)
    m = len(edges)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    din, dout = dict(g.in_degree()), dict(g.out_degree())
    nx.directed_edge_swap(g, nswap=10 * m, max_tries=1000 * m, seed=int(rng.integers(2**31)))
    assert dict(g.in_degree()) == din, "zmiana stopni wejsciowych"
    assert dict(g.out_degree()) == dout, "zmiana stopni wyjsciowych"
    assert nx.number_of_selfloops(g) == 0, "petla wlasna"
    assert g.number_of_edges() == m, "krawedz podwojna lub utracona"
    E = np.array(list(g.edges()))
    W = np.zeros((n, n))
    W[E[:, 0], E[:, 1]] = rng.permutation(w)
    B = (W != 0).astype(float)
    sw, _ = summary(W)
    sb, _ = summary(B, vectors=False)
    scc = largest_scc(W)
    ss, _ = summary(W[np.ix_(scc, scc)], vectors=False)
    return dict(weighted=sw, binary=sb, scc=ss, scc_size=int(len(scc)))


def _noise_worker(args):
    A, amp, seed = args
    rng = np.random.default_rng(seed)
    E = rng.standard_normal(A.shape)
    return summary(A + amp * np.linalg.norm(A) * E / np.linalg.norm(E), vectors=False)[0]


# ---------------------------------------------------------------- kroki 2-5
def cmd_data(a):
    pre = open("prereg3.md").read()
    aneks = open("prereg3_aneks.md").read()
    mm = re.search(r"STATYSTYKA_GLOWNA:\s*(mean_r|mean_cos)", aneks)
    assert mm, "brak wyboru statystyki glownej w prereg3_aneks.md: zatrzymuje sie"
    P = mm.group(1)
    S2 = "mean_cos" if P == "mean_r" else "mean_r"
    A = np.load("herm_chem_272_noauto.npy")
    labels = json.load(open("labels_272.json"))
    n = A.shape[0]
    assert not np.diag(A).any()
    out = dict(seed=SEED, n=n, nreal=a.nreal, primary=P, secondary=S2, imtol=IMTOL, ztol=ZTOL, merge_tol=MERGE_TOL,
               prereg3_text=pre, prereg3_aneks_text=aneks)
    t0 = time.time()

    # krok 2
    obs, ev = summary(A)
    zero_rows = [labels[i] for i in np.where(~A.any(axis=1))[0]]
    zero_cols = [labels[i] for i in np.where(~A.any(axis=0))[0]]
    scc = largest_scc(A)
    obs_scc, ev_scc = summary(A[np.ix_(scc, scc)])
    outside = [labels[i] for i in range(n) if i not in set(scc.tolist())]
    abs_sorted = np.sort(np.abs(ev))
    out["step2"] = dict(full=obs, zero_rows=zero_rows, zero_cols=zero_cols, scc_size=int(len(scc)),
                        outside_scc=outside, scc=obs_scc,
                        n_abs_lt={f"{t:g}": int((np.abs(ev) < t).sum()) for t in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)},
                        smallest_abs_eigs=abs_sorted[:12].tolist(),
                        note="Neuron bez wejsc lub bez wyjsc tworzy jednoelementowa silnie spojna skladowa; w postaci "
                             "blokowo-trojkatnej (po skladowych) jej blok 1 x 1 to zero z przekatnej, wiec daje "
                             "wartosc wlasna 0.")

    # krok 4 (przed krokiem 3, bo kryterium stabilnosci uzywa sd modelu zerowego)
    E = np.argwhere(A != 0)
    edges = [tuple(e) for e in E.tolist()]
    w = A[E[:, 0], E[:, 1]]
    seeds = np.random.SeedSequence(SEED + 4).spawn(a.nreal)
    with Pool(WORKERS) as pool:
        null = pool.map(_null_worker, [(n, edges, w, s) for s in seeds], chunksize=5)
    obs_bin, _ = summary((A != 0).astype(float), vectors=False)
    out["step4"] = dict(
        note="Model zerowy: networkx.directed_edge_swap (nswap = 10 x liczba krawedzi) z asercjami: stopnie wejsciowe "
             "i wyjsciowe zachowane, brak petli wlasnych, liczba krawedzi zachowana (brak krawedzi podwojnych). "
             "Wagi przetasowane losowo miedzy krawedziami, co zrywa korelacje wag ze stopniami.",
        m_edges=len(edges), asserts_passed=len(null),
        primary=position([r["weighted"][P] for r in null], obs[P]),
        secondary=position([r["weighted"][S2] for r in null], obs[S2]),
        binary_primary=position([r["binary"][P] for r in null], obs_bin[P]),
        binary_secondary=position([r["binary"][S2] for r in null], obs_bin[S2]),
        binary_observed=obs_bin,
        scc_primary=position([r["scc"][P] for r in null], obs_scc[P]),
        scc_size_null=dist([r["scc_size"] for r in null]),
        n_upper_null=dist([r["weighted"]["n_upper"] for r in null]),
        n_zero_null=dist([r["weighted"]["n_zero"] for r in null]))
    null_sd = out["step4"]["primary"]["null_sd"]

    # krok 3
    nseeds = np.random.SeedSequence(SEED + 3).spawn(300)
    step3 = {}
    with Pool(WORKERS) as pool:
        for k, amp in enumerate((1e-6, 1e-4, 1e-2)):
            res = pool.map(_noise_worker, [(A, amp, s) for s in nseeds[100 * k:100 * (k + 1)]], chunksize=5)
            step3[f"{amp:.0e}"] = dict(primary=dist([r[P] for r in res]), n_zero=dist([r["n_zero"] for r in res]),
                                     n_upper=dist([r["n_upper"] for r in res]), n_real=dist([r["n_real"] for r in res]),
                                     shift_primary_in_null_sd=(float(np.mean([r[P] for r in res])) - obs[P]) / null_sd)
    step3["unstable_flag_at_1e-4"] = bool(abs(step3["1e-04"]["shift_primary_in_null_sd"]) > 1)
    step3["criterion"] = "niestabilny, jesli |srednia statystyki glownej przy 1e-4 - obserwowana| > sd modelu zerowego"
    out["step3"] = step3

    # krok 5
    out["step5"] = dict(ipr_mean=position([r["weighted"]["ipr_mean"] for r in null], obs["ipr_mean"]),
                        n_ipr_gt_0p1=position([r["weighted"]["n_ipr_gt_0p1"] for r in null], obs["n_ipr_gt_0p1"]),
                        n_real=position([r["weighted"]["n_real"] for r in null], obs["n_real"]),
                        eigenvalues_re=ev.real.tolist(), eigenvalues_im=ev.imag.tolist())
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out or "chem_spectral.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    ax.scatter(ev.real, ev.imag, s=12, color=C["c1"], edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=C["muted"], lw=0.6)
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.set_title(f"Widmo macierzy synaps chemicznych (272, ważona); rzeczywistych: {obs['n_real']}",
                 loc="left", fontsize=10)
    _style(ax)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    fig.tight_layout(); fig.savefig("chem_eigs.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
    for ax, key, lab in ((axs[0], "weighted", "ważona"), (axs[1], "binary", "binarna")):
        x = [r[key][P] for r in null]
        o = obs[P] if key == "weighted" else obs_bin[P]
        ax.hist(x, bins=40, color=C["c1"], alpha=0.8)
        ax.axvline(o, color=C["ink"], lw=2)
        ax.text(o, ax.get_ylim()[1] * 0.95, " obserwowane", color=C["ink"], fontsize=8, va="top")
        ax.set_xlabel(f"{P} (macierz {lab})")
        ax.set_title(f"Model zerowy ({a.nreal} grafów, stopnie we/wy zachowane)", loc="left", fontsize=10)
        _style(ax)
    axs[0].set_ylabel("liczba grafów")
    fig.tight_layout(); fig.savefig("chem_null.png"); plt.close(fig)

    s4 = out["step4"]
    print(f"[krok 2] {obs}\n  zerowe wiersze {zero_rows}, kolumny {zero_cols}; |l|<t: {out['step2']['n_abs_lt']}")
    print(f"  SCC {len(scc)} (poza: {outside}): {obs_scc}")
    for k in ("primary", "secondary", "binary_primary", "binary_secondary", "scc_primary"):
        v = s4[k]
        print(f"[krok 4] {k}: obs {v['observed']:.4f} null {v['null_mean']:.4f} ± {v['null_sd']:.4f} "
              f"z {v['z']:+.2f} percentyl {v['percentile']:.1f}")
    for amp in ("1e-06", "1e-04", "1e-02"):
        v = step3[amp]
        print(f"[krok 3] amp {amp}: {P} {v['primary']['mean']:.4f} ± {v['primary']['sd']:.4f} "
              f"(przesuniecie {v['shift_primary_in_null_sd']:+.2f} sd null); n_zero {v['n_zero']['mean']:.1f}")
    print(f"[krok 3] niestabilny przy 1e-4: {step3['unstable_flag_at_1e-4']}")
    for k in ("ipr_mean", "n_ipr_gt_0p1", "n_real"):
        v = out["step5"][k]
        print(f"[krok 5] {k}: obs {v['observed']:.4f} null {v['null_mean']:.4f} ± {v['null_sd']:.4f} "
              f"percentyl {v['percentile']:.1f}")
    print(f"gotowe w {out['seconds']:.0f} s")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["controls", "data"])
    ap.add_argument("--nreal", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.exit(cmd_controls(a) if a.cmd == "controls" else cmd_data(a))


if __name__ == "__main__":
    main()
