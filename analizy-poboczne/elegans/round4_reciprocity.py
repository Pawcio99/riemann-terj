"""Runda 4 (eksploracyjna, poza rejestrem TERJ): wzajemnosc polaczen a nadmiar wartosci rzeczywistych.

Prerejestracja: prereg4.md. Wszystko opisowe.
Krok 1: pary wzajemne / jednokierunkowe w danych i w 2000 grafach modelu zerowego z rundy 3 (odtworzonych
        z tych samych ziaren; zgodnosc z chem_spectral.json sprawdzana asercja), korelacje Spearmana.
Krok 2-3: model zerowy zachowujacy stopnie wejsciowe, wyjsciowe i wzajemne kazdego neuronu (a wiec i liczbe
        par wzajemnych); ta sama procedura widmowa co w rundzie 3.

Uruchom (w katalogu dane-elegans/, po chem_spectral.py data):  python round4_reciprocity.py [--nreal 2000]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from chem_spectral import SEED, WORKERS, dist, position, summary

KEYS = ("n_real", "mean_cos", "mean_r")


def recip_counts(W):
    B = W != 0
    rec = int(np.triu(B & B.T, 1).sum())
    return dict(reciprocal_pairs=rec, unidirectional_edges=int(B.sum()) - 2 * rec)


def spectral(W):
    sw, _ = summary(W)
    sb, _ = summary((W != 0).astype(float), vectors=False)
    return dict(weighted={k: sw[k] for k in KEYS}, binary={k: sb[k] for k in KEYS})


# ---------------------------------------------------------------- krok 1: odtworzenie modelu z rundy 3
def _regen_worker(args):
    """Identyczna procedura i kolejnosc losowan jak _null_worker w chem_spectral.py."""
    n, edges, w, seed = args
    rng = np.random.default_rng(seed)
    m = len(edges)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    nx.directed_edge_swap(g, nswap=10 * m, max_tries=1000 * m, seed=int(rng.integers(2**31)))
    E = np.array(list(g.edges()))
    W = np.zeros((n, n))
    W[E[:, 0], E[:, 1]] = rng.permutation(w)
    return dict(**recip_counts(W), **spectral(W))


# ---------------------------------------------------------------- krok 2: model z zachowana wzajemnoscia
def recip_swap(n, rec_pairs, uni_edges, rec_w, uni_w, seed):
    rng = np.random.default_rng(seed)
    E = set()
    for i, j in rec_pairs:
        E.add((i, j)); E.add((j, i))
    E.update(uni_edges)
    rec = [list(p) for p in rec_pairs]
    uni = [list(e) for e in uni_edges]
    acc = dict(rec=0, uni=0)

    # pary wzajemne: podwojna zamiana w grafie nieskierowanym {a,b},{c,d} -> {a,d},{c,b}
    for _ in range(10 * len(rec)):
        k, l = rng.integers(len(rec), size=2)
        if k == l:
            continue
        (a, b), (c, d) = rec[k], rec[l]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or (a, d) in E or (d, a) in E or (c, b) in E or (b, c) in E:
            continue
        for x, y in ((a, b), (c, d)):
            E.discard((x, y)); E.discard((y, x))
        for x, y in ((a, d), (c, b)):
            E.add((x, y)); E.add((y, x))
        rec[k], rec[l] = [a, d], [c, b]
        acc["rec"] += 1

    # krawedzie jednokierunkowe: (a->b),(c->d) -> (a->d),(c->b), bez tworzenia par wzajemnych
    for _ in range(10 * len(uni)):
        k, l = rng.integers(len(uni), size=2)
        if k == l:
            continue
        (a, b), (c, d) = uni[k], uni[l]
        if a == d or c == b or (a, d) in E or (c, b) in E or (d, a) in E or (b, c) in E:
            continue
        E.discard((a, b)); E.discard((c, d))
        E.add((a, d)); E.add((c, b))
        uni[k], uni[l] = [a, d], [c, b]
        acc["uni"] += 1

    W = np.zeros((n, n))
    rec_dir = [(x, y) for x, y in rec] + [(y, x) for x, y in rec]
    for (x, y), v in zip(rec_dir, rng.permutation(rec_w)):
        W[x, y] = v
    for (x, y), v in zip(uni, rng.permutation(uni_w)):
        W[x, y] = v
    return W, acc


def degree_seqs(W):
    B = W != 0
    R = B & B.T
    return B.sum(axis=0), B.sum(axis=1), R.sum(axis=1)


def _recip_worker(args):
    n, rec_pairs, uni_edges, rec_w, uni_w, ref, seed = args
    W, acc = recip_swap(n, rec_pairs, uni_edges, rec_w, uni_w, seed)
    din, dout, drec = degree_seqs(W)
    assert np.array_equal(din, ref["in"]), "zmiana stopni wejsciowych"
    assert np.array_equal(dout, ref["out"]), "zmiana stopni wyjsciowych"
    assert np.array_equal(drec, ref["rec"]), "zmiana stopni wzajemnych"
    assert not np.diag(W).any(), "petla wlasna"
    assert int(np.count_nonzero(W)) == ref["m"], "krawedz podwojna lub utracona"
    rc = recip_counts(W)
    assert rc["reciprocal_pairs"] == ref["n_rec"], "zmiana liczby par wzajemnych"
    return dict(**rc, **spectral(W), accepted=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreal", type=int, default=2000)
    ap.add_argument("--out", default="round4_reciprocity.json")
    a = ap.parse_args()
    assert os.path.exists("prereg4.md"), "brak prereg4.md: zatrzymuje sie"
    A = np.load("herm_chem_272_noauto.npy")
    n = A.shape[0]
    t0 = time.time()
    obs = dict(**recip_counts(A), **spectral(A))

    # krok 1
    E = np.argwhere(A != 0)
    edges = [tuple(e) for e in E.tolist()]
    w = A[E[:, 0], E[:, 1]]
    seeds3 = np.random.SeedSequence(SEED + 4).spawn(2000)[:a.nreal]
    with Pool(WORKERS) as pool:
        old = pool.map(_regen_worker, [(n, edges, w, s) for s in seeds3], chunksize=5)
    r3 = json.load(open("chem_spectral.json"))["step4"]
    reproduced = None
    if a.nreal == r3["asserts_passed"]:
        chk = dict(weighted_cos=(r3["primary"], [r["weighted"]["mean_cos"] for r in old]),
                   weighted_r=(r3["secondary"], [r["weighted"]["mean_r"] for r in old]),
                   binary_cos=(r3["binary_primary"], [r["binary"]["mean_cos"] for r in old]))
        reproduced = {k: bool(abs(np.mean(x) - ref["null_mean"]) < 1e-12 and abs(np.std(x, ddof=1) - ref["null_sd"]) < 1e-12)
                      for k, (ref, x) in chk.items()}
        assert all(reproduced.values()), f"model z rundy 3 nie odtworzony: {reproduced}"
    step1 = dict(observed=recip_counts(A), reproduced_round3_null=reproduced,
                 reciprocal_pairs=position([r["reciprocal_pairs"] for r in old], obs["reciprocal_pairs"]),
                 unidirectional_edges=position([r["unidirectional_edges"] for r in old], obs["unidirectional_edges"]),
                 spearman_nreal_vs_cos={}, spearman_nreal_vs_reciprocal={})
    for v in ("weighted", "binary"):
        nr = [r[v]["n_real"] for r in old]
        rho, p = spearmanr(nr, [r[v]["mean_cos"] for r in old])
        step1["spearman_nreal_vs_cos"][v] = dict(rho=float(rho), p=float(p))
        rho, p = spearmanr(nr, [r["reciprocal_pairs"] for r in old])
        step1["spearman_nreal_vs_reciprocal"][v] = dict(rho=float(rho), p=float(p))
    step1["round3_null_n_real_binary"] = position([r["binary"]["n_real"] for r in old], obs["binary"]["n_real"])

    # kroki 2-3
    B = A != 0
    rec_pairs = [(int(i), int(j)) for i, j in np.argwhere(np.triu(B & B.T, 1))]
    uni_edges = [(int(i), int(j)) for i, j in np.argwhere(B & ~B.T)]
    rec_w = np.array([A[i, j] for i, j in rec_pairs] + [A[j, i] for i, j in rec_pairs])
    uni_w = np.array([A[i, j] for i, j in uni_edges])
    din, dout, drec = degree_seqs(A)
    ref = dict(**{"in": din, "out": dout, "rec": drec}, m=int(B.sum()), n_rec=len(rec_pairs))
    seeds = np.random.SeedSequence(SEED + 40).spawn(a.nreal)
    with Pool(WORKERS) as pool:
        new = pool.map(_recip_worker, [(n, rec_pairs, uni_edges, rec_w, uni_w, ref, s) for s in seeds], chunksize=5)
    step3 = {v: {k: position([r[v][k] for r in new], obs[v][k]) for k in KEYS} for v in ("weighted", "binary")}
    out = dict(
        note="Runda eksploracyjna, poza rejestrem TERJ; wszystko opisowe. Nowy model: zamiany osobno w klasie par "
             "wzajemnych (podwojna zamiana w grafie nieskierowanym) i krawedzi jednokierunkowych (zamiana (a->b),(c->d) "
             "-> (a->d),(c->b)), z odrzucaniem zamian tworzacych lub niszczacych pary wzajemne, petle lub krawedzie "
             "podwojne; nswap = 10 x liczba elementow klasy. Wagi przetasowane w obrebie klasy (miedzy krawedziami "
             "skierowanymi), co zrywa korelacje wag ze stopniami i sparowanie wag w parach wzajemnych.",
        seed=SEED, nreal=a.nreal, observed=obs, step1=step1,
        step2=dict(asserts_passed=len(new), n_reciprocal_pairs=len(rec_pairs), n_unidirectional=len(uni_edges),
                   accepted_swaps_rec=dist([r["accepted"]["rec"] for r in new]),
                   accepted_swaps_uni=dist([r["accepted"]["uni"] for r in new]),
                   attempts_rec=10 * len(rec_pairs), attempts_uni=10 * len(uni_edges)),
        step3=step3, seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)

    s1 = step1
    print(f"[krok 1] dane: {s1['observed']}; odtworzenie rundy 3: {reproduced}")
    for k in ("reciprocal_pairs", "unidirectional_edges"):
        v = s1[k]
        print(f"  {k}: obs {v['observed']:.0f} null3 {v['null_mean']:.1f} ± {v['null_sd']:.1f} percentyl {v['percentile']:.2f}")
    print(f"  spearman n_real~cos: {s1['spearman_nreal_vs_cos']}")
    print(f"  spearman n_real~pary wzajemne: {s1['spearman_nreal_vs_reciprocal']}")
    print(f"[krok 2] asercje {len(new)}/{a.nreal}; przyjete zamiany rec {out['step2']['accepted_swaps_rec']['mean']:.0f}"
          f"/{out['step2']['attempts_rec']}, uni {out['step2']['accepted_swaps_uni']['mean']:.0f}/{out['step2']['attempts_uni']}")
    for v in ("weighted", "binary"):
        for k in KEYS:
            x = step3[v][k]
            print(f"[krok 3] {v} {k}: obs {x['observed']:.4f} null {x['null_mean']:.4f} ± {x['null_sd']:.4f} "
                  f"z {x['z']:+.2f} percentyl {x['percentile']:.2f}")
    print(f"gotowe w {out['seconds']:.0f} s")


if __name__ == "__main__":
    main()
