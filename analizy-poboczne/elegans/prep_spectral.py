"""Zerowanie przekatnej, skladowe spojne i stopnie dla macierzy 272 (bez widm).

Uruchom: python prep_spectral.py  (w katalogu dane-elegans/, po build_matrices.py; potrzebne numpy, scipy)
"""
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

L = json.load(open("labels_272.json"))


def zero_sets(W):
    return ({L[i] for i in np.where(~W.any(axis=1))[0]}, {L[i] for i in np.where(~W.any(axis=0))[0]})


def components(A, connection):
    n, lab = connected_components(csr_matrix(A), directed=True, connection=connection)
    sizes = np.bincount(lab)
    return n, sorted(sizes.tolist(), reverse=True), lab


def compact_sizes(sizes):
    """[250, 1, 1, 1] -> '250, 1 x3'"""
    out, i = [], 0
    while i < len(sizes):
        j = i
        while j < len(sizes) and sizes[j] == sizes[i]:
            j += 1
        out.append(f"{sizes[i]}" + (f" x{j - i}" if j - i > 1 else ""))
        i = j
    return ", ".join(out)


def degree_stats(name, deg, strength):
    top = np.argsort(-deg, kind="stable")[:5]
    s = dict(mean=float(deg.mean()), median=float(np.median(deg)), max=int(deg.max()),
             top5=[dict(label=L[i], degree=int(deg[i]), strength=float(strength[i])) for i in top])
    hubs = ", ".join(f"{h['label']} {h['degree']}" for h in s["top5"])
    print(f"  {name}: srednia {s['mean']:.2f}, mediana {s['median']:.1f}, max {s['max']}; top5: {hubs}")
    return s


def main():
    out = {}
    raw = {"chem": np.load("herm_chem_272.npy"), "gap": np.load("herm_gap_sym_272.npy")}
    W = {}
    for key, fn in (("chem", "herm_chem_272_noauto.npy"), ("gap", "herm_gap_sym_272_noauto.npy")):
        M = raw[key].copy()
        diag_nz = int(np.count_nonzero(np.diag(M)))
        np.fill_diagonal(M, 0)
        np.save(fn, M)
        W[key] = M
        zr0, zc0 = zero_sets(raw[key])
        zr, zc = zero_sets(M)
        out[key] = dict(file=fn, nonzero_before=int(np.count_nonzero(raw[key])), diag_nonzero_removed=diag_nz,
                        nonzero_after=int(np.count_nonzero(M)), zero_rows=sorted(zr), zero_cols=sorted(zc),
                        new_zero_rows=sorted(zr - zr0), new_zero_cols=sorted(zc - zc0))
        o = out[key]
        print(f"[{key}] niezerowe {o['nonzero_before']} - {diag_nz} (przekatna) = {o['nonzero_after']}")
        print(f"  zerowe wiersze {o['zero_rows']} (nowe: {o['new_zero_rows']}); "
              f"zerowe kolumny {o['zero_cols']} (nowe: {o['new_zero_cols']})")

    # zlacza: graf nieskierowany, najwieksza skladowa
    G = W["gap"]
    assert np.array_equal(G, G.T)
    n, sizes, lab = components(G != 0, "weak")
    big = np.argmax(np.bincount(lab))
    idx = np.where(lab == big)[0]
    core = G[np.ix_(idx, idx)]
    core_labels = [L[i] for i in idx]
    np.save("herm_gap_sym_core.npy", core)
    json.dump(core_labels, open("herm_gap_sym_core_labels.json", "w"))
    assert core.any(axis=1).all(), "izolowany neuron w rdzeniu"
    outside = [L[i] for i in range(len(L)) if lab[i] != big]
    out["gap"].update(n_components=n, component_sizes=sizes, core_shape=list(core.shape),
                      core_nonzero=int(np.count_nonzero(core)), outside_core=outside)
    print(f"[gap] skladowe: {n}; rozmiary: {compact_sizes(sizes)}; rdzen {core.shape}, poza rdzeniem: {outside}")

    # chemiczne: slabo i silnie spojne
    A = W["chem"] != 0
    nw, sw, _ = components(A, "weak")
    ns, ss, _ = components(A, "strong")
    out["chem"].update(weak_n=nw, weak_sizes=sw, strong_n=ns, strong_sizes=ss)
    print(f"[chem] slabo spojne: {nw}; rozmiary: {compact_sizes(sw)}")
    print(f"[chem] silnie spojne: {ns}; rozmiary: {compact_sizes(ss)}")

    # stopnie (liczba roznych partnerow, bez petli); sila = suma wag
    print("[stopnie]")
    C = W["chem"]
    out["degrees"] = dict(
        chem_out=degree_stats("chem out", (C != 0).sum(axis=1), C.sum(axis=1)),
        chem_in=degree_stats("chem in", (C != 0).sum(axis=0), C.sum(axis=0)),
        chem_total_undirected=degree_stats("chem total (nosnik zsymetryzowany)", ((C != 0) | (C.T != 0)).sum(axis=1),
                                           C.sum(axis=0) + C.sum(axis=1)),
        gap=degree_stats("gap", (G != 0).sum(axis=1), G.sum(axis=1)))
    json.dump(out, open("prep_spectral_summary.json", "w"), indent=1)
    print("zapisano: *_noauto.npy, herm_gap_sym_core.npy, herm_gap_sym_core_labels.json, prep_spectral_summary.json")


if __name__ == "__main__":
    main()
