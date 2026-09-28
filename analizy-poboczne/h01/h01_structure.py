#!/usr/bin/env python3
"""H01: opis struktury grafu neuron->neuron (pairs.csv + neurons.csv). Tylko biblioteka standardowa, nic nie pobiera.

  python3 h01_structure.py --pairs parts/pairs.csv --neurons parts/neurons.csv --out parts/structure.json

Uwagi (zapisane tez w wyniku):
 * W H01 wiekszosc aksonow jest obcieta lub odlaczona od somy, wiec brak krawedzi nie oznacza braku polaczenia.
   Kompletnosc aksonow rozni sie miedzy typami komórek (patrz udzial neuronow z krawedzia wychodzaca).
 * Kolumny n_type1/n_type2 to kod pola "type"; jego znaczenie (E/I?) NIE jest potwierdzone.
 * Odleglosc = odleglosc w plaszczyznie xy miedzy somami, w jednostkach siatki (jednostki fizyczne niepotwierdzone).
   Osi z nie uzywamy (inna skala woksela). Dusze z ktoras wspolrzedna = 0 (prawdopodobny brak danych) pomijamy.
 * p(d): prawdopodobienstwo polaczenia skierowanego w przedzialach odleglosci o rownej liczbie par (kwantyle z losowej
   probki 1 mln par); mianownik = N(N-1)/liczba_przedzialow. Przyblizenie.
"""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    k = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[k]


def read_neurons(path):
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        d = dict(celltype=r["celltype"], layer=r["layer"])
        try:
            d["xyz"] = (float(r["soma_x"]), float(r["soma_y"]), float(r["soma_z"]))
        except ValueError:
            d["xyz"] = None
        try:
            d["bbox"] = [float(r[k]) for k in ("bx0", "by0", "bz0", "bx1", "by1", "bz1")]
        except ValueError:
            d["bbox"] = None
        out[int(r["c3_seg"])] = d
    return out


def read_pairs(path):
    return [(int(r["pre"]), int(r["post"]), int(r["n_type1"]), int(r["n_type2"]), int(r["n_other"]))
            for r in csv.DictReader(open(path, newline=""))]


def degree_block(nodes, pairs):
    outd, ind = Counter(), Counter()
    for a, b, *_ in pairs:
        outd[a] += 1
        ind[b] += 1
    ov = [outd.get(n, 0) for n in nodes]
    iv = [ind.get(n, 0) for n in nodes]

    def summ(v):
        nz = [x for x in v if x > 0]
        return dict(mean=sum(v) / len(v), nonzero_share=len(nz) / len(v), median_nonzero=pct(nz, 0.5),
                    p90_nonzero=pct(nz, 0.9), p99_nonzero=pct(nz, 0.99), max=max(v))
    wts = Counter(min(a[2] + a[3] + a[4], 3) for a in pairs)
    return dict(out=summ(ov), inn=summ(iv), pair_weight_1_2_3plus=[wts[1], wts[2], wts[3]]), outd, ind


def distance_block(neurons, pairs, nbins=10, nsamp=1_000_000, seed=1, logbins=0):
    ids = [n for n, d in neurons.items() if d["xyz"] and all(c != 0 for c in d["xyz"])]
    idset = set(ids)
    if len(ids) < 50:
        return None
    rnd = random.Random(seed)
    xy = {n: neurons[n]["xyz"][:2] for n in ids}
    samp = []
    for _ in range(nsamp):
        a, b = rnd.sample(ids, 2)
        samp.append(math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]))
    samp.sort()
    if logbins:
        # przedzialy logarytmiczne od 1-go promila probki do maksimum; mianownik z rzeczywistego udzialu probki w przedziale
        import bisect
        e0 = max(samp[int(len(samp) * 1e-4)], 1.0)
        edges = [e0 * (samp[-1] / e0) ** (k / logbins) for k in range(0, logbins)]
        lo_b = [0.0] + edges
        hi_b = edges + [samp[-1] + 1.0]
        scnt = [0] * len(lo_b)
        for d in samp:
            scnt[bisect.bisect_right(edges, d)] += 1
        cnt = [0] * len(lo_b)
        used = 0
        for a, b, *_ in pairs:
            if a in idset and b in idset:
                d = math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
                cnt[bisect.bisect_right(edges, d)] += 1
                used += 1
        N = len(ids)
        bins = []
        for k in range(len(lo_b)):
            frac = scnt[k] / len(samp)
            den = N * (N - 1) * frac
            bins.append(dict(d_from=lo_b[k], d_to=hi_b[k], pairs=cnt[k], sample_points=scnt[k],
                             p=(cnt[k] / den) if den > 0 else None, unreliable=scnt[k] < 200))
        return dict(neurons_used=N, pairs_used=used, mode="log", bins=bins)
    edges = [samp[int(len(samp) * k / nbins)] for k in range(1, nbins)]
    cnt = [0] * nbins
    used = 0
    for a, b, *_ in pairs:
        if a in idset and b in idset:
            d = math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
            k = sum(1 for e in edges if d >= e)
            cnt[k] += 1
            used += 1
    N = len(ids)
    denom = N * (N - 1) / nbins
    lo = [0.0] + edges
    hi = edges + [samp[-1]]
    return dict(neurons_used=N, pairs_used=used,
                bins=[dict(d_from=lo[k], d_to=hi[k], pairs=cnt[k], p=cnt[k] / denom) for k in range(nbins)])


def analyze(neurons, pairs, nbins, logbins=0):
    nodes = sorted(neurons)
    res = {}
    res["n_neurons"], res["n_pairs"], res["n_synapses"] = len(nodes), len(pairs), sum(p[2] + p[3] + p[4] for p in pairs)
    deg, outd, ind = degree_block(nodes, pairs)
    res["degrees"] = deg
    pairset = {(a, b) for a, b, *_ in pairs}
    recip_dir = sum(1 for (a, b) in pairset if (b, a) in pairset)
    res["reciprocity"] = dict(directed_edges_reciprocated=recip_dir, share_of_edges=recip_dir / max(1, len(pairset)),
                              mutual_pairs=recip_dir // 2)
    ct = Counter(neurons[n]["celltype"] for n in nodes)
    by = {}
    for c in sorted(ct):
        ns = [n for n in nodes if neurons[n]["celltype"] == c]
        pre_pairs = [p for p in pairs if neurons[p[0]]["celltype"] == c]
        rec = sum(1 for p in pre_pairs if (p[1], p[0]) in pairset)
        by[c] = dict(n=len(ns), share_out_gt0=sum(1 for n in ns if outd.get(n, 0) > 0) / len(ns),
                     share_in_gt0=sum(1 for n in ns if ind.get(n, 0) > 0) / len(ns),
                     mean_out=sum(outd.get(n, 0) for n in ns) / len(ns), mean_in=sum(ind.get(n, 0) for n in ns) / len(ns),
                     outgoing_pairs=len(pre_pairs), outgoing_reciprocated_share=rec / max(1, len(pre_pairs)),
                     type1_type2_other=[sum(p[2] for p in pre_pairs), sum(p[3] for p in pre_pairs), sum(p[4] for p in pre_pairs)])
    res["by_celltype"] = by
    lay = {}
    for l in sorted({neurons[n]["layer"] for n in nodes}):
        ns = [n for n in nodes if neurons[n]["layer"] == l]
        lay[l] = dict(n=len(ns), share_out_gt0=sum(1 for n in ns if outd.get(n, 0) > 0) / len(ns))
    res["by_layer"] = lay
    mat = defaultdict(Counter)
    for a, b, *_ in pairs:
        mat[neurons[a]["celltype"]][neurons[b]["celltype"]] += 1
    res["pre_post_celltype_pairs"] = {a: dict(mat[a]) for a in sorted(mat)}
    res["distance_dependence"] = distance_block(neurons, pairs, nbins, nsamp=2_000_000 if logbins else 1_000_000,
                                                logbins=logbins)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="parts/pairs.csv")
    ap.add_argument("--neurons", default="parts/neurons.csv")
    ap.add_argument("--out", default="parts/structure.json")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--logbins", type=int, default=0,
                    help="jesli >0: p(d) w tylu przedzialach logarytmicznych zamiast kwantylowych (lepsza rozdzielczosc blisko)")
    ap.add_argument("--interior-mz", type=float, default=100.0,
                    help="margines z (jedn. siatki) dla podzbioru neuronow oddalonych od granicy; 0 = pomin")
    a = ap.parse_args()
    neurons = read_neurons(a.neurons)
    pairs = read_pairs(a.pairs)
    out = dict(note="patrz docstring: obciecie aksonow, type niepotwierdzone, odleglosc xy w jednostkach siatki")
    out["all"] = analyze(neurons, pairs, a.nbins, a.logbins)
    bb = [d["bbox"] for d in neurons.values() if d["bbox"]]
    if a.interior_mz > 0 and bb:
        lo = [min(b[i] for b in bb) for i in range(3)]
        hi = [max(b[3 + i] for b in bb) for i in range(3)]
        keep = {n for n, d in neurons.items() if d["bbox"] and d["bbox"][2] >= lo[2] + a.interior_mz
                and d["bbox"][5] <= hi[2] - a.interior_mz}
        sub = {n: neurons[n] for n in keep}
        sp = [p for p in pairs if p[0] in keep and p[1] in keep]
        out["interior_z"] = dict(margin_z=a.interior_mz, **analyze(sub, sp, a.nbins, a.logbins)) if sub else None
    json.dump(out, open(a.out, "w"), indent=1)

    def show(tag, r):
        print("\n=== %s: %d neuronow, %d par, %d synaps ===" % (tag, r["n_neurons"], r["n_pairs"], r["n_synapses"]))
        for k in ("out", "inn"):
            d = r["degrees"][k]
            print(" stopien %s: srednia %.2f | z krawedzia %.1f%% | mediana(>0) %s p90 %s p99 %s max %s" % (
                "wyjsciowy" if k == "out" else "wejsciowy", d["mean"], 100 * d["nonzero_share"], d["median_nonzero"],
                d["p90_nonzero"], d["p99_nonzero"], d["max"]))
        w = r["degrees"]["pair_weight_1_2_3plus"]
        print(" pary o wadze (liczba synaps) 1 / 2 / 3+: %d / %d / %d" % tuple(w))
        rc = r["reciprocity"]
        print(" odwzajemnienie: %.2f%% krawedzi (%d par wzajemnych)" % (100 * rc["share_of_edges"], rc["mutual_pairs"]))
        print(" typ komorki: n | z out>0 | z in>0 | sr.out | sr.in | odwzajemnione(wyjsciowe)")
        for c, d in r["by_celltype"].items():
            print("  %-22s %6d | %5.1f%% | %5.1f%% | %5.2f | %5.2f | %5.1f%%" % (
                c, d["n"], 100 * d["share_out_gt0"], 100 * d["share_in_gt0"], d["mean_out"], d["mean_in"],
                100 * d["outgoing_reciprocated_share"]))
        dd = r["distance_dependence"]
        if dd:
            print(" p(polaczenia) wg odleglosci xy (przedzialy %s; uzyto %d neuronow, %d par):" % (
                "logarytmiczne" if dd.get("mode") == "log" else "o rownej liczbie par", dd["neurons_used"], dd["pairs_used"]))
            for b in dd["bins"]:
                flag = "  (za malo punktow probki: p niepewne)" if b.get("unreliable") else ""
                pv = "brak" if b["p"] is None else "%.2e" % b["p"]
                print("   %9.0f-%9.0f  par=%6d  p=%s%s" % (b["d_from"], b["d_to"], b["pairs"], pv, flag))

    show("CALY GRAF", out["all"])
    if out.get("interior_z"):
        show("PODZBIOR oddalony od granicy w z (margines %g)" % a.interior_mz, out["interior_z"])
    print("\nzapisano", a.out)


if __name__ == "__main__":
    main()
