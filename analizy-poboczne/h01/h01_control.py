#!/usr/bin/env python3
"""H01: kontrola, czy roznica miedzy grafem "AXON" a pelnym wynika z samej liczby krawedzi.

  python3 h01_control.py --pairs-all parts2/pairs_all.csv --n-neurons 15487 --reps 200 --out parts2/control.json

Porownuje: (a) graf z parami zawierajacymi >=1 rekord AXON, (b) graf uzupelniajacy (pary bez zadnego rekordu AXON),
(c) graf pelny, (d) rozklad statystyk dla losowych podzbiorow pelnego grafu o tej samej liczbie par co (a).
Wymaga h01_components.py w tym samym katalogu. Tylko biblioteka standardowa.
"""
import argparse
import csv
import json
import random

import h01_components as c


def read_all(path):
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        v = [int(r[k]) for k in ("ax_t1", "ax_t2", "ax_other", "dend", "soma", "unknown", "other_class")]
        rows.append((int(r["pre"]), int(r["post"]), v))
    return rows


def graph_stats(pairs, n_total):
    """pairs: iterowalne (a, b). Zwraca statystyki jak w h01_components.summarize (bez wag)."""
    pairs = [(a, b) for a, b in pairs if a != b]
    ps = set(pairs)
    nodes = set()
    outs, ins = set(), set()
    adj = {}
    for a, b in ps:
        nodes.add(a)
        nodes.add(b)
        outs.add(a)
        ins.add(b)
        adj.setdefault(a, []).append(b)
    scc = c.scc_sizes(sorted(nodes), adj)
    wcc = c.wcc_sizes(sorted(nodes), [(a, b, 1) for a, b in ps])
    mutual = sum(1 for (a, b) in ps if (b, a) in ps) // 2
    return dict(pairs=len(ps), largest_scc=scc[0] if scc else 0, largest_wcc=wcc[0] if wcc else 0,
                mutual_pairs=mutual, share_with_out=len(outs) / n_total, share_with_in=len(ins) / n_total)


def recip_given_multi(rows, keep):
    """Wsrod par o >=2 rekordach (sposrod 'keep') udzial takich, ktorych para odwrotna istnieje w tym samym zbiorze."""
    ps = {(a, b) for a, b, v in rows if keep(v) and a != b}
    multi = [(a, b) for a, b, v in rows if keep(v) and a != b and sum(v) >= 2]
    if not multi:
        return None
    return dict(pairs_multi=len(multi), share_reverse_exists=sum(1 for (a, b) in multi if (b, a) in ps) / len(multi))


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-all", default="parts2/pairs_all.csv")
    ap.add_argument("--n-neurons", type=int, default=15487)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--out", default="parts2/control.json")
    a = ap.parse_args()
    rows = read_all(a.pairs_all)
    is_ax = lambda v: v[0] + v[1] + v[2] > 0
    all_pairs = [(x, y) for x, y, _ in rows]
    ax_pairs = [(x, y) for x, y, v in rows if is_ax(v)]
    co_pairs = [(x, y) for x, y, v in rows if not is_ax(v)]
    res = dict(n_neurons=a.n_neurons,
               full=graph_stats(all_pairs, a.n_neurons), axon=graph_stats(ax_pairs, a.n_neurons),
               complement=graph_stats(co_pairs, a.n_neurons),
               recip_multi_axon=recip_given_multi(rows, is_ax),
               recip_multi_complement=recip_given_multi(rows, lambda v: not is_ax(v)))
    rnd = random.Random(a.seed)
    k = len(ax_pairs)
    keys = ("largest_scc", "largest_wcc", "mutual_pairs", "share_with_out", "share_with_in")
    dist = {kk: [] for kk in keys}
    for _ in range(a.reps):
        st = graph_stats(rnd.sample(all_pairs, k), a.n_neurons)
        for kk in keys:
            dist[kk].append(st[kk])
    res["random_subsets"] = dict(size=k, reps=a.reps,
                                 **{kk: dict(mean=sum(v) / len(v), p2_5=pct(v, 0.025), median=pct(v, 0.5), p97_5=pct(v, 0.975))
                                    for kk, v in dist.items()})
    json.dump(res, open(a.out, "w"), indent=1)
    print("%-22s %8s %9s %9s %9s %9s %9s" % ("graf", "pary", "max_SCC", "max_WCC", "wzajemne", "z_out%", "z_in%"))
    for name in ("full", "axon", "complement"):
        s = res[name]
        print("%-22s %8d %9d %9d %9d %8.1f%% %8.1f%%" % (name, s["pairs"], s["largest_scc"], s["largest_wcc"],
                                                        s["mutual_pairs"], 100 * s["share_with_out"], 100 * s["share_with_in"]))
    r = res["random_subsets"]
    print("\nLosowe podzbiory pelnego grafu, po %d par (%d powtorzen), srednia [2,5%%; 97,5%%]:" % (r["size"], r["reps"]))
    for kk in keys:
        d = r[kk]
        f = (lambda x: "%.1f%%" % (100 * x)) if kk.startswith("share") else (lambda x: "%d" % x)
        print("  %-16s %s  [%s; %s]" % (kk, f(d["mean"]), f(d["p2_5"]), f(d["p97_5"])))
    print("\nOdwzajemnienie wsrod par o >=2 rekordach: AXON %s | uzupelniajacy %s" % (
        res["recip_multi_axon"], res["recip_multi_complement"]))
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
