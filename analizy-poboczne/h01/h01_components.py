#!/usr/bin/env python3
"""H01: skladowe grafu par (pairs.csv) oraz podzbiory neuronow oddalone od granicy wolumenu.

Tylko biblioteka standardowa. Nic nie pobiera. Uruchom w katalogu z parts/:
  python3 h01_components.py --pairs parts/pairs.csv --neurons parts/neurons.csv \
      --size 515892 356400 5293 --out parts/components.json

Uwaga metodologiczna: bbox = obwiednia synaps neuronu, a nie jego neurytow. Neuron, ktorego synapsy nie dotykaja
krawedzi, moze byc mimo to obciety, wiec liczby oznaczaja GORNA granice liczby neuronow kompletnych.
Jednostki bbox zakladamy takie same jak rozmiar wolumenu (--size); to nie jest potwierdzone.
"""
import argparse
import csv
import json
from collections import defaultdict


def read_pairs(path):
    edges = []
    for r in csv.DictReader(open(path, newline="")):
        edges.append((int(r["pre"]), int(r["post"]), int(r["n_type1"]) + int(r["n_type2"]) + int(r["n_other"])))
    return edges


def wcc_sizes(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    cnt = defaultdict(int)
    for n in nodes:
        cnt[find(n)] += 1
    return sorted(cnt.values(), reverse=True)


def scc_sizes(nodes, adj):
    """Iteracyjny Tarjan."""
    index, low, onst, st, sizes = {}, {}, set(), [], []
    idx = 0
    for s in nodes:
        if s in index:
            continue
        index[s] = low[s] = idx
        idx += 1
        st.append(s)
        onst.add(s)
        work = [(s, iter(adj.get(s, ())))]
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = idx
                    idx += 1
                    st.append(w)
                    onst.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                elif w in onst:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = 0
                while True:
                    w = st.pop()
                    onst.discard(w)
                    comp += 1
                    if w == v:
                        break
                sizes.append(comp)
    return sorted(sizes, reverse=True)


def summarize(nodes, edges):
    nodes = set(nodes)
    E = [(a, b, w) for a, b, w in edges if a in nodes and b in nodes]
    selfp = sum(1 for a, b, _ in E if a == b)
    E2 = [(a, b, w) for a, b, w in E if a != b]
    adj = defaultdict(list)
    outn, inn = set(), set()
    pairset = set()
    for a, b, _ in E2:
        adj[a].append(b)
        outn.add(a)
        inn.add(b)
        pairset.add((a, b))
    recip = sum(1 for (a, b) in pairset if (b, a) in pairset) // 2
    scc = scc_sizes(sorted(nodes), adj)
    wcc = wcc_sizes(sorted(nodes), E2)
    return dict(neurons=len(nodes), pairs=len(E2), synapses=sum(w for _, _, w in E2), self_pairs=selfp,
                with_out=len(outn), with_in=len(inn), reciprocal_pairs=recip,
                largest_wcc=wcc[0] if wcc else 0, largest_scc=scc[0] if scc else 0,
                scc_top5=[s for s in scc[:5]], scc_gt1=sum(1 for s in scc if s > 1))


def read_bbox(path):
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        try:
            b = [float(r[k]) for k in ("bx0", "by0", "bz0", "bx1", "by1", "bz1")]
        except ValueError:
            continue
        out[int(r["c3_seg"])] = b
    return out


def extents(bbox):
    lo = [min(b[i] for b in bbox.values()) for i in range(3)]
    hi = [max(b[3 + i] for b in bbox.values()) for i in range(3)]
    return lo, hi


def interior(bbox, lo, hi, mxy, mz):
    keep = set()
    for n, b in bbox.items():
        if (b[0] >= lo[0] + mxy and b[1] >= lo[1] + mxy and b[2] >= lo[2] + mz and
                b[3] <= hi[0] - mxy and b[4] <= hi[1] - mxy and b[5] <= hi[2] - mz):
            keep.add(n)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="parts/pairs.csv")
    ap.add_argument("--neurons", default="parts/neurons.csv")
    ap.add_argument("--size", type=float, nargs=3, default=[515892, 356400, 5293])
    ap.add_argument("--empirical", action="store_true",
                    help="granice = empiryczne zasiegi obwiedni synaps (min/max po wszystkich neuronach) zamiast --size")
    ap.add_argument("--margins-xy", type=float, nargs="+", default=[0, 2000, 10000, 30000, 60000])
    ap.add_argument("--margins-z", type=float, nargs="+", default=[0, 100, 500])
    ap.add_argument("--out", default="parts/components.json")
    a = ap.parse_args()
    edges = read_pairs(a.pairs)
    bbox = read_bbox(a.neurons)
    allnodes = set(bbox)
    for r in csv.DictReader(open(a.neurons, newline="")):  # takze neurony bez obwiedni (bez synaps w eksporcie)
        allnodes.add(int(r["c3_seg"]))
    for x, y, _ in edges:
        allnodes.add(x)
        allnodes.add(y)
    res = {"all": summarize(allnodes, edges), "interior": []}
    print("CALY GRAF:", json.dumps(res["all"]))
    lo, hi = extents(bbox) if a.empirical else ([0.0, 0.0, 0.0], list(a.size))
    print("\nGranice (%s): lo=%s hi=%s" % ("empiryczne" if a.empirical else "nominalne", lo, hi))
    res["bounds"] = dict(kind="empirical" if a.empirical else "nominal", lo=lo, hi=hi)
    print("Podzbiory oddalone od granicy (bbox synaps; margines w jednostkach siatki):")
    print("%9s %9s | %8s %8s %9s %8s %9s %9s" % ("marg_xy", "marg_z", "neurony", "pary", "synapsy", "z_out", "max_SCC", "max_WCC"))
    for mxy in a.margins_xy:
        for mz in a.margins_z:
            keep = interior(bbox, lo, hi, mxy, mz)
            s = summarize(keep, edges)
            s.update(margin_xy=mxy, margin_z=mz)
            res["interior"].append(s)
            print("%9.0f %9.0f | %8d %8d %9d %8d %9d %9d" % (mxy, mz, s["neurons"], s["pairs"], s["synapses"],
                                                            s["with_out"], s["largest_scc"], s["largest_wcc"]))
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nzapisano", a.out)


if __name__ == "__main__":
    main()
