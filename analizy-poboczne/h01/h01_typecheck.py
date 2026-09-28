#!/usr/bin/env python3
"""H01: rozklad laczny (type rekordu, type miejsca pre, type miejsca post, class_label pre, class_label post)
na kilku rownomiernie rozlozonych plikach JSON synaps. Tylko biblioteka standardowa; strumieniowo, nic nie zapisuje poza
--out. Cel: sprawdzic, czy "type" == 2 wsrod par wewnatrz zbioru neuronow oznacza cos w rodzaju odwroconego kierunku,
oraz czy miejsca presynaptyczne wewnatrz zbioru maja czesciej klase DENDRITE niz w calej populacji.

  python3 h01_typecheck.py --somas somas.csv --nfiles 4 --out typecheck.json

Wymaga h01_pairs.py w tym samym katalogu (uzywa load_somas, list_files, obj_url).
Wniosek wolno wyciagac tylko z liczb tego wydruku; probka to kilka plikow, nie caly zbior.
"""
import argparse
import json
import time
import urllib.request
from collections import Counter

import h01_pairs as h


def _int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return None


def tally(lines, nset):
    allc, pre_in, both_in = Counter(), Counter(), Counter()
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        pre = r.get("pre_synaptic_site")
        post = r.get("post_synaptic_partner")
        if not isinstance(pre, dict) or post is None:
            continue
        for po in (post if isinstance(post, list) else [post]):
            if not isinstance(po, dict):
                continue
            n += 1
            key = (str(r.get("type")), str(pre.get("type")), str(po.get("type")),
                   str(pre.get("class_label")), str(po.get("class_label")))
            allc[key] += 1
            p, q = _int(pre.get("neuron_id")), _int(po.get("neuron_id"))
            if p in nset:
                pre_in[key] += 1
                if q in nset:
                    both_in[key] += 1
    return dict(records=n, all=allc, pre_in=pre_in, both_in=both_in)


def stream_file(name, nset):
    req = urllib.request.Request(h.obj_url(name), headers={"User-Agent": "h01-typecheck/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return tally(resp, nset)


def merge(parts):
    tot = dict(records=0, all=Counter(), pre_in=Counter(), both_in=Counter())
    for p in parts:
        tot["records"] += p["records"]
        for k in ("all", "pre_in", "both_in"):
            tot[k].update(p[k])
    return tot


def show(title, c, top=8):
    s = sum(c.values())
    print("\n%s (n=%d) [type_rekordu, type_pre, type_post, klasa_pre, klasa_post]:" % (title, s))
    for k, v in c.most_common(top):
        print("  %-52s %9d  %5.1f%%" % (str(list(k)), v, 100.0 * v / max(1, s)))
    rec2 = sum(v for k, v in c.items() if k[0] == "2")
    predend = sum(v for k, v in c.items() if k[3] == "DENDRITE")
    print("  udzial type_rekordu=2: %.1f%% | udzial pre=DENDRITE: %.1f%%" % (
        100.0 * rec2 / max(1, s), 100.0 * predend / max(1, s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--somas", default="somas.csv")
    ap.add_argument("--nfiles", type=int, default=4)
    ap.add_argument("--out", default="typecheck.json")
    a = ap.parse_args()
    nset = set(h.load_somas(a.somas))
    files = [n for n, _ in h.list_files()]
    step = max(1, len(files) // a.nfiles)
    chosen = files[::step][:a.nfiles]
    print("plikow ogolem %d, wybrano %d: %s" % (len(files), len(chosen), [c.split("/")[-1] for c in chosen]))
    parts = []
    for c in chosen:
        t0 = time.time()
        parts.append(stream_file(c, nset))
        print("  %s: %d rekordow, %.0fs" % (c.split("/")[-1], parts[-1]["records"], time.time() - t0), flush=True)
    tot = merge(parts)
    show("WSZYSTKIE rekordy", tot["all"])
    show("Miejsce presynaptyczne w zbiorze neuronow", tot["pre_in"])
    show("Obie strony w zbiorze neuronow (pary)", tot["both_in"])
    json.dump({k: ({" | ".join(kk): v for kk, v in c.items()} if isinstance(c, Counter) else c) for k, c in tot.items()},
              open(a.out, "w"), indent=1)
    print("\nzapisano", a.out)


if __name__ == "__main__":
    main()
