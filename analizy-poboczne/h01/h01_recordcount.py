#!/usr/bin/env python3
"""H01: ile rekordow przypada na jedno miejsce presynaptyczne? (Czy 166 mln rekordow moze odpowiadac ~130 mln synaps?)

  python3 h01_recordcount.py --index 0            # strumieniuje plik nr 0 z bucketu (potrzebuje h01_pairs2.py obok)
  python3 h01_recordcount.py --file lokalny.json  # albo lokalny plik (jeden rekord JSON na linie)

Liczy: rekordy, rozne id miejsc presynaptycznych, rozne id miejsc postsynaptycznych, rozne pary (pre id, post id),
rozklad liczby rekordow na jedno miejsce presynaptyczne i na jedno postsynaptyczne. Tylko biblioteka standardowa.
Wniosek dotyczy jednego pliku (okolo 1/166 zbioru), nie calosci.
"""
import argparse
import json
import urllib.request
from collections import Counter


def count(lines):
    pre, post, pair = Counter(), Counter(), set()
    n = bad = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        p = r.get("pre_synaptic_site")
        q = r.get("post_synaptic_partner")
        if not isinstance(p, dict) or q is None:
            continue
        for po in (q if isinstance(q, list) else [q]):
            if not isinstance(po, dict):
                continue
            n += 1
            a, b = p.get("id"), po.get("id")
            pre[a] += 1
            post[b] += 1
            pair.add((a, b))
    return dict(records=n, bad=bad, distinct_pre=len(pre), distinct_post=len(post), distinct_pairs=len(pair),
                pre_mult=Counter(min(v, 5) for v in pre.values()), post_mult=Counter(min(v, 5) for v in post.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    if a.file:
        res = count(open(a.file, "rb"))
        src = a.file
    elif a.index is not None:
        import h01_pairs2 as h
        name = h.list_files()[a.index][0]
        with urllib.request.urlopen(urllib.request.Request(h.obj_url(name), headers={"User-Agent": "h01-rc/1.0"}), timeout=120) as resp:
            res = count(resp)
        src = name
    else:
        ap.error("podaj --index albo --file")
    print("zrodlo:", src)
    print("rekordow: %d (bledne linie: %d)" % (res["records"], res["bad"]))
    print("rozne id miejsc presynaptycznych : %d  (%.1f%% liczby rekordow)" % (res["distinct_pre"], 100 * res["distinct_pre"] / max(1, res["records"])))
    print("rozne id miejsc postsynaptycznych: %d  (%.1f%%)" % (res["distinct_post"], 100 * res["distinct_post"] / max(1, res["records"])))
    print("rozne pary (pre id, post id)     : %d  (%.1f%%)" % (res["distinct_pairs"], 100 * res["distinct_pairs"] / max(1, res["records"])))
    print("rekordow na jedno miejsce presynaptyczne (1,2,3,4,5+):", [res["pre_mult"].get(k, 0) for k in range(1, 6)])
    print("rekordow na jedno miejsce postsynaptyczne (1,2,3,4,5+):", [res["post_mult"].get(k, 0) for k in range(1, 6)])


if __name__ == "__main__":
    main()
