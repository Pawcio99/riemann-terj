#!/usr/bin/env python3
"""H01 (c3): graf par neuron->neuron i bounding box synaps na neuron, przez strumieniowanie surowych JSON-ów.

Tylko biblioteka standardowa. Wznawialne: gotowe pliki part_*.json.gz sa pomijane.
Nic nie jest zapisywane poza katalogiem --out. Wielkie pliki nie ladują sie na dysk (strumien -> parser).

Tryby:
  python h01_pairs.py --selftest --somas somas.csv --head json_head.bin
  python h01_pairs.py --list
  python h01_pairs.py --run   --somas somas.csv --out parts --workers 4 [--limit 2]
  python h01_pairs.py --merge --somas somas.csv --out parts --expect 166

Zalozenie o danych (do potwierdzenia przez --selftest): rekord = jedna linia JSON z polami
pre_synaptic_site.neuron_id, post_synaptic_partner.neuron_id (dict albo lista), location (dict x/y/z albo lista),
type (1/2; znaczenie E/I NIE jest potwierdzone). Zbior neuronow: c3_rep_manual z somas.csv dla typow neuronalnych
w segmentach c3 z dokladnie jedna dusza.
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from multiprocessing import Pool

BUCKET = "h01-release"
PREFIX = "data/20210601/c3/synapses/exported/json/"
NEURON_TYPES = {"PYRAMIDAL", "INTERNEURON", "SPINY_ATYPICAL", "UNCLASSIFIED_NEURON", "SPINY_STELLATE", "C_SHAPED"}
EXPECTED_SINGLE_SOMA_NEURONS = 15487  # z rozpoznania; tylko do ostrzezenia


def _int(s):
    s = str(s).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def load_somas(path):
    rows = list(csv.DictReader(open(path, newline="")))
    per_seg = {}
    for r in rows:
        seg = _int(r["c3_rep_manual"])
        if seg:
            per_seg[seg] = per_seg.get(seg, 0) + 1
    neurons = {}
    for r in rows:
        seg = _int(r["c3_rep_manual"])
        if seg and r["celltype"] in NEURON_TYPES and per_seg[seg] == 1:
            neurons[seg] = r
    return neurons


def _i(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return None


def _c(x, y, z):
    x, y, z = _i(x), _i(y), _i(z)
    return None if None in (x, y, z) else (x, y, z)


def _xyz(o):
    if isinstance(o, dict):
        if all(k in o for k in ("x", "y", "z")):
            return _c(o["x"], o["y"], o["z"])
        return None
    if isinstance(o, (list, tuple)) and len(o) >= 3:
        return _c(o[0], o[1], o[2])
    return None


def _upd(bbox, nid, p):
    if p is None:
        return
    b = bbox.get(nid)
    if b is None:
        bbox[nid] = [p[0], p[1], p[2], p[0], p[1], p[2]]
    else:
        for i in range(3):
            if p[i] < b[i]:
                b[i] = p[i]
            if p[i] > b[3 + i]:
                b[3 + i] = p[i]


def process_lines(lines, nset):
    """lines: iterowalne obiekty bytes/str (jedna linia JSON). Zwraca (pairs, bbox, stats)."""
    pairs, bbox = {}, {}
    st = dict(records=0, bad=0, no_ids=0, pre_in=0, post_in=0, pair_in=0)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            st["bad"] += 1
            continue
        st["records"] += 1
        pre = rec.get("pre_synaptic_site")
        post = rec.get("post_synaptic_partner")
        if not isinstance(pre, dict) or post is None:
            st["no_ids"] += 1
            continue
        posts = post if isinstance(post, list) else [post]
        pid = _i(pre.get("neuron_id"))
        rloc = _xyz(rec.get("location"))
        pre_in = pid in nset
        if pre_in:
            st["pre_in"] += 1
            _upd(bbox, pid, _xyz(pre.get("centroid") or pre.get("location")) or rloc)
        t = _i(rec.get("type"))
        ti = 0 if t == 1 else (1 if t == 2 else 2)
        for po in posts:
            if not isinstance(po, dict):
                continue
            qid = _i(po.get("neuron_id"))
            q_in = qid in nset
            if q_in:
                st["post_in"] += 1
                _upd(bbox, qid, _xyz(po.get("centroid") or po.get("location")) or rloc)
            if pre_in and q_in:
                st["pair_in"] += 1
                k = (pid, qid)
                v = pairs.get(k)
                if v is None:
                    v = pairs[k] = [0, 0, 0]
                v[ti] += 1
    return pairs, bbox, st


def list_files():
    out, token = [], None
    while True:
        q = {"prefix": PREFIX, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        url = "https://storage.googleapis.com/storage/v1/b/%s/o?%s" % (BUCKET, urllib.parse.urlencode(q))
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for it in d.get("items", []):
            if not it["name"].endswith("/"):
                out.append((it["name"], int(it["size"])))
        token = d.get("nextPageToken")
        if not token:
            break
    return sorted(out)


def obj_url(name):
    return "https://storage.googleapis.com/%s/%s" % (BUCKET, urllib.parse.quote(name))


def _part_path(outdir, name):
    return os.path.join(outdir, "part_%s.json.gz" % os.path.basename(name))


def fetch_process(url, nset, tries=4):
    last = None
    for k in range(tries):
        try:
            nbytes = [0]

            def gen(resp):
                for ln in resp:
                    nbytes[0] += len(ln)
                    yield ln

            req = urllib.request.Request(url, headers={"User-Agent": "h01-pairs/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                pairs, bbox, st = process_lines(gen(resp), nset)
            st["bytes"] = nbytes[0]
            return pairs, bbox, st
        except Exception as e:  # siec, timeout, reset
            last = e
            time.sleep(5 * (k + 1))
    raise RuntimeError("nie udalo sie: %s (%s)" % (url, last))


def worker(args):
    name, url, outdir, nset = args
    path = _part_path(outdir, name)
    if os.path.exists(path):
        return name, 0.0, None
    t0 = time.time()
    pairs, bbox, st = fetch_process(url, nset)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(dict(file=name, stats=st, pairs=[[a, b] + v for (a, b), v in pairs.items()],
                       bbox={str(k): v for k, v in bbox.items()}), f)
    os.replace(tmp, path)
    return name, time.time() - t0, st


def cmd_selftest(a):
    nset = load_somas(a.somas)
    print("neurony jednodusze (zbior do filtrowania): %d (oczekiwano ok. %d)" % (len(nset), EXPECTED_SINGLE_SOMA_NEURONS))
    data = open(a.head, "rb").read()
    lines = data.split(b"\n")[:-1]  # ostatnia linia jest urwana
    print("kompletnych linii w naglowku: %d" % len(lines))
    rec = json.loads(lines[0])

    def tree(d, ind=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * ind + "%s: {" % k)
                tree(v, ind + 2)
                print(" " * ind + "}")
            else:
                print(" " * ind + "%s: %s" % (k, repr(v)[:60]))

    tree(rec)
    pre = rec.get("pre_synaptic_site", {})
    post = rec.get("post_synaptic_partner")
    print("pre neuron_id:", pre.get("neuron_id"), "| post typ:", type(post).__name__,
          "| location typ:", type(rec.get("location")).__name__, "| type:", rec.get("type"))
    pairs, bbox, st = process_lines(lines, nset)
    print("statystyki na naglowku:", st)
    print("neurony z bbox:", len(bbox), "| pary:", len(pairs))
    ok = st["records"] > 0 and st["no_ids"] == 0
    print("WYNIK: %s" % ("pola znalezione, mozna uruchomic --run" if ok else
                         "UWAGA: brakuje pol - nie uruchamiaj --run, wyslij mi ten wydruk"))


def cmd_list(a):
    fs = list_files()
    print("plikow: %d, lacznie %.1f GB, min %.0f MB, max %.0f MB" % (
        len(fs), sum(s for _, s in fs) / 1e9, min(s for _, s in fs) / 1e6, max(s for _, s in fs) / 1e6))


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    nset = load_somas(a.somas)
    fs = list_files()
    if a.limit:
        fs = fs[:a.limit]
    todo = [(n, obj_url(n), a.out, nset) for n, _ in fs if not os.path.exists(_part_path(a.out, n))]
    print("plikow: %d, do zrobienia: %d, workerow: %d, start %s" % (
        len(fs), len(todo), a.workers, time.strftime("%H:%M:%S")), flush=True)
    t0, done, tot_b = time.time(), 0, 0
    with Pool(a.workers) as pool:
        for name, dt, st in pool.imap_unordered(worker, todo):
            done += 1
            if st:
                tot_b += st["bytes"]
            el = time.time() - t0
            eta = el / done * (len(todo) - done)
            print("[%d/%d] %s %.0fs  rekordow=%s pary=%s | srednio %.1f MB/s | ETA %.0f min" % (
                done, len(todo), os.path.basename(name), dt, st and st["records"], st and st["pair_in"],
                tot_b / max(el, 1) / 1e6, eta / 60), flush=True)
    print("koniec %s" % time.strftime("%H:%M:%S"), flush=True)


def cmd_merge(a):
    neurons = load_somas(a.somas)
    files = sorted(f for f in os.listdir(a.out) if f.startswith("part_") and f.endswith(".json.gz"))
    print("plikow czesciowych: %d (oczekiwano %s)" % (len(files), a.expect))
    if a.expect and len(files) != a.expect:
        print("UWAGA: niekompletne - wynik bedzie obciazony, nie analizuj")
    pairs, bbox, tot = {}, {}, dict(records=0, bad=0, no_ids=0, pre_in=0, post_in=0, pair_in=0, bytes=0)
    for f in files:
        d = json.load(gzip.open(os.path.join(a.out, f), "rt"))
        for k in tot:
            tot[k] += d["stats"].get(k, 0)
        for p in d["pairs"]:
            v = pairs.setdefault((p[0], p[1]), [0, 0, 0])
            for i in range(3):
                v[i] += p[2 + i]
        for k, b in d["bbox"].items():
            k = int(k)
            c = bbox.get(k)
            if c is None:
                bbox[k] = list(b)
            else:
                for i in range(3):
                    c[i] = min(c[i], b[i])
                    c[3 + i] = max(c[3 + i], b[3 + i])
    with open(os.path.join(a.out, "pairs.csv"), "w") as f:
        f.write("pre,post,n_type1,n_type2,n_other\n")
        for (x, y), v in sorted(pairs.items()):
            f.write("%d,%d,%d,%d,%d\n" % (x, y, v[0], v[1], v[2]))
    with open(os.path.join(a.out, "neurons.csv"), "w") as f:
        f.write("c3_seg,celltype,layer,soma_x,soma_y,soma_z,bx0,by0,bz0,bx1,by1,bz1\n")
        for s, r in sorted(neurons.items()):
            b = bbox.get(s, [""] * 6)
            f.write(",".join(str(x) for x in [s, r["celltype"], r["layer"], r["x"], r["y"], r["z"]] + list(b)) + "\n")
    outdeg, indeg = {}, {}
    for (x, y) in pairs:
        outdeg[x] = outdeg.get(x, 0) + 1
        indeg[y] = indeg.get(y, 0) + 1
    summ = dict(parts=len(files), totals=tot, neurons_in_set=len(neurons), neurons_with_bbox=len(bbox),
                distinct_pairs=len(pairs), synapses_in_pairs=sum(sum(v) for v in pairs.values()),
                max_outdeg=max(outdeg.values(), default=0), max_indeg=max(indeg.values(), default=0),
                note="bbox = obwiednia synaps neuronu (proxy obciecia); jednostki wspolrzednych do potwierdzenia")
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--somas", default="somas.csv")
    ap.add_argument("--head", default="json_head.bin")
    ap.add_argument("--out", default="parts")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--expect", type=int, default=0)
    a = ap.parse_args()
    for flag, fn in (("selftest", cmd_selftest), ("list", cmd_list), ("run", cmd_run), ("merge", cmd_merge)):
        if getattr(a, flag):
            return fn(a)
    ap.print_help()


if __name__ == "__main__":
    main()
