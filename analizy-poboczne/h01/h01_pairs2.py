#!/usr/bin/env python3
"""H01 (c3), wersja 2: pary neuron->neuron z rozbiciem na klase miejsca presynaptycznego (class_label) i typ rekordu.

Poprawki wzgledem wersji 1: identyfikatory i wspolrzedne w JSON to TEKSTY (rzutowane na int), polozenie z pola centroid,
selftest zglasza blad przy zerowej liczbie trafien. Tylko biblioteka standardowa. Wznawialne (part_*.json.gz w --out).

  python3 h01_pairs2.py --selftest --somas somas.csv --head json_head.bin
  python3 h01_pairs2.py --run   --somas somas.csv --out parts2 --workers 4 [--limit 2]
  python3 h01_pairs2.py --merge --somas somas.csv --out parts2 --expect 166

Wyjscie --merge (w --out):
  pairs_all.csv   pre,post,ax_t1,ax_t2,ax_other,dend,soma,unknown,other_class
                  (ax_* = rekordy z presynapta AXON wg typu rekordu 1/2/inny; dend/soma/unknown/other_class = klasa presynapty)
  pairs_axon.csv  pre,post,n_type1,n_type2,n_other   tylko pary z >=1 rekordem AXON, w formacie dla h01_components.py i h01_structure.py
  neurons.csv     jak w wersji 1 (obwiednia synaps wszystkich klas)
  summary.json
"""
import argparse
import csv
import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from multiprocessing import Pool

BUCKET = "h01-release"
PREFIX = "data/20210601/c3/synapses/exported/json/"
NEURON_TYPES = {"PYRAMIDAL", "INTERNEURON", "SPINY_ATYPICAL", "UNCLASSIFIED_NEURON", "SPINY_STELLATE", "C_SHAPED"}


def _i(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return None


def load_somas(path):
    rows = list(csv.DictReader(open(path, newline="")))
    per_seg = {}
    for r in rows:
        seg = _i(r["c3_rep_manual"]) or 0
        if seg:
            per_seg[seg] = per_seg.get(seg, 0) + 1
    out = {}
    for r in rows:
        seg = _i(r["c3_rep_manual"]) or 0
        if seg and r["celltype"] in NEURON_TYPES and per_seg[seg] == 1:
            out[seg] = r
    return out


def _c(x, y, z):
    x, y, z = _i(x), _i(y), _i(z)
    return None if None in (x, y, z) else (x, y, z)


def _xyz(o):
    if isinstance(o, dict):
        return _c(o.get("x"), o.get("y"), o.get("z"))
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
    """pairs[(pre,post)] = [ax_t1, ax_t2, ax_other, dend, soma, unknown, other_class]"""
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
        pre, post = rec.get("pre_synaptic_site"), rec.get("post_synaptic_partner")
        if not isinstance(pre, dict) or post is None:
            st["no_ids"] += 1
            continue
        pid = _i(pre.get("neuron_id"))
        rloc = _xyz(rec.get("location"))
        pre_in = pid in nset
        if pre_in:
            st["pre_in"] += 1
            _upd(bbox, pid, _xyz(pre.get("centroid") or pre.get("location")) or rloc)
        t = _i(rec.get("type"))
        ti = 0 if t == 1 else (1 if t == 2 else 2)
        pc = pre.get("class_label")
        for po in (post if isinstance(post, list) else [post]):
            if not isinstance(po, dict):
                continue
            qid = _i(po.get("neuron_id"))
            q_in = qid in nset
            if q_in:
                st["post_in"] += 1
                _upd(bbox, qid, _xyz(po.get("centroid") or po.get("location")) or rloc)
            if pre_in and q_in:
                st["pair_in"] += 1
                v = pairs.setdefault((pid, qid), [0] * 7)
                if pc == "AXON":
                    v[ti] += 1
                elif pc == "DENDRITE":
                    v[3] += 1
                elif pc == "SOMA":
                    v[4] += 1
                elif pc == "UNKNOWN":
                    v[5] += 1
                else:
                    v[6] += 1
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
            nb = [0]

            def gen(resp):
                for ln in resp:
                    nb[0] += len(ln)
                    yield ln

            req = urllib.request.Request(url, headers={"User-Agent": "h01-pairs2/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                pairs, bbox, st = process_lines(gen(resp), nset)
            st["bytes"] = nb[0]
            return pairs, bbox, st
        except Exception as e:
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
    nset = set(load_somas(a.somas))
    print("neurony jednodusze: %d (oczekiwano ok. 15487)" % len(nset))
    lines = open(a.head, "rb").read().split(b"\n")[:-1]
    pairs, bbox, st = process_lines(lines, nset)
    print("statystyki na naglowku:", st, "| pary:", len(pairs), "| neurony z bbox:", len(bbox))
    ok = st["records"] > 0 and st["no_ids"] == 0 and (st["pre_in"] + st["post_in"]) > 0
    print("WYNIK: %s" % ("OK" if ok else "BLAD: brak trafien w zbiorze neuronow - nie uruchamiaj --run"))


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    nset = set(load_somas(a.somas))
    fs = list_files()
    if a.limit:
        fs = fs[:a.limit]
    todo = [(n, obj_url(n), a.out, nset) for n, _ in fs if not os.path.exists(_part_path(a.out, n))]
    print("plikow: %d, do zrobienia: %d, workerow: %d, start %s" % (len(fs), len(todo), a.workers,
                                                                  time.strftime("%H:%M:%S")), flush=True)
    t0, done, tb = time.time(), 0, 0
    with Pool(a.workers) as pool:
        for name, dt, st in pool.imap_unordered(worker, todo):
            done += 1
            if st:
                tb += st["bytes"]
            el = time.time() - t0
            print("[%d/%d] %s %.0fs rekordow=%s pary=%s | %.1f MB/s | ETA %.0f min" % (
                done, len(todo), os.path.basename(name), dt, st and st["records"], st and st["pair_in"],
                tb / max(el, 1) / 1e6, el / done * (len(todo) - done) / 60), flush=True)
    print("koniec %s" % time.strftime("%H:%M:%S"), flush=True)


def cmd_merge(a):
    neurons = load_somas(a.somas)
    files = sorted(f for f in os.listdir(a.out) if f.startswith("part_") and f.endswith(".json.gz"))
    print("plikow czesciowych: %d (oczekiwano %s)" % (len(files), a.expect))
    if a.expect and len(files) != a.expect:
        print("UWAGA: niekompletne - nie analizuj")
    pairs, bbox = {}, {}
    tot = dict(records=0, bad=0, no_ids=0, pre_in=0, post_in=0, pair_in=0, bytes=0)
    for f in files:
        d = json.load(gzip.open(os.path.join(a.out, f), "rt"))
        for k in tot:
            tot[k] += d["stats"].get(k, 0)
        for p in d["pairs"]:
            v = pairs.setdefault((p[0], p[1]), [0] * 7)
            for i in range(7):
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
    with open(os.path.join(a.out, "pairs_all.csv"), "w") as f:
        f.write("pre,post,ax_t1,ax_t2,ax_other,dend,soma,unknown,other_class\n")
        for (x, y), v in sorted(pairs.items()):
            f.write("%d,%d,%s\n" % (x, y, ",".join(str(t) for t in v)))
    n_ax_pairs = 0
    with open(os.path.join(a.out, "pairs_axon.csv"), "w") as f:
        f.write("pre,post,n_type1,n_type2,n_other\n")
        for (x, y), v in sorted(pairs.items()):
            if v[0] + v[1] + v[2] > 0:
                f.write("%d,%d,%d,%d,%d\n" % (x, y, v[0], v[1], v[2]))
                n_ax_pairs += 1
    with open(os.path.join(a.out, "neurons.csv"), "w") as f:
        f.write("c3_seg,celltype,layer,soma_x,soma_y,soma_z,bx0,by0,bz0,bx1,by1,bz1\n")
        for s, r in sorted(neurons.items()):
            b = bbox.get(s, [""] * 6)
            f.write(",".join(str(x) for x in [s, r["celltype"], r["layer"], r["x"], r["y"], r["z"]] + list(b)) + "\n")
    cls = [sum(v[i] for v in pairs.values()) for i in range(7)]
    summ = dict(parts=len(files), totals=tot, neurons_in_set=len(neurons), neurons_with_bbox=len(bbox),
                distinct_pairs_all=len(pairs), distinct_pairs_axon=n_ax_pairs,
                records_in_pairs=sum(cls), records_pre_axon=cls[0] + cls[1] + cls[2], records_pre_dendrite=cls[3],
                records_pre_soma=cls[4], records_pre_unknown=cls[5], records_pre_other=cls[6],
                note="pairs_axon.csv = tylko rekordy z presynapta o klasie AXON")
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("selftest", "run", "merge"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--somas", default="somas.csv")
    ap.add_argument("--head", default="json_head.bin")
    ap.add_argument("--out", default="parts2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--expect", type=int, default=0)
    a = ap.parse_args()
    for flag, fn in (("selftest", cmd_selftest), ("run", cmd_run), ("merge", cmd_merge)):
        if getattr(a, flag):
            return fn(a)
    ap.print_help()


if __name__ == "__main__":
    main()
