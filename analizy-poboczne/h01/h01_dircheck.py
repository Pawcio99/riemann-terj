#!/usr/bin/env python3
"""H01: czy neurony polaczone wieloma kontaktami maja je w obu kierunkach? Porownanie z wartoscia oczekiwana przy dowolnym kierunku (1 - 2^(1-n)).
Uzycie: python3 h01_dircheck.py parts2/pairs_all.csv 3 4 6   (plik z h01_pairs2.py --merge; nie zalezy od znaczenia pola type)"""
import csv, collections, sys
path = sys.argv[1] if len(sys.argv) > 1 else "parts2/pairs_all.csv"
NMIN = [int(x) for x in sys.argv[2:]] or [3, 4, 6]
cnt = {"axon": collections.defaultdict(lambda: [0, 0]), "dendryt": collections.defaultdict(lambda: [0, 0])}
for r in csv.DictReader(open(path, newline="")):
    a, b = int(r["pre"]), int(r["post"])
    if a == b:
        continue
    key, side = ((a, b), 0) if a < b else ((b, a), 1)
    ax = int(r["ax_t1"]) + int(r["ax_t2"]) + int(r["ax_other"])
    de = int(r["dend"])
    cnt["axon"][key][side] += ax
    cnt["dendryt"][key][side] += de
print("%-8s %4s %9s %10s %10s" % ("klasa", "n>=", "par nieuporz.", "oba kierunki", "oczekiwane przy dowolnym kierunku"))
for cls, d in cnt.items():
    for nm in NMIN:
        pairs = [(f, r) for f, r in d.values() if f + r >= nm]
        if not pairs:
            print("%-8s %4d %9d" % (cls, nm, 0)); continue
        both = sum(1 for f, r in pairs if f > 0 and r > 0)
        exp = sum(1 - 2.0 ** (1 - (f + r)) for f, r in pairs)
        print("%-8s %4d %9d %6d (%4.1f%%) %6.0f (%4.1f%%)" % (cls, nm, len(pairs), both, 100 * both / len(pairs), exp, 100 * exp / len(pairs)))
