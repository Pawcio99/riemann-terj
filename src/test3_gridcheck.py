"""Scan-grid density check for selected pairs (rerun with grid_factor > 1, compare start hump and outcome).

    python -m src.test3_gridcheck --gen results/test3/gen2.json --pairs 6-21,42-77,90-99,99-114 \\
        --factors 2,4 --workers 6 --out results/test3/gridcheck.json
Outcome key: t_c if the pair collides, otherwise the last time at which the hump was still tracked.
"""
import argparse
import json
import time
from typing import Any

from src.test3_gen import run_jobs


def outcome(r):
    return r["tc"] if "tc" in r else r.get("t_absorbed_upto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--factors", default="2,4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tmin", type=float, default=-100.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    d = json.load(open(a.gen))
    want = {tuple(int(v) for v in p.split("-")) for p in a.pairs.split(",")}
    recs = [r for r in d["gaps"] + d["spawned"] if (r["i"] + 1, r["j"] + 1) in want and r.get("job")]
    facs = [int(f) for f in a.factors.split(",")]
    jobs = [dict(r["job"], grid_factor=f, dts=r.get("dts", 1.0)) for r in recs for f in facs]
    print(f"gridcheck: {len(recs)} pairs found of {len(want)} requested "
          f"({sorted(want - {(r['i'] + 1, r['j'] + 1) for r in recs})} without record), factors {facs}, "
          f"{len(jobs)} jobs started", flush=True)
    res = run_jobs(jobs, a.workers, d["zeros"][-1] + 2, a.tmin)
    out, k = [], 0
    for r in recs:
        base: dict[str, Any] = dict(x0=r["path"][0][1] if r.get("path") else None, out=outcome(r), status=r["status"])
        row: dict[str, Any] = dict(pair=[r["i"] + 1, r["j"] + 1], gen=r["gen"], tracked=bool(r.get("zeros_at_start")), base=base, runs=[])
        for f in facs:
            q = res[k]
            k += 1
            x0 = q["path"][0][1] if q.get("path") else None
            o = outcome(q)
            row["runs"].append(dict(factor=f, x0=x0, out=o, raw_status=q["status"],
                                    dx0=None if x0 is None or base["x0"] is None else abs(x0 - base["x0"]),
                                    dout=None if o is None or base["out"] is None else abs(o - base["out"])))
            print(f"  {row['pair']} gen {r['gen']} f{f}: dx0 {row['runs'][-1]['dx0']} out {o} d_out {row['runs'][-1]['dout']} raw {q['status']}")
        out.append(row)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
