"""Test 3, phase A2: collisions of later generations (event-driven).

    python -m src.test3_gen --nzeros 10 --extra 8 --workers 6 --out results/test3/gen_pilot.json
    python -m src.test3_gen --nzeros 100 --extra 12 --workers 6 --out results/test3/gen.json

Zeros z_0 < z_1 < ... (0-based in code, 1-based in output). Pair (a, b) of neighbouring *real* zeros: between
them sH > 0 (s = +-1), with humps (local maxima of sH). The pair collides when the highest hump of sH between
a and b reaches 0: g(t) = max sH on (a(t), b(t)) (continuous, piecewise smooth). The hump is tracked by
warm start: highest local maximum of sH with sH > 0 on the whole path from the previous position (this keeps
it inside the same interval); past t_c it is the nearest critical point (a minimum of |H|).
Event order: candidates by decreasing t_c, both zeros must be unused, otherwise `preempted`. After an accepted
collision (i, j) the new neighbours (L, R) = (real predecessor of i, real successor of j) form a pair of
generation 1 + max(generation of removed zeros between them), started at t_c from the humps of pairs (L, i)
and (j, R). No spawn if i is the lowest real zero (the mirror pair (-z, z) never collides: H_t(0) > 0).
Step in t: dt = 0.6 g / (dg/dt), dg/dt = -s H_xx, clamped to [0.02, 0.5]; on a lost hump the step is halved.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from multiprocessing import Pool
from typing import cast

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from src.heatflow import HeatFlow
from src.test3 import DIGITS, Absorbed, _G, _hi, _init, newton, zeros_z

DT_MAX, DT_MIN = 0.5, 0.02


def hump(eng, t, x0, w, s, ngrid):
    """Highest local max of sH with sH > 0 on the path from x0 (else the nearest critical point).
    Returns (x, g = sH(x), dg/dt)."""
    xs = np.linspace(x0 - w, x0 + w, ngrid)
    d = [float(eng.derivs(x, t)[1]) for x in xs]
    roots = [cast(float, brentq(lambda x: float(eng.derivs(x, t)[1]), xs[k], xs[k + 1], xtol=1e-13,
                                rtol=np.float64(1e-14)))
             for k in range(ngrid - 1) if d[k] * d[k + 1] < 0]
    if not roots:
        raise Absorbed
    best = None
    for r in roots:
        h0, _, h2 = eng.derivs(r, t)
        if s * float(h0) > 0 and s * float(h2) < 0:
            ok = all(s * float(eng.H(x0 + f * (r - x0), t)) > 0 for f in (0.2, 0.4, 0.6, 0.8))
            if ok and (best is None or s * float(h0) > best[1]):
                best = (r, s * float(h0), -s * float(h2))
    if best is not None:
        return best
    r = min(roots, key=lambda r: abs(r - x0))
    h0, _, h2 = eng.derivs(r, t)
    return r, s * float(h0), -s * float(h2)


def track_zero(eng, x0, t_end, dt0=0.1, dt_min=0.0125):
    """Follow a real zero of H_t from t = 0 (x0) down to t_end by Newton with linear predictor; None if lost."""
    t, x, xp, tp = 0.0, float(x0), None, None
    dt = dt0
    while t > t_end + 1e-12:
        dtn = min(dt, t - t_end)
        tn = t - dtn
        pred = x if xp is None or tp is None else x + (x - xp) * (dtn / (tp - t))
        y, ok = mp.mpf(pred), False
        for _ in range(12):
            h0, h1, _ = eng.derivs(y, tn)
            step = h0 / h1
            y -= step
            if abs(step) < 1e-13:
                ok = True
                break
        y = float(y)
        if ok and abs(y - pred) < 0.05 + 0.3 * dtn:
            xp, tp, x, t = x, t, y, tn
            dt = min(dt0, dt * 1.5)
        elif dtn <= dt_min + 1e-12:
            return None
        else:
            dt = max(dt_min, dtn / 2)
    return x


def pair_tc(job):
    eng = _G["eng"]
    n0, t0 = eng.nev, time.time()
    a, b, ts, tmin = job["a"], job["b"], job["t_start"], job["tmin"]
    dts = job.get("dts", 1.0)
    gf = job.get("grid_factor", 1)
    dt_max, dt_min = DT_MAX * dts, DT_MIN * dts
    span = b - a
    w = max(1.5, (0.6 if job["inter"] else 0.35) * span)
    ngrid = int(gf * max(24, int(2 * w / 0.6)))
    rec = dict(i=job["i"], j=job["j"], a=a, b=b, gen=job["gen"], parent=job["parent"], t_start=ts,
               inter=job["inter"], tc0=-(span ** 2) / 8)
    try:
        if job["seeds"] is None:
            aa, bb = a, b
            if ts < 0:  # no seed from neighbours: follow both zeros from t = 0 to t_start, scan between them
                aa, bb = track_zero(eng, a, ts), track_zero(eng, b, ts)
                if aa is None or bb is None or bb <= aa:
                    raise Absorbed
                rec["zeros_at_start"] = [aa, bb]
            mid = 0.5 * (aa + bb)
            s = 1.0 if float(eng.H(mid, ts)) > 0 else -1.0
            x, g, sl = hump(eng, ts, mid, 0.5 * (bb - aa), s, int(gf * max(24, int((bb - aa) / 0.3))))
        else:
            s = 1.0 if float(eng.H(job["seeds"][0], ts)) > 0 else -1.0
            best = None
            for x0 in job["seeds"]:
                try:
                    c = hump(eng, ts, x0, 1.0, s, 24)
                except Absorbed:
                    continue
                if best is None or c[1] > best[1]:
                    best = c
            if best is None:
                raise Absorbed
            # neighbour humps are only seeds: take the highest hump of the whole interval, as the tracker does
            ws = 0.8 * span + 1.0
            xc = float(np.mean(job["seeds"]))
            x, g, sl = hump(eng, ts, xc, ws, s, int(gf * max(24, int(2 * ws / 0.3))))
    except Absorbed:
        rec.update(status="no_seed", nev=eng.nev - n0, seconds=time.time() - t0, path=[], dts=dts, _job=job)
        return rec
    path = [(ts, x)]
    t = ts
    bracket, status = None, "lost"
    while True:
        dt = dt_max if sl <= 0 else min(dt_max, max(dt_min, 0.6 * g / sl))
        got, tn = None, t
        while got is None:
            tn = t - dt
            if tn < tmin:
                break
            try:
                got = hump(eng, tn, x, w, s, ngrid)
            except Absorbed:
                if dt <= dt_min + 1e-12:
                    break
                dt = max(dt_min, dt / 2)
        if got is None:
            status = "lost" if tn >= tmin else "censored"  # lost -> absorbed only if a cause event exists
            rec["t_absorbed_upto"] = t
            break
        xn, gn, sln = got
        if gn < 0:
            bracket = (t, tn, x)
            break
        t, x, g, sl = tn, xn, gn, sln
        path.append((t, x))
    if bracket is not None:
        tp, tn, xp = bracket
        try:
            tc = cast(float, brentq(lambda tt: hump(eng, tt, xp, w, s, ngrid)[1], tn, tp, xtol=1e-12,
                                    rtol=np.float64(1e-14)))
            xstar = hump(eng, tc, xp, w, s, ngrid)[0]
            hi = _hi()
            xh = newton(hi, mp.mpf(xstar), tc)
            h0, _, h2 = hi.derivs(xh, tc)
            dtc = h0 / h2
            sg = []
            for te in (tc - 1e-6, tc + 1e-6):
                xe = newton(hi, xh, te)
                sg.append(float(s * hi.H(xe, te)))
            status = "candidate"
            rec.update(tc=tc, xstar=float(xstar), stab_dt=float(abs(dtc)), sign_flip=bool(sg[0] < 0 < sg[1]),
                       ratio=tc / rec["tc0"])
            path.append((tc, float(xstar)))
        except (Absorbed, ValueError) as ex:
            status = "lost"
            rec["note"] = repr(ex)[:80]
    rec.update(status=status, nev=eng.nev - n0, seconds=time.time() - t0, path=path, dts=dts, _job=job)
    return rec


def run_jobs(jobs, workers, zmax, tmin):
    if not jobs:
        return []
    if workers <= 1:
        _init(zmax, tmin)
        return [pair_tc(j) for j in jobs]
    with Pool(min(workers, len(jobs)), initializer=_init, initargs=(zmax, tmin)) as p:
        return p.map(pair_tc, jobs, chunksize=1)


def greedy(cands, n_zeros):
    """Event order by decreasing t_c. Returns (accepted events, preempted keys)."""
    real = list(range(n_zeros))
    used, acc, accset, pre = set(), [], {}, []
    accLR = {}
    for r in sorted([c for c in cands if c["status"] == "candidate"], key=lambda c: -c["tc"]):
        if r["parent"] is not None and not (tuple(r["parent"]) in accset
                                            and abs(accset[tuple(r["parent"])] - r["t_start"]) < 1e-9
                                            and accLR[tuple(r["parent"])] == (r["i"], r["j"])):
            continue
        i, j = r["i"], r["j"]
        if i in used or j in used:
            pre.append((i, j))
            continue
        used.update((i, j))
        k = real.index(i)
        L = real[k - 1] if k > 0 else None
        R = real[real.index(j) + 1] if real.index(j) + 1 < len(real) else None
        real.remove(i)
        real.remove(j)
        acc.append(dict(i=i, j=j, tc=r["tc"], gen=r["gen"], L=L, R=R, parent=r["parent"]))
        accset[(i, j)] = r["tc"]
        accLR[(i, j)] = (L, R)
    return acc, pre


def seed_at(rec, t):
    """Hump position of pair record at time t from its path (None if the path does not reach t)."""
    if rec is None or not rec.get("path"):
        return None
    p = np.array(rec["path"])
    if p[-1, 0] > t + 1e-9 or p[0, 0] < t - 1e-9:
        return None
    return float(np.interp(t, p[::-1, 0], p[::-1, 1]))


def cause_of(rec, acc):
    """Accepted collision of a zero of the pair with another neighbour, not later (in t) than the loss of the hump."""
    t_up = rec.get("t_absorbed_upto")
    if t_up is None:
        return None
    for e in acc:
        if (e["i"], e["j"]) != (rec["i"], rec["j"]) and {e["i"], e["j"]} & {rec["i"], rec["j"]} \
                and e["tc"] >= t_up - 0.05:
            return [e["i"] + 1, e["j"] + 1, e["tc"]]
    return None


def simulate(zf, workers, tmin, log=print, a_force_track=False, gen1=None, spawned=None):
    N = len(zf)
    zmax = zf[-1] + 2
    jobs = [dict(i=i, j=i + 1, a=zf[i], b=zf[i + 1], t_start=0.0, seeds=None, inter=0, gen=1, parent=None,
                 tmin=tmin) for i in range(N - 1)]
    if gen1 is None:
        gen1 = run_jobs(jobs, workers, zmax, tmin)
    spawned = {} if spawned is None else spawned
    wave = 0
    while True:
        cands = gen1 + [r for r in spawned.values() if r is not None]
        acc, pre = greedy(cands, N)
        accd = {(e["i"], e["j"]): (e["tc"], e["L"], e["R"]) for e in acc}
        valid = [r for r in cands if r["parent"] is None or (
            tuple(r["parent"]) in accd and abs(accd[tuple(r["parent"])][0] - r["t_start"]) < 1e-9
            and accd[tuple(r["parent"])][1:] == (r["i"], r["j"]))]
        lookup = {}
        for r in valid:
            lookup[(r["i"], r["j"])] = r
        todo = [e for e in acc if (e["i"], e["j"], round(e["tc"], 10), e["L"], e["R"]) not in spawned]
        if not todo:
            # status reconciliation (Rolle): a vanished gap needs a cause event, otherwise retry with dt/4
            lost = [r for r in valid if r["status"] in ("absorbed", "lost") and cause_of(r, acc) is None
                    and r["dts"] > 1 / 64]
            if not lost:
                break
            log(f"retrying {len(lost)} pairs without a cause event at dt scale {lost[0]['dts'] / 4}")
            res = run_jobs([dict(r["_job"], dts=r["dts"] / 4) for r in lost], workers, zmax, tmin)
            for old, new in zip(lost, res):
                for k, v in enumerate(gen1):
                    if v is old:
                        gen1[k] = new
                for kk, v in list(spawned.items()):
                    if v is old:
                        spawned[kk] = new
            continue
        rgen = {}
        for e in acc:
            rgen[e["i"]] = rgen[e["j"]] = e["gen"]
        jobs = []
        for e in todo:
            key = (e["i"], e["j"], round(e["tc"], 10), e["L"], e["R"])
            L, R = e["L"], e["R"]
            if L is None or R is None:
                spawned[key] = None
                continue
            seeds = [x for x in (seed_at(lookup.get((L, e["i"])), e["tc"]), seed_at(lookup.get((e["j"], R)), e["tc"]))
                     if x is not None]
            if a_force_track:
                seeds = []
            between = list(range(L + 1, R))
            jobs.append(dict(i=L, j=R, a=zf[L], b=zf[R], t_start=e["tc"], seeds=seeds or None, inter=len(between),
                             gen=1 + max(rgen.get(k, 0) for k in between), parent=[e["i"], e["j"]], tmin=tmin,
                             tracked=not seeds))
        real_jobs = jobs
        res = run_jobs(real_jobs, workers, zmax, tmin)
        for j, r in zip(real_jobs, res):
            spawned[(j["parent"][0], j["parent"][1], round(j["t_start"], 10), j["i"], j["j"])] = r
        wave += 1
        log(f"wave {wave}: {len(real_jobs)} new pairs, accepted events so far {len(acc)}")
    return gen1, spawned, acc, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nzeros", type=int, default=10)
    ap.add_argument("--extra", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tmin", type=float, default=-100.0)
    ap.add_argument("--ref", default="results/test3/tc.json", help="gen-1 results for regression")
    ap.add_argument("--force-track", action="store_true", help="validation: ignore neighbour seeds, track zeros")
    ap.add_argument("--resume", default=None, help="finished gen json to continue from (adds missing spawns)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    n = a.nzeros
    zf = [float(x) for x in zeros_z(n + 2 + a.extra)]
    g0 = s0 = None
    if a.resume:  # continue from a finished run (statuses back to raw tracker outcomes)
        old = json.load(open(a.resume))
        raw = {"collided": "candidate", "preempted": "candidate"}
        def load(r):
            r["_job"] = r.pop("job")
            r["status"] = raw.get(r["status"], r["status"])
            return r
        g0 = [load(r) for r in old["gaps"]]
        s0 = {(r["parent"][0], r["parent"][1], round(r["t_start"], 10), r["i"], r["j"]): load(r) for r in old["spawned"]}
    gen1, spawned, acc, pre = simulate(zf, a.workers, a.tmin, log=lambda m: print(m, flush=True),
                                       a_force_track=a.force_track, gen1=g0, spawned=s0)
    accd = {(e["i"], e["j"]): (e["tc"], e["L"], e["R"]) for e in acc}
    sp = [r for r in spawned.values() if r is not None and tuple(r["parent"]) in accd
          and abs(accd[tuple(r["parent"])][0] - r["t_start"]) < 1e-9
          and accd[tuple(r["parent"])][1:] == (r["i"], r["j"])]
    cands = gen1 + sp
    accset = set(accd)
    preset = set(pre)
    for r in cands:
        r["gap"], r["edge"] = r["i"] + 1, r["i"] + 1 >= n
        if r["status"] in ("absorbed", "lost", "censored"):
            # a pair ceases to exist when one of its zeros collides with another neighbour (Rolle); the tracked
            # hump may live on as a critical point, so this cause takes precedence over `censored`
            r["cause"] = cause_of(r, acc)
            if r["cause"]:
                r["status"] = "absorbed"
            elif r["status"] == "absorbed":
                r["status"] = "lost"
        elif r["status"] == "candidate":
            r["status"] = "collided" if (r["i"], r["j"]) in accset else "preempted"
        r["job"] = r.pop("_job", None)
    incons = []
    ev = [dict(zi=e["i"] + 1, zj=e["j"] + 1, tc=e["tc"], gen=e["gen"], parent=e["parent"]) for e in acc]
    ngen = {}
    for e in ev:
        ngen[e["gen"]] = ngen.get(e["gen"], 0) + 1
    g1 = [r for r in gen1 if r["status"] == "collided" and r["i"] + 1 < n]
    viol = [r["gap"] for r in g1 if r["ratio"] < 1 - 1e-9]
    reg = None
    if os.path.exists(a.ref):
        old = {r["gap"]: r for r in json.load(open(a.ref))["gaps"]}
        dif, chg = [], []
        for r in gen1:
            o = old.get(r["gap"])
            if o is None:
                continue
            if "tc" in o and "tc" in r and o["status"] == "collided":
                dif.append(abs(o["tc"] - r["tc"]))
            elif ("tc" in o) != ("tc" in r):
                chg.append((r["gap"], o["status"], r["status"]))
        reg = dict(max_abs_dtc=max(dif) if dif else None, n_common=len(dif), status_changes=chg)
    stat = {k: sum(1 for r in cands if r["status"] == k)
            for k in ("collided", "preempted", "absorbed", "lost", "censored", "no_seed")}
    for r in cands:
        r["path"] = [[round(t, 6), round(x, 6)] for t, x in r.get("path", [])]
    used = {k for e in acc for k in (e["i"], e["j"])}
    surv = [k for k in range(len(zf)) if k not in used]
    have = {(r["i"], r["j"]) for r in cands}
    missing = [[x + 1, y + 1] for x, y in zip(surv, surv[1:]) if (x, y) not in have]
    nospawn = [[e["i"] + 1, e["j"] + 1] for e in acc if (e["L"] is None or e["R"] is None)]
    out = dict(n=n, extra=a.extra, zeros=zf, events=ev, events_per_generation=ngen, pair_status=stat,
               gen1_ratio_violations=viol, survivors=[k + 1 for k in surv], missing_survivor_pairs=missing,
               events_without_spawn_edge=nospawn, censored_inconsistent=incons, regression_vs_gen1=reg, gaps=gen1, spawned=sp,
               seconds=time.time() - t0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"gen n={n}: events per generation {ngen}; pair status {stat}; survivors {[k + 1 for k in surv]} missing pairs {missing}; gen1 ratio violations {viol}; censored inconsistent {incons}; "
          f"{time.time() - t0:.0f}s")
    print(f"regression vs {a.ref}: {json.dumps(reg)[:400]}")
    for e in sorted(ev, key=lambda e: e["gen"])[-8:]:
        if e["gen"] > 1:
            print(f"  gen {e['gen']}: zeros {e['zi']}-{e['zj']} t_c {e['tc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
