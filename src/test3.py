"""Test 3, phase A: heat flow H_t at low heights and collision times t_c of neighbouring zero pairs.

    python -m src.test3 validate --out results/test3/validate.json
    python -m src.test3 pilot --nzeros 10 --out results/test3/tc_pilot.json
    python -m src.test3 tc --nzeros 100 --workers 6 --out results/test3/tc.json

Zeros of H_0 are z_n = 2 gamma_n. For gap j (zeros z_j, z_{j+1}) t_c is the root in t of
g(t) = s H_t(x*(t)), x*(t) the local extremum of H_t in the gap, s = sign of H_0 there. Isolated-pair
approximation t_c0 = -dz^2/8. Event order: each zero takes part in at most one collision; gaps are
processed from t_c closest to 0 downwards, a collision is accepted only if both zeros are still real,
the remaining gaps are marked `preempted`. Control invariant: for a non-preempted gap the other zeros
act as a tide that stretches the gap, so t_c/t_c0 >= 1 (close to 1 for tight pairs); ratio < 1 is a bug.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from src.heatflow import HeatFlow, h_quad

DIGITS = 20
NGRID = 24
_G = {}


def zeros_z(n, dps=40):
    """z_k = 2 gamma_k, k = 1..n, as mpf strings (Arb, ball radius checked)."""
    import flint
    flint.ctx.dps = dps
    zs = flint.acb.zeta_zeros(1, n)
    assert max(float(z.imag.rad()) for z in zs) < 1e-25
    return [2 * mp.mpf(z.imag.mid().str(dps - 4, radius=False)) for z in zs]


def xi_over_8(z, dps=50):
    with mp.workdps(dps):
        s = mp.mpf(1) / 2 + 1j * mp.mpf(z) / 2
        return mp.mpf(1) / 2 * s * (s - 1) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s) / 8


def newton(eng, x, t, it=8):
    for _ in range(it):
        _, h1, h2 = eng.derivs(x, t)
        x = x - h1 / h2
    return x


def newton_root(eng, x, t, it=8):
    for _ in range(it):
        h0, h1, _ = eng.derivs(x, t)
        x = x - h0 / h1
    return x


# ---------------------------------------------------------------- validate
def cmd_validate(a):
    t0 = time.time()
    z = zeros_z(51)
    zmax = float(z[49]) + 2
    eng = HeatFlow(zmax, DIGITS, tmin=0.0)
    pts = [0, 10, float(z[0]) + 0.5, 50, 100, 200, float(z[49]) + 0.5]  # off the zeros: rel. error ill-defined there
    rows, worst_re, worst_im = [], 0.0, 0.0
    for p in pts:
        ref = xi_over_8(p)
        h = eng.H(p, 0.0)
        re_err = abs(h - ref.real) / abs(ref.real)
        im_err = abs(ref.imag) / abs(ref.real)
        worst_re, worst_im = max(worst_re, float(re_err)), max(worst_im, float(im_err))
        rows.append(dict(z=p, H0=mp.nstr(h, 12), rel_err=float(re_err), rel_im=float(im_err)))
    ok_val = worst_re < 1e-15 and worst_im < 1e-15
    dz = []
    for n in range(50):
        xr = newton_root(eng, z[n], 0.0)
        dz.append(float(abs(xr - z[n])))
    ok_zero = max(dz) < 1e-8
    xs = np.arange(0.0, float(z[49]) + 1.0, 0.25)
    sg = [mp.sign(eng.H(x, 0.0)) for x in xs[1:]]
    nchg = sum(1 for i in range(1, len(sg)) if sg[i] != sg[i - 1])
    ok_cnt = nchg == 50
    # info: zdot = H_xx/H_x vs truncated 2 sum 1/(z_j - z_k) over +-z_k, k <= 51
    zdot = []
    for j in range(5):
        _, h1, h2 = eng.derivs(z[j], 0.0)
        lhs = h2 / h1
        rhs = sum(2 / (z[j] - z[k]) for k in range(51) if k != j) + sum(2 / (z[j] + z[k]) for k in range(51))
        zdot.append((float(lhs), float(rhs)))
    out = dict(ok=bool(ok_val and ok_zero and ok_cnt), dps=eng.dps, nodes=eng.K + 1, h=eng.h,
               max_rel_err_H0=worst_re, max_rel_im=worst_im, max_zero_dev=max(dz), n_sign_changes=nchg,
               zdot_vs_truncated_sum=zdot, points=rows, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"validate: value err {worst_re:.2e} (im {worst_im:.2e}) {'OK' if ok_val else 'FAIL'}; "
          f"max |z_n-2g_n| {max(dz):.2e} {'OK' if ok_zero else 'FAIL'}; sign changes {nchg} "
          f"{'OK' if ok_cnt else 'FAIL'}; dps {eng.dps}, nodes {eng.K + 1}; {time.time() - t0:.1f}s")
    print("zdot H_xx/H_x vs truncated sum (info):", ", ".join(f"{l:.4f}/{r:.4f}" for l, r in zdot[:3]))
    return 0 if out["ok"] else 1


# ---------------------------------------------------------------- t_c for one gap
class Absorbed(Exception):
    pass


def find_xstar(eng, t, a, b, xprev):
    """Local extremum of H_t in the gap (no H*H_xx filter: after t_c it is a minimum of |H|).
    Start: max |H| among critical points in [a, b].
    Later: tracked from xprev (zeros drift with t) -- scan of H_x in a window around xprev, nearest root."""
    if xprev is None:
        lo, hi = a, b
    else:
        w = 0.35 * (b - a)
        lo, hi = xprev - w, xprev + w
    xs = np.linspace(lo, hi, NGRID)
    d = [float(eng.derivs(x, t)[1]) for x in xs]
    roots = []
    for i in range(NGRID - 1):
        if d[i] * d[i + 1] < 0:
            r = brentq(lambda x: float(eng.derivs(x, t)[1]), xs[i], xs[i + 1], xtol=1e-13, rtol=1e-14)
            roots.append(r)
    if not roots:
        raise Absorbed
    if xprev is None:
        return max(roots, key=lambda r: abs(float(eng.H(r, t))))
    return min(roots, key=lambda r: abs(r - xprev))


def gap_tc(args):
    i, a, b, tmin = args
    eng = _G["eng"]
    n0 = eng.nev
    t0 = time.time()
    tc0 = -(b - a) ** 2 / 8
    rec = dict(gap=i + 1, z_lo=a, z_hi=b, tc0=tc0)
    s = 1.0 if float(eng.H(0.5 * (a + b), 0.0)) > 0 else -1.0
    state = dict(x=None)

    def g(t):
        x = find_xstar(eng, t, a, b, state["x"])
        state["x"] = x
        return s * float(eng.H(x, t))

    fs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2, 2.5, 3, 4, 6, 8]
    prev = None
    bracket = None
    status = "no_bracket"
    for f in fs:
        t = f * tc0
        if t < tmin:
            break
        try:
            gt = g(t)
        except Absorbed:
            status = "absorbed"
            rec["t_absorbed_upto"] = t
            break
        if gt < 0:
            if prev is None:
                status = "no_bracket"
                rec["note"] = "g<0 already at 0.05*tc0"
            else:
                bracket = (prev[0], t)
            break
        prev = (t, gt)
        state["xprev_ok"] = state["x"]
    if bracket is not None:
        state["x"] = state.get("xprev_ok")
        try:
            tc = brentq(g, bracket[1], bracket[0], xtol=1e-12, rtol=1e-14)
            xstar = state["x"]
            # stability: double dps, half h; Newton correction dt = H/H_xx at x*
            hi = _hi()
            xh = newton(hi, mp.mpf(xstar), tc)
            h0, _, h2 = hi.derivs(xh, tc)
            dt = h0 / h2
            eps = 1e-6
            sg = []
            for te in (tc - eps, tc + eps):
                xe = newton(hi, xh, te)
                sg.append(float(s * hi.H(xe, te)))
            status = "candidate"
            rec.update(tc=tc, xstar=float(xstar), stab_dt=float(abs(dt)), gm=sg[0], gp=sg[1],
                       sign_flip=bool(sg[0] < 0 < sg[1]), ratio=tc / tc0)
        except (Absorbed, ValueError) as e:
            status = "absorbed" if isinstance(e, Absorbed) else "no_bracket"
            rec["note"] = repr(e)[:80]
    rec.update(status=status, nev=eng.nev - n0, seconds=time.time() - t0)
    return rec


def _hi():
    if "hi" not in _G:
        e = _G["eng"]
        _G["hi"] = HeatFlow(e.zmax, DIGITS, e.tmin, dps_factor=2.0, h_factor=2.0)
    return _G["hi"]


def _init(zmax, tmin):
    _G["eng"] = HeatFlow(zmax, DIGITS, tmin=tmin)


def run_gaps(zf, gaps, workers, tmin):
    zmax = zf[-1] + 2
    jobs = [(i, zf[i], zf[i + 1], tmin) for i in gaps]
    if workers <= 1:
        _init(zmax, tmin)
        return [gap_tc(j) for j in jobs]
    with Pool(workers, initializer=_init, initargs=(zmax, tmin)) as p:
        return list(p.imap(gap_tc, jobs, chunksize=1))


def order_events(recs):
    """Greedy event order: closest to t=0 first; both zeros must still be real."""
    cand = sorted([r for r in recs if r["status"] == "candidate"], key=lambda r: -r["tc"])
    used = set()
    for r in cand:
        j = r["gap"]
        if j in used or j + 1 in used:
            r["status"] = "preempted"
        else:
            r["status"] = "collided"
            used.update((j, j + 1))
    return recs


def add_geometry(recs, zf):
    zl = [-zf[0]] + list(zf)  # mirror zero -z_1 as left neighbour of gap 1
    for r in recs:
        i = r["gap"] - 1
        r["dz"] = zf[i + 1] - zf[i]
        r["d_left"] = zl[i + 1] - zl[i]
        r["d_right"] = zf[i + 2] - zf[i + 1] if i + 2 < len(zf) else None


def summarize(recs):
    recs = [r for r in recs if not r.get("edge")]
    col = [r for r in recs if r["status"] == "collided"]
    bad = [r["gap"] for r in col if r["ratio"] < 1 - 1e-9]
    st = {k: sum(1 for r in recs if r["status"] == k) for k in ("collided", "preempted", "absorbed", "no_bracket")}
    rr = [r["ratio"] for r in col]
    return dict(counts=st, ratio_min=min(rr) if rr else None, ratio_max=max(rr) if rr else None,
                invariant_violations=bad, max_stab_dt=max((r["stab_dt"] for r in col), default=None),
                all_sign_flip=all(r["sign_flip"] for r in col))


def cmd_tc(a, pilot=False):
    t0 = time.time()
    n = a.nzeros
    ex = getattr(a, "extra", 0)
    zm = zeros_z(n + 2 + ex)
    zf = [float(x) for x in zm]
    gaps = list(range(0, n - 1 + ex))  # gaps between zero j and j+1, j = 1..n-1 (+ ex edge gaps for counting)
    recs = run_gaps(zf[: n + 2 + ex], gaps, a.workers, a.tmin)
    for r in recs:
        r["edge"] = r["gap"] >= n
    add_geometry(recs, zf)
    e0 = HeatFlow(30.0, DIGITS, tmin=a.tmin)
    ts = np.linspace(a.tmin, 0.0, 201)
    h0 = [float(e0.H(0.0, float(t))) for t in ts]
    sym = dict(t_min=a.tmin, min_H_t0=min(h0), all_positive=bool(min(h0) > 0), n_t=len(ts))
    order_events(recs)
    summ = summarize(recs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(summary=summ, sym_gap_H_t0=sym, n=n, gaps=recs, zeros=zf, seconds=time.time() - t0), open(a.out, "w"), indent=1)
    print(f"H_t(0) > 0 on [{a.tmin}, 0] ({sym['n_t']} points): {sym['all_positive']}, min {sym['min_H_t0']:.4e}")
    print(f"tc n={n}: {summ['counts']}; ratio t_c/t_c0 in [{summ['ratio_min']}, {summ['ratio_max']}]; "
          f"violations {summ['invariant_violations']}; max stab dt {summ['max_stab_dt']}; "
          f"sign_flip all {summ['all_sign_flip']}; {time.time() - t0:.1f}s")
    for r in recs[:12]:
        print(f"  gap {r['gap']:3d} dz {r['dz']:6.3f} {r['status']:10s} "
              f"tc {r.get('tc', float('nan')):9.4f} tc0 {r['tc0']:9.4f} ratio {r.get('ratio', float('nan')):6.3f} "
              f"nev {r['nev']} {r['seconds']:.1f}s")
    return recs, zf


def cmd_pilot(a):
    a.workers = 1
    recs, zf = cmd_tc(a, pilot=True)
    # timing of one evaluation vs z: trapezoid vs mp.quad, and extrapolation to n = 100
    tm = {}
    for zz in (30.0, 100.0, 250.0, 480.0):
        e = HeatFlow(zz + 2, DIGITS, tmin=a.tmin)
        t1 = time.time()
        for _ in range(5):
            e.derivs(zz - 0.3, -3.0)
        tm[zz] = (time.time() - t1) / 5
    q_z = 28.0
    e = HeatFlow(q_z + 2, DIGITS, tmin=a.tmin)
    t1 = time.time()
    hq = h_quad(q_z, -2.0, e.dps)
    tq = time.time() - t1
    ht = e.H(q_z, -2.0)
    dq = float(abs(hq - ht) / abs(hq))
    zz = np.array(list(tm)); tt = np.array(list(tm.values()))
    p = np.polyfit(np.log(zz), np.log(tt), 1)
    from src.test3 import zeros_z as _zz
    z100 = [float(x) for x in _zz(101)]
    nev = np.mean([r["nev"] for r in recs])
    est = sum(nev * math.exp(np.polyval(p, math.log(z100[j + 1]))) for j in range(99))
    hi_ratio = 2.0  # hi-res engine ~ 4x per eval, included in nev only for lo engine -> add 25%
    print(f"eval time s: " + ", ".join(f"z={k:.0f}: {v * 1e3:.1f} ms" for k, v in tm.items())
          + f"; power law ~ z^{p[0]:.2f}")
    print(f"trapezoid vs mp.quad at z=28, t=-2: rel diff {dq:.2e}, quad {tq:.1f}s")
    print(f"mean evaluations/gap (lo engine) {nev:.0f}; estimated serial time for 100 gaps ~ {est * 1.25:.0f} s "
          f"(~{est * 1.25 / 6:.0f} s on 6 workers)")
    return 0



# ---------------------------------------------------------------- independent zero count
def count_window(args):
    t, X, step = args
    eng = _G["eng"]
    xs = np.arange(step / 2, X, step)
    prev = float(eng.H(0.0, t)) > 0
    n = 0
    for x in xs:
        cur = float(eng.H(float(x), t)) > 0
        n += cur != prev
        prev = cur
    return t, n


def cmd_count(a):
    """Zero count of H_t on the comoving window (0, X_e(t)): X_e tracked extremum of an edge gap e above
    the analysed zeros. Prediction: e - 2 * #{accepted collisions with both zeros <= e and t_c > t}."""
    t0 = time.time()
    d = json.load(open(a.tc))
    recs, zf, n = d["gaps"], d["zeros"], d["n"]
    valid = {}
    for r in recs:
        if r["edge"]:
            valid[r["gap"]] = r["tc"] if "tc" in r else r.get("t_absorbed_upto", 0.0)
    e = min(valid, key=lambda k: valid[k])
    v = valid[e]
    if "events" in d:  # all generations (src.test3_gen)
        evs = [(x["zi"], x["zj"], x["tc"]) for x in d["events"]]
    else:
        evs = [(r["gap"], r["gap"] + 1, r["tc"]) for r in recs if r["status"] == "collided"]
    strad = [tc for zi, zj, tc in evs if zi <= e < zj]  # collision straddling the window edge: stop checking
    if strad:
        v = max(v, max(strad))
    tcs = sorted([tc for zi, zj, tc in evs if zj <= e], reverse=True)
    grid = [0.0] + [0.5 * (tcs[i] + tcs[i + 1]) for i in range(len(tcs) - 1)] + [tcs[-1] - 0.5]
    grid = [t for t in grid if t > v + 0.05] if v < 0 else [0.0]
    zmax = zf[e] + 2
    eng = HeatFlow(zmax, DIGITS, tmin=a.tmin)
    _G["eng"] = eng
    ai, bi = zf[e - 1], zf[e]
    xe = None
    tcur, X = 0.0, []
    for t in grid:
        while tcur > t + 1e-12:
            tcur = max(t, tcur - 0.25)
            xe = find_xstar(eng, tcur, ai, bi, xe)
        if xe is None:
            xe = find_xstar(eng, 0.0, ai, bi, None)
        X.append(float(xe))
    jobs = [(t, x, a.step) for t, x in zip(grid, X)]
    with Pool(a.workers, initializer=_init, initargs=(zmax, a.tmin)) as p:
        res = p.map(count_window, jobs, chunksize=1)
    rows = []
    for (t, nn), x in zip(res, X):
        pred = e - 2 * sum(1 for tc in tcs if tc > t)
        rows.append(dict(t=t, X=x, counted=nn, predicted=pred, diff=nn - pred))
    counts = [r["counted"] for r in rows]
    mono = all(counts[i + 1] <= counts[i] for i in range(len(counts) - 1))
    first_bad = next((r for r in rows if r["diff"] != 0), None)
    out = dict(edge_gap=e, edge_valid_until=v, n_points=len(rows), monotone_nonincreasing=mono,
               all_match=first_bad is None, first_mismatch=first_bad, rows=rows, seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"count: edge gap {e} valid down to t={v:.3f}; {len(rows)} times; count never increases: {mono}; "
          f"all equal prediction: {first_bad is None}; {time.time() - t0:.0f}s")
    for r in rows[:6] + ([first_bad] if first_bad else []):
        print(f"  t {r['t']:9.4f} X {r['X']:8.2f} counted {r['counted']} predicted {r['predicted']} diff {r['diff']}")
    return 0


def count_div(args):
    return count_window(args)


def cmd_count_div(a):
    """Zero count on windows (0, X) with X a hump of a *living* pair (p, q) of neighbouring real zeros.
    Living at t: path of the pair record covers t and neither p nor q took part in a collision with t_c > t.
    Prediction: p - 2 * #{collisions with t_c > t and both zeros < p} (p = 1-based number of the lower zero)."""
    t0 = time.time()
    d = json.load(open(a.tc))
    recs = [r for r in d["gaps"] + d["spawned"] if r.get("path")]
    ev = sorted([(x["zi"], x["zj"], x["tc"]) for x in d["events"]], key=lambda e: -e[2])
    tcs = [e[2] for e in ev]
    grid = [0.0] + [0.5 * (tcs[i] + tcs[i + 1]) for i in range(len(tcs) - 1) if tcs[i] - tcs[i + 1] > 0.03]
    grid.append(tcs[-1] - 0.5)
    grid = [t for t in grid if t > a.tmin]
    plan = []
    for t in grid:
        used = {z for zi, zj, tc in ev if tc > t for z in (zi, zj)}
        alive = []
        for r in recs:
            p, q = r["i"] + 1, r["j"] + 1
            pa = r["path"]
            if p in used or q in used or not (pa[0][0] + 1e-9 >= t >= pa[-1][0] - 1e-9):
                continue
            ts_ = [u[0] for u in pa][::-1]
            xs_ = [u[1] for u in pa][::-1]
            alive.append((p, q, float(np.interp(t, ts_, xs_))))
        hi = sorted([x for x in alive if x[0] >= a.pcover])
        mid = sorted([x for x in alive if a.pmin <= x[0] < a.pcover])
        pick = hi[0] if hi else (mid[0] if mid else (max(alive) if alive else None))
        plan.append((t, pick, bool(hi)))
    jobs = [(t, pk[2], a.step) for t, pk, _ in plan if pk]
    zmax = max(j[1] for j in jobs) + 2
    _G["eng"] = HeatFlow(zmax, DIGITS, tmin=a.tmin)
    with Pool(a.workers, initializer=_init, initargs=(zmax, a.tmin)) as pool:
        res = dict(pool.map(count_window, jobs, chunksize=1))
    rows = []
    for t, pk, full in plan:
        if pk is None:
            rows.append(dict(t=t, note="no living divider"))
            continue
        p, q, X = pk
        pred = p - 2 * sum(1 for zi, zj, tc in ev if tc > t and zj < p)
        unver = sum(1 for zi, zj, tc in ev if tc > t and zj > p)
        rows.append(dict(t=t, p=p, q=q, X=X, counted=res[t], predicted=pred, diff=res[t] - pred,
                         covers_all_above=full, unverified_events=unver))
    chk = [r for r in rows if "diff" in r]
    bad = [r for r in chk if r["diff"] != 0]
    ev5556 = [r for r in chk if r["t"] < -76.6 and r["p"] >= 57]
    out = dict(n_points=len(rows), n_no_divider=len(rows) - len(chk), all_match=not bad, mismatches=bad,
               t_min_checked=min(r["t"] for r in chk), points_below_m76_6_p_ge_57=len(ev5556),
               match_below_76_6=all(r["diff"] == 0 for r in ev5556) if ev5556 else None, rows=rows,
               seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"count(dividers): {len(rows)} times, checked {len(chk)}, no divider {len(rows) - len(chk)}; "
          f"all equal prediction: {not bad}; lowest t checked {out['t_min_checked']:.2f}; "
          f"points below -76.6 with p>=57: {len(ev5556)} match: {out['match_below_76_6']}; {time.time() - t0:.0f}s")
    for r in (bad[:5] or chk[-4:]):
        print(f"  t {r['t']:9.3f} p {r['p']} q {r['q']} X {r['X']:8.2f} counted {r['counted']} predicted {r['predicted']} "
              f"diff {r['diff']} unverified {r['unverified_events']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--out", required=True)
    c = sub.add_parser("count")
    c.add_argument("--tc", required=True); c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=6); c.add_argument("--tmin", type=float, default=-100.0)
    c.add_argument("--step", type=float, default=0.1)
    c.add_argument("--dividers", action="store_true", help="window edge = hump of any living pair (gen.json with paths)")
    c.add_argument("--pcover", type=int, default=99); c.add_argument("--pmin", type=int, default=57)
    for name in ("pilot", "tc"):
        s = sub.add_parser(name)
        s.add_argument("--nzeros", type=int, default=10 if name == "pilot" else 100)
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--tmin", type=float, default=-100.0)
        s.add_argument("--extra", type=int, default=0 if name == "pilot" else 12)
        s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "validate":
        return cmd_validate(a)
    if a.cmd == "count":
        return cmd_count_div(a) if a.dividers else cmd_count(a)
    if a.cmd == "pilot":
        return cmd_pilot(a)
    cmd_tc(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
