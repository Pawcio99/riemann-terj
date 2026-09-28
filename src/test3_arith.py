"""Test 3, arithmetic residual test (plan: docs/PLAN_TEST3_B.md; report: results/test3/REPORT_B.md).

    python -m src.test3_arith corr --out results/test3/arith_corr.json    # stage 1: covariates and their geometry content
    python -m src.test3_arith test --out results/test3/arith_test.json    # stage 2: partial-correlation test

Sample: the 36 primary pairs with x < 1 (constant-tide predictor finite), and the cautious subset (n = 21).
Candidates (fixed in advance, 3 tests, Bonferroni alpha = 0.05/3):
  c1 = mean over the pair's two zeros of ln|zeta'(rho)| minus a smooth trend (linear in ln ln(gamma/2pi), fitted on all zeros),
  c2 = Gram phase deviation of the pair midpoint: |u - round(u)|, u = theta(gamma_mid)/pi (0..0.5),
  c3 = Gram-law margin: min over the two Gram points bracketing the midpoint of (-1)^k Z(g_k) / rms of Z(g_j) over 12 consecutive Gram points around k0 (n >= -1).
Residual r = ln(t_c/t_c0) - flexible function of x = ln(predicted ratio): isotonic regression (primary) or GCV smoothing spline.
Statistic: Spearman(r, c_perp), c_perp = c residualised (OLS) on [1, ln delta, ln dL, ln dR, x]; permutation p (20000, seed).
"""
import argparse
import json
import math
import os
from typing import Any, cast

import mpmath as mp
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import isotonic_regression
from scipy.stats import norm, spearmanr

from src.test3 import zeros_z

SEED = 20260926
NPERM = 20000
R = "results/test3/"
ALPHA = 0.05 / 3


def covariates(N):
    mp.mp.dps = 30
    zm = zeros_z(N)
    gam = [z / 2 for z in zm]
    lz = np.array([float(mp.log(abs(mp.zeta(mp.mpf(1) / 2 + 1j * g, derivative=1)))) for g in gam])
    lnln = np.log(np.log(np.array([float(g) for g in gam]) / (2 * math.pi)))
    b = np.polyfit(lnln, lz, 1)
    lz_res = lz - np.polyval(b, lnln)
    return gam, lz, lz_res, b


def gram_data(gmid):
    """u = theta/pi, deviation, margin for a list of mid-heights (mp)."""
    theta = lambda t: mp.siegeltheta(t) / mp.pi
    out = []
    zc = {}

    def zg(k):
        if k not in zc:
            zc[k] = ((-1) ** k) * mp.re(mp.siegelz(mp.grampoint(k)))
        return zc[k]

    for g in gmid:
        u = theta(g)
        k0 = int(mp.floor(u))
        dev = float(abs(u - mp.nint(u)))
        j0 = max(-1, k0 - 5)  # Gram points are defined for n >= -1; window of 12 shifted up near the start
        rms = math.sqrt(float(sum(zg(j) ** 2 for j in range(j0, j0 + 12)) / 12))
        margin = float(min(zg(k0), zg(k0 + 1))) / rms
        out.append((float(u), dev, margin, k0))
    return out


def gram_first_violation(nmax=200):
    mp.mp.dps = 30
    bad = [n for n in range(0, nmax + 1) if ((-1) ** n) * mp.siegelz(mp.grampoint(n)) <= 0]
    return bad


def load_sample():
    d = json.load(open(R + "geom.json"))
    rows = [r for r in d["rows"] if r["status"] == "collided" and not r["x_ge_1"]]
    return rows


def ols_resid(c, G):
    X = np.column_stack([np.ones(len(c))] + G)
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    fit = X @ beta
    ss = float(np.sum((c - c.mean()) ** 2))
    e = c - fit
    h = np.einsum("ij,ji->i", X, np.linalg.pinv(X))
    loo = e / (1 - h)
    return e, (1 - float(e @ e) / ss), (1 - float(loo @ loo) / ss)


def build(rows, N=114):
    gam, lz, lz_res, b = covariates(N)
    mids = [mp.mpf(0.5) * (gam[r["gap"] - 1] + gam[r["gap"]]) for r in rows]
    gd = gram_data(mids)
    for r, (u, dev, mar, k0) in zip(rows, gd):
        i = r["gap"] - 1
        r["c1"] = 0.5 * (lz_res[i] + lz_res[i + 1])
        r["c1_zeros"] = [lz_res[i], lz_res[i + 1]]
        r["c2"], r["c3"], r["gram_k0"] = dev, mar, k0
    return dict(trend_ln_zeta_prime=dict(slope=float(b[0]), intercept=float(b[1]), var="ln ln(gamma/2pi)"))


def geom_arrays(rows):
    dl = np.log([r["delta"] for r in rows])
    return dict(delta=dl, dL=np.log([r["dL_n"] for r in rows]), dR=np.log([r["dR_n"] for r in rows]),
                x=np.log([r["ratio_pred"] for r in rows]), y=np.log([r["ratio_obs"] for r in rows]))


def cmd_corr(a):
    rows = load_sample()
    info = build(rows)
    gv = gram_first_violation()
    ga = geom_arrays(rows)
    out: dict[str, Any] = dict(n=len(rows), trend=info, gram_law_violations_n_le_200=gv, candidates={})
    for c in ("c1", "c2", "c3"):
        v = np.array([r[c] for r in rows])
        cs: dict[str, Any] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), spearman={}, R2_geometry=None)
        for k, nm in (("delta", "ln_delta"), ("dL", "ln_dL"), ("dR", "ln_dR"), ("x", "x")):
            rho, p = cast(Any, spearmanr(v, ga[k]))
            cs["spearman"][nm] = dict(rho=float(rho), p=float(p))
        _, r2, r2loo = ols_resid(v, [ga["delta"], ga["dL"], ga["dR"]])
        cs["R2_geometry"] = dict(R2=r2, R2_loo=r2loo, predictors="ln delta, ln dL, ln dR")
        _, r2x, r2xl = ols_resid(v, [ga["delta"], ga["dL"], ga["dR"], ga["x"]])
        cs["R2_geometry_plus_x"] = dict(R2=r2x, R2_loo=r2xl)
        out["candidates"][c] = cs
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"n={out['n']}; ln|zeta'| trend slope {info['trend_ln_zeta_prime']['slope']:.3f}; "
          f"Gram-law violations n<=200: {gv[:6]}{'...' if len(gv) > 6 else ''} (count {len(gv)})")
    for c, cs in out["candidates"].items():
        s = cs["spearman"]
        print(f"{c}: rho(delta) {s['ln_delta']['rho']:+.2f} rho(dL) {s['ln_dL']['rho']:+.2f} rho(dR) {s['ln_dR']['rho']:+.2f} "
              f"rho(x) {s['x']['rho']:+.2f} | R2 on geometry {cs['R2_geometry']['R2']:.2f} (LOO {cs['R2_geometry']['R2_loo']:.2f}); "
              f"+x {cs['R2_geometry_plus_x']['R2']:.2f} (LOO {cs['R2_geometry_plus_x']['R2_loo']:.2f})")
    return 0


def detrend(x, y, kind):
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    if kind == "isotonic":
        fit = isotonic_regression(ys, increasing=True).x
    else:
        fit = make_smoothing_spline(xs, ys)(xs)
    res = np.empty_like(y)
    res[o] = ys - fit
    return res


def perm_p(r, c, rng):
    obs = cast(Any, spearmanr(r, c))[0]
    cnt = 0
    for _ in range(NPERM):
        if abs(cast(Any, spearmanr(r, rng.permutation(c)))[0]) >= abs(obs) - 1e-12:
            cnt += 1
    return float(obs), (cnt + 1) / (NPERM + 1)


def min_detectable(n, alpha=ALPHA, power=0.8):
    z = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) / math.sqrt(n - 3)
    return math.tanh(z)


def cmd_test(a):
    rows = load_sample()
    build(rows)
    d = json.load(open(R + "geom.json"))
    cautious = {r["gap"] for r in d["rows"] if r["status"] == "collided" and not r["neighbor_zero_removed_first"]
                and r["gap"] != 55}
    out: dict[str, Any] = dict(seed=SEED, n_perm=NPERM, alpha_bonferroni=ALPHA, sets={}, note="p is two-sided permutation p; primary = isotonic")
    for setname, rs in (("primary_36", rows), ("cautious_21", [r for r in rows if r["gap"] in cautious])):
        ga = geom_arrays(rs)
        so: dict[str, Any] = dict(n=len(rs), min_detectable_abs_rho_80pct=min_detectable(len(rs)), detrend={})
        G = [ga["delta"], ga["dL"], ga["dR"], ga["x"]]
        for kind in ("isotonic", "spline"):
            r = detrend(ga["x"], ga["y"], kind)
            trace = {}
            for k, nm in (("delta", "ln_delta"), ("dL", "ln_dL"), ("dR", "ln_dR")):
                rho, p = cast(Any, spearmanr(r, ga[k]))
                trace[nm] = dict(rho=float(rho), p=float(p))
            block: dict[str, Any] = dict(resid_sd=float(r.std(ddof=1)), resid_range=[float(r.min()), float(r.max())],
                         leftover_geometry_trace=trace, candidates={})
            rng = np.random.default_rng(SEED)
            for c in ("c1", "c2", "c3"):
                cv = np.array([q[c] for q in rs])
                cperp, _, _ = ols_resid(cv, G)
                rho, p = perm_p(r, cperp, rng)
                rho_raw = float(cast(Any, spearmanr(r, cv))[0])
                block["candidates"][c] = dict(rho_partial=rho, p_perm=p, rho_raw=rho_raw,
                                              significant_bonferroni=bool(p < ALPHA))
            so["detrend"][kind] = block
        out["sets"][setname] = so
    json.dump(out, open(a.out, "w"), indent=1)
    for sn, so in out["sets"].items():
        print(f"[{sn}] n={so['n']} min detectable |rho| (80% power, Bonferroni) {so['min_detectable_abs_rho_80pct']:.2f}")
        for kind, b in so["detrend"].items():
            t = b["leftover_geometry_trace"]
            print(f"  {kind}: resid sd {b['resid_sd']:.4f}; leftover rho with delta {t['ln_delta']['rho']:+.2f} (p {t['ln_delta']['p']:.2f}), "
                  f"dL {t['ln_dL']['rho']:+.2f}, dR {t['ln_dR']['rho']:+.2f}")
            print("     " + "; ".join(f"{c}: rho {v['rho_partial']:+.2f} p {v['p_perm']:.3f} (raw {v['rho_raw']:+.2f})"
                                       for c, v in b["candidates"].items()))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["corr", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    return cmd_corr(a) if a.stage == "corr" else cmd_test(a)


if __name__ == "__main__":
    raise SystemExit(main())
