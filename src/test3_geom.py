"""Test 3, geometry of collision times (plan: docs/PLAN_TEST3_B.md).

    python -m src.test3_geom --out results/test3/geom.json

Sample: gen-1 pairs (zeros i, i+1) of results/test3/gen3.json with both zeros <= n. Primary set = status
`collided`; extended set adds `preempted` (weaker evidence). `neighbor_preempted`: the left (L,a) or right (b,R)
neighbour pair has its own t_c with |t_c| smaller than the pair's, i.e. the neighbour leaves the real axis first
(the constant-tide approximation is then broken qualitatively).

Constant-tide predictor. From zdot_j = 2 sum_k 1/(z_j - z_k): d(Delta)/dt = 4/Delta + T, T = zdot_b - zdot_a - 4/Delta
(zdot = H_xx/H_x at the zeros, t = 0, exact). With x = -T Delta/4 < 1 the pair collides at
tau_c = Delta^2/(4x^2) (-x - ln(1-x)), i.e. t_c/t_c0 = 2(-x - ln(1-x))/x^2  [own derivation, checked numerically here].
Nearest-neighbour variant: T_nn = -2 Delta [1/(dL (Delta+dL)) + 1/(dR (Delta+dR))].
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.optimize import least_squares

from src.heatflow import HeatFlow
from src.test3 import DIGITS, zeros_z
from src.zeros import n_smooth

SEED = 20260926
NBOOT = 2000
R = "results/test3/"


def ratio_pred(x):
    if x >= 1.0:
        return math.inf
    if abs(x) < 1e-8:
        return 1.0 + 2 * x / 3
    return 2 * (-x - math.log1p(-x)) / (x * x)


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X]) if X is not None else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    h = np.einsum("ij,ji->i", X, np.linalg.pinv(X))
    loo = e / (1 - h)
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(beta=beta.tolist(), r2=1 - float(e @ e) / ss if ss > 0 else None,
                rmse=math.sqrt(float(e @ e) / len(y)), loo_rmse=math.sqrt(float(loo @ loo) / len(y)), n=len(y))


def huber_fit(x, y, scale=0.1):
    f = lambda p: (p[0] + p[1] * x) - y
    r = least_squares(f, [0.0, 1.0], loss="huber", f_scale=scale)
    return r.x.tolist()


def boot_slope(x, y, rng):
    n = len(y)
    out = []
    for _ in range(NBOOT):
        k = rng.integers(0, n, n)
        if np.ptp(x[k]) == 0:
            continue
        b = np.polyfit(x[k], y[k], 1)
        out.append(b[0])
    return np.quantile(out, [0.025, 0.975]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=R + "gen3.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.gen))
    n = d["n"]
    N = len(d["zeros"])
    zm = zeros_z(N)
    zf = [float(z) for z in zm]
    eng = HeatFlow(zf[-1] + 2, DIGITS, tmin=0.0)
    zdot = []
    for z in zm:
        h0, h1, h2 = eng.derivs(z, 0.0)
        zdot.append(float(h2 / h1))
    S = np.array(zf + [-z for z in zf])
    recs = {r["gap"]: r for r in d["gaps"]}
    ns = lambda z: float(n_smooth(z / 2.0))
    rho = lambda z: math.log(z / 2 / (2 * math.pi)) / (2 * math.pi) / 2.0  # zeros per unit z

    rows = []
    for g, r in sorted(recs.items()):
        if g >= n or r["status"] not in ("collided", "preempted"):
            continue
        i = g - 1
        a_, b_ = zf[i], zf[i + 1]
        D = b_ - a_
        T = zdot[i + 1] - zdot[i] - 4.0 / D
        m = (S != a_) & (S != b_)
        T_sum = float(np.sum(2.0 * (1.0 / (b_ - S[m]) - 1.0 / (a_ - S[m]))))
        dL = a_ - zf[i - 1] if i > 0 else 2 * a_
        dR = zf[i + 2] - b_
        dL_n = ns(a_) - ns(zf[i - 1]) if i > 0 else 2 * a_ * rho(a_)
        dR_n = ns(zf[i + 2]) - ns(b_)
        x = -T * D / 4.0
        x_nn = 0.5 * D * D * (1 / (dL * (D + dL)) + 1 / (dR * (D + dR)))
        nb = {}
        for side, gg in (("left", g - 1), ("right", g + 1)):
            q = recs.get(gg)
            nb[side] = None if q is None or "tc" not in q else dict(tc=q["tc"], status=q["status"])
        early = [s for s in nb if nb[s] is not None and abs(nb[s]["tc"]) < abs(r["tc"])]
        # generalisation: the neighbouring zero L (number g-1) or R (number g+2) took part in an accepted collision
        # (with its outer neighbour) at |t_c| smaller than the pair's -> it left the real axis first
        gen_early = [dict(zero=zz, event=[e["zi"], e["zj"]], tc=e["tc"]) for zz in (g - 1, g + 2) if zz >= 1
                     for e in d["events"] if zz in (e["zi"], e["zj"]) and abs(e["tc"]) < abs(r["tc"])
                     and (e["zi"], e["zj"]) != (g, g + 1)]
        rows.append(dict(gap=g, status=r["status"], z_lo=a_, z_hi=b_, dz=D, delta=ns(b_) - ns(a_), dL_n=dL_n, dR_n=dR_n,
                         T=T, T_trunc_sum=T_sum, x=x, x_nn=x_nn, tc=r["tc"], tc0=r["tc0"], ratio_obs=r["ratio"],
                         ratio_pred=ratio_pred(x), ratio_pred_nn=ratio_pred(x_nn), neighbors=nb,
                         neighbor_preempted=bool(early), neighbor_preempted_sides=early,
                         neighbor_zero_removed_first=bool(gen_early), neighbor_zero_events=gen_early,
                         x_ge_1=bool(x >= 1.0)))
    prim = [r for r in rows if r["status"] == "collided"]
    ext = rows
    cautious = [r for r in prim if not r["neighbor_preempted"] and not r["neighbor_zero_removed_first"]
                and r["gap"] != 55]

    def arr(rs, k):
        return np.array([r[k] for r in rs], dtype=float)

    def analyse(rs, tag):
        yall = np.log(arr(rs, "ratio_obs"))
        valid = np.isfinite(arr(rs, "ratio_pred")) & np.isfinite(arr(rs, "ratio_pred_nn"))
        res = dict(n=len(rs), n_valid=int(valid.sum()), n_pred_inf=int((~np.isfinite(arr(rs, "ratio_pred"))).sum()),
                   n_pred_nn_inf=int((~np.isfinite(arr(rs, "ratio_pred_nn"))).sum()),
                   T_negative=int((arr(rs, "T") < 0).sum()),
                   x_ge_1_gaps=[r["gap"] for r in rs if r["x_ge_1"]])
        res["full_M0_const"] = ols(None, yall)
        res["full_M1_ln_delta"] = ols(np.log(arr(rs, "delta"))[:, None], yall)
        res["full_M2_ln_delta_dL_dR"] = ols(np.column_stack([np.log(arr(rs, "delta")), np.log(arr(rs, "dL_n")),
                                                             np.log(arr(rs, "dR_n"))]), yall)
        # every model on the same subset (both predictors finite)
        rv = [r for r, v in zip(rs, valid) if v]
        y = yall[valid]
        res["M0_const"] = ols(None, y)
        res["M1_ln_delta"] = ols(np.log(arr(rv, "delta"))[:, None], y)
        res["M2_ln_delta_dL_dR"] = ols(np.column_stack([np.log(arr(rv, "delta")), np.log(arr(rv, "dL_n")),
                                                        np.log(arr(rv, "dR_n"))]), y)
        rng = np.random.default_rng(SEED)
        for key, col in (("M3_x_nn", "ratio_pred_nn"), ("M4_x_exact", "ratio_pred")):
            if valid.sum() < 4:
                res[key] = None
                continue
            lx = np.log(arr(rv, col))
            f = ols(lx[:, None], y)
            f["beta_ci95"] = boot_slope(lx, y, rng)
            f["huber"] = huber_fit(lx, y)
            r0 = y - lx
            f["param_free"] = dict(mean_resid=float(r0.mean()), rmse=math.sqrt(float(r0 @ r0) / len(r0)),
                                   sd_resid=float(r0.std(ddof=1)), n=int(valid.sum()))
            res[key] = f
        return res

    out = dict(seed=SEED, n_boot=NBOOT, n_zeros=N, sets=dict(primary=len(prim), extended=len(ext), cautious=len(cautious)),
               analyses={"primary": analyse(prim, "primary"), "cautious": analyse(cautious, "cautious"),
                         "extended": analyse(ext, "extended")})
    out["neighbor_zero_removed_first"] = [dict(gap=r["gap"], status=r["status"], events=r["neighbor_zero_events"],
                                               ratio_obs=r["ratio_obs"], ratio_pred=r["ratio_pred"])
                                          for r in ext if r["neighbor_zero_removed_first"]]
    out["x_ge_1"] = [dict(gap=r["gap"], status=r["status"], x=r["x"], x_nn=r["x_nn"], ratio_obs=r["ratio_obs"], tc=r["tc"])
                     for r in ext if r["x_ge_1"]]
    out["neighbor_preempted"] = [dict(gap=r["gap"], sides=r["neighbor_preempted_sides"], ratio_obs=r["ratio_obs"],
                                      ratio_pred=r["ratio_pred"]) for r in ext if r["neighbor_preempted"]]
    out["T_exact_vs_truncated_sum_max_abs_diff"] = max(abs(r["T"] - r["T_trunc_sum"]) for r in ext)
    out["ratio_ge_1_and_T_negative"] = dict(n=len(prim), n_ratio_ge_1=sum(1 for r in prim if r["ratio_obs"] >= 1),
                                            n_T_neg=sum(1 for r in prim if r["T"] < 0))
    out["rows"] = rows
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    figure(rows, R + "fig_geom.png")
    report(out)


def figure(rows, path):
    blue, orange, violet = "#2a78d6", "#eb6834", "#4a3aa7"  # validated (validate_palette.js): CVD dE >= 24.7
    ink, grid = "#1a1a19", "#e1e0d9"
    STRIP = 9.0  # x position of pairs with x >= 1 (constant-tide model predicts no collision)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor="#fcfcfb")
    for a_ in ax:
        a_.set_facecolor("#fcfcfb")
        a_.grid(color=grid, lw=0.6)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(colors=ink)
    sel = {"c": lambda r: r["status"] == "collided" and not r["neighbor_zero_removed_first"],
           "n": lambda r: r["status"] == "collided" and r["neighbor_zero_removed_first"],
           "p": lambda r: r["status"] == "preempted"}
    groups = [("c", "zderzone; sąsiednie zera do t_c nie znikają", blue, "o", False),
              ("n", "zderzone; sąsiednie zero zniknęło wcześniej", orange, "^", False),
              ("p", "preempted (słabsze potwierdzenie)", violet, "D", True)]
    for key, lab, col, mk, hollow in groups:
        rs = [r for r in rows if sel[key](r)]
        fin = [r for r in rs if np.isfinite(r["ratio_pred"])]
        inf = [r for r in rs if not np.isfinite(r["ratio_pred"])]
        kw = dict(s=40, marker=mk, facecolors="none" if hollow else col, edgecolors=col, lw=1.3, zorder=3)
        ax[0].scatter([r["ratio_pred"] for r in fin], [r["ratio_obs"] for r in fin], label=f"{lab} (n={len(rs)})", **kw)
        if inf:
            ax[0].scatter([STRIP] * len(inf), [r["ratio_obs"] for r in inf], **kw)
        ax[1].scatter([r["delta"] for r in fin],
                      [np.log(r["ratio_obs"]) - np.log(r["ratio_pred"]) for r in fin], **kw)
    ax[0].plot([0.9, 5], [0.9, 5], color=ink, lw=1.0, ls="--", zorder=2)
    ax[0].text(3.3, 2.6, "y = x", color=ink, fontsize=9)
    ax[0].axvline(6.3, color=ink, lw=0.8, ls=":")
    ax[0].text(STRIP, 1.0, "x ≥ 1\n(model: brak\nzderzenia)", color=ink, fontsize=8, ha="center", va="center")
    r55 = [r for r in rows if r["gap"] == 55]
    if r55:
        ax[0].annotate("przerwa 55", (STRIP, r55[0]["ratio_obs"]), textcoords="offset points", xytext=(-62, -3),
                       fontsize=8, color=ink)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(0.9, 13); ax[0].set_ylim(0.85, 30)
    ax[0].set_xticks([1, 2, 4]); ax[0].set_xticklabels(["1", "2", "4"])
    ax[0].set_yticks([1, 2, 4, 8, 16]); ax[0].set_yticklabels(["1", "2", "4", "8", "16"])
    ax[0].set_xlabel("przewidywany t_c/t_c⁰ (stałe T, dokładne)", color=ink)
    ax[0].set_ylabel("obserwowany t_c/t_c⁰", color=ink)
    ax[0].set_title("Iloraz obserwowany a przewidywany", loc="left", color=ink, fontsize=11)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[1].axhline(0, color=ink, lw=1.0, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.4, 0.6, 0.8, 1.0]); ax[1].set_xticklabels(["0,4", "0,6", "0,8", "1,0"])
    ax[1].set_xlabel("znormalizowana odległość pary δ (średnie odstępy)", color=ink)
    ax[1].set_ylabel("reszta ln(obs) − ln(przewid.)", color=ink)
    ax[1].set_title("Reszta bez dopasowania a δ (tylko x < 1)", loc="left", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(o):
    print(f"sets {o['sets']}; T exact vs truncated sum: max |diff| {o['T_exact_vs_truncated_sum_max_abs_diff']:.3e}")
    c = o["ratio_ge_1_and_T_negative"]
    print(f"primary: ratio>=1 {c['n_ratio_ge_1']}/{c['n']}, T<0 {c['n_T_neg']}/{c['n']}")
    print("neighbor_preempted (literal):", [(x["gap"], x["sides"]) for x in o["neighbor_preempted"]])
    print("neighbour zero removed first (general):", [(x["gap"], x["status"][:4]) for x in o["neighbor_zero_removed_first"]])
    print("x>=1:", [(x["gap"], x["status"][:4], round(x["x"], 2), round(x["ratio_obs"], 2)) for x in o["x_ge_1"]])
    for k in ("primary", "cautious", "extended"):
        A = o["analyses"][k]
        print(f"[{k}] n={A['n']} valid={A['n_valid']} pred_inf={A['n_pred_inf']} nn_inf={A['n_pred_nn_inf']} | "
              f"LOO RMSE on valid: " + " ".join(f"{m}={A[m]['loo_rmse']:.3f}" for m in ("M0_const", "M1_ln_delta", "M2_ln_delta_dL_dR")))
        for m in ("M3_x_nn", "M4_x_exact"):
            f = A[m]
            if f:
                pf = f["param_free"]
                print(f"   {m}: alpha {f['beta'][0]:+.3f} beta {f['beta'][1]:.3f} CI {f['beta_ci95'][0]:.2f}..{f['beta_ci95'][1]:.2f} "
                      f"R2 {f['r2']:.3f} LOO {f['loo_rmse']:.3f} huber b {f['huber'][1]:.2f} | param-free mean {pf['mean_resid']:+.3f} "
                      f"rmse {pf['rmse']:.3f}")


if __name__ == "__main__":
    main()
