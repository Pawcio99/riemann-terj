"""Test 2 summary: per-block N-hat/N_eff, variance z-scores vs CUE(N_eff), trend in 1/rho-bar.

    python -m src.test2_summary            # reads results/test2/{fits.json,cue_table.npz}

rho-bar = ln(T/2pi)/(2pi) is the mean density of zeros. x = 1/rho-bar. N-hat above the table range
(N > 128, i.e. u < 1/128^2) is reported as a lower bound. Fixed seeds throughout.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import stats
from src.test1 import cue_stream, neff
from src.test2 import SG, Family, n_of

R = "results/test2/"
NMAX = 128
NREP = 1000
CHAT = [8.7, 9.8, 8.7, 5.4, 3.3, 1.8, 1.0, 0.6, -0.7]  # control values quoted by the user (chat), for comparison only


def var_family(fam, u):
    """Variance of the spacing law F(.; u) on the grid s in [0, 4] (mean is checked separately)."""
    f = fam(min(u, 0.25))[0]  # N < 2 is outside the table: clipped to CUE(2), flagged by the caller
    m1 = np.trapezoid(1 - f, SG)
    m2 = np.trapezoid(2 * SG * (1 - f), SG)
    return m2 - m1**2, m1


def lsq(x, y, sd, powers):
    a = np.stack([x**p for p in powers], axis=1)
    w = 1 / sd**2
    c = np.linalg.solve(a.T @ (w[:, None] * a), a.T @ (w * y))
    return c, float(np.sum(w * (y - a @ c) ** 2))


def fmt_n(q):
    return f">{NMAX}" if q > NMAX else f"{q:.2f}"


def main():
    fits = json.load(open(R + "fits.json"))
    cal = json.load(open(R + "calib.json"))
    fam = Family(R + "cue_table.npz", 0)
    blocks = fits["blocks"]
    rows = []
    for i, b in enumerate(blocks):
        base, x, _ = stats.load(f"data/{b['file']}")
        s = stats.unfold(base, x)
        ne = float(neff(b["T"]))
        rng = np.random.default_rng(9000 + i)
        v_null = np.array([np.var(cue_stream(ne, len(s), rng), ddof=1) for _ in range(NREP)])
        vd = float(np.var(s, ddof=1))
        vf, m1 = var_family(fam, 1 / ne**2)
        r = {"file": b["file"], "T": b["T"], "N_eff": ne, "neff_below_table": bool(ne < 2), "x": float(2 * np.pi / b["L"]), "var": vd,
             "var_null_mix": float(v_null.mean()), "var_null_sd": float(v_null.std()), "z_var_mix": float((vd - v_null.mean()) / v_null.std()),
             "var_family": float(vf), "mean_family": float(m1), "z_var_fam": float((vd - vf) / v_null.std())}
        for tag in ("dens", "loc"):
            d = b[tag]
            ub = np.array(d["ub"])
            nb = np.array([min(n_of(u), 1e6) for u in ub])
            q =np.quantile(nb, [0.16, 0.5, 0.84])
            near = min((2, 3, 4, 6, 8, 11, 16), key=lambda n: abs(n - min(d["N"], 16)))  # calibration N nearest to N-hat
            refks = np.array(cal[f"{tag}_{near}"]["ks"])
            d["p_gof"] = float((1 + np.sum(refks >= d["ks_min"])) / (len(refks) + 1))
            r[tag] = {"gof_ref_N": near, "ks_min": d["ks_min"],"N_hat": min(d["N"], np.inf), "N_hat_is_lower_bound": bool(d["N"] > NMAX), "N_q16_50_84": [fmt_n(v) for v in q],
                      "ratio_q16_50_84": [float(v / ne) for v in q], "u_hat": d["u"], "du_mean": float(np.mean(ub - 1 / ne**2)),
                      "du_sd": float(np.std(ub - 1 / ne**2)), "p_gof": d["p_gof"], "censored": d["censored"]}
        rows.append(r)
    x = np.array([r["x"] for r in rows])
    out = {"rows": rows, "control_chat_z_var": CHAT}
    # SD of the variance at n ~ 1e4 and resolution of N >= 8 versus N = infinity
    vinf = var_family(fam, 1 / NMAX**2)[0]
    out["var_resolution"] = {"sd_var_null_range": [min(r["var_null_sd"] for r in rows), max(r["var_null_sd"] for r in rows)],
                             "var_family_N": {str(n): float(var_family(fam, 1 / n**2)[0]) for n in (2, 3, 4, 6, 8, 11, 16, 40, NMAX)},
                             "var_inf_proxy_N128": float(vinf)}
    # trend / extrapolation x -> 0: (a) z-score of variance, (b) du = u_hat - 1/N_eff^2 (bootstrap draws)
    rng = np.random.default_rng(77)
    mods = {"linear": [0, 1], "quad": [0, 1, 2], "quad_through0": [1, 2], "linear_through0": [1]}
    ex = {}
    zs = np.array([r["z_var_fam"] for r in rows])
    for name, pw in mods.items():
        c, chi = lsq(x, zs, np.ones_like(zs), pw)
        ex[f"zvar_{name}"] = {"coef": c.tolist(), "chi2": chi, "dof": len(x) - len(pw)}
    for tag in ("dens", "loc"):
        ubs = np.array([b[tag]["ub"] for b in blocks])  # (blocks, B)
        ueff = np.array([1 / r["N_eff"] ** 2 for r in rows])
        du = ubs - ueff[:, None]
        sd = du.std(axis=1)
        for name, pw in mods.items():
            cb = np.array([lsq(x, du[:, j], sd, pw)[0] for j in range(du.shape[1])])
            c, chi = lsq(x, du.mean(axis=1), sd, pw)
            ex[f"du_{tag}_{name}"] = {"coef": c.tolist(), "coef_sd": cb.std(axis=0).tolist(), "chi2": chi, "dof": len(x) - len(pw)}
    # extrapolation of z: intercept uncertainty from resampling z with unit variance
    zb = zs[None, :] + rng.normal(size=(4000, len(zs)))
    for name in ("linear", "quad"):
        cb = np.array([lsq(x, z, np.ones_like(z), mods[name])[0] for z in zb])
        ex[f"zvar_{name}"]["coef_sd"] = cb.std(axis=0).tolist()
    out["extrapolation"] = ex
    json.dump(out, open(R + "summary.json", "w"), indent=1)
    for b, r in zip(blocks, out["rows"]):  # p_gof w fits.json przeliczone przy N-hat (nie przy N_eff)
        for tag in ("dens", "loc"):
            b[tag]["gof_ref_N"] = r[tag]["gof_ref_N"]
    json.dump(fits, open(R + "fits.json", "w"))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for tag, mk in (("dens", "o"), ("loc", "s")):
        y = np.array([r[tag]["ratio_q16_50_84"] for r in rows])
        ax[0].errorbar(x + (0.004 if tag == "loc" else 0), y[:, 1], yerr=[y[:, 1] - y[:, 0], y[:, 2] - y[:, 1]], fmt=mk, label=tag, capsize=2)
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].set(xlabel="1/ρ̄", ylabel="N̂ / N_eff", yscale="log", title="N̂/N_eff (68%)")
    ax[0].legend()
    ax[1].plot(x, zs, "o", label="z (family)")
    ax[1].plot(x, [r["z_var_mix"] for r in rows], "x", label="z (mixture)")
    ax[1].plot(x, CHAT, "+", label="control (chat)")
    xx = np.linspace(0, x.max(), 50)
    c = ex["zvar_linear"]["coef"]
    ax[1].plot(xx, c[0] + c[1] * xx, "--", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="1/ρ̄", ylabel="z (variance vs CUE(N_eff))", title="Variance z-score")
    ax[1].legend()
    L = np.array([b["L"] for b in blocks])
    y = np.array([r["dens"]["ratio_q16_50_84"] for r in rows]) * np.array([r["N_eff"] for r in rows])[:, None]
    ax[2].errorbar(L, y[:, 1], yerr=[y[:, 1] - y[:, 0], y[:, 2] - y[:, 1]], fmt="o", capsize=2)
    ax[2].plot(L, np.array([r["N_eff"] for r in rows]), "k-", label="N_eff")
    ax[2].axhline(NMAX, color="gray", ls=":")
    ax[2].set(xlabel="ln(T/2π)", ylabel="N̂", yscale="log", ylim=(1, 300), title="N̂ vs ln(T/2π) (dens)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(R + "fig_trend_vs_invrho.png", dpi=110)

    # console summary (<= 30 lines)
    print("block            Neff   1/rho | N-hat dens [68%]         ratio | z_var(fam) z(mix) chat | pGOF")
    for r, cz in zip(rows, CHAT):
        d = r["dens"]
        nh = f">{NMAX}" if d["N_hat_is_lower_bound"] else f"{d['N_hat']:.2f}"
        print(f"{r['file'][:15]:15s} {r['N_eff']:5.2f}{'*' if r['neff_below_table'] else ' '}{r['x']:6.3f} | {nh:>6s} [{d['N_q16_50_84'][0]},{d['N_q16_50_84'][2]}] "
              f"{d['ratio_q16_50_84'][1]:5.2f} | {r['z_var_fam']:+6.1f} {r['z_var_mix']:+6.1f} {cz:+5.1f} | {d['p_gof']:.2f}")
    v = out["var_resolution"]
    print(f"SD(var) null: {v['sd_var_null_range'][0]:.4f}-{v['sd_var_null_range'][1]:.4f}; var(N): " +
          ", ".join(f"{n}:{val:.4f}" for n, val in v["var_family_N"].items()))
    for k, e in ex.items():
        sdtxt = f" sd={np.round(e['coef_sd'], 4).tolist()}" if "coef_sd" in e else ""
        print(f"{k}: coef={np.round(e['coef'], 4).tolist()}{sdtxt} chi2={e['chi2']:.1f}/{e['dof']}")


if __name__ == "__main__":
    main()
