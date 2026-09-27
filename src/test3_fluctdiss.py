"""Test 3, reanaliza: hipoteza fluktuacyjno-dyssypacyjna dla par T-zderzeń (results/test3/REPORT_B.md).

    python -m src.test3_fluctdiss --out results/test3/fluctdiss.json

Czysta reanaliza istniejących liczb z results/test3/geom.json (bez ponownego liczenia H_t). Próba: 36 par
pierwotnych (status "collided", x < 1) z podrozdziału "Wyniki liczbowe" w REPORT_B.md - te same 36 par, którym
odpowiada geom.json:analyses.primary.M4_x_exact.param_free (kontrola zgodności w cmd_run poniżej).

Hipoteza [HIPOTEZA, do sprawdzenia]: jeśli T pełni rolę tłumienia w mechanizmie zbliżonym do
fluktuacyjno-dyssypacyjnego, silniejsze tłumienie pływowe (większe |T|) powinno wiązać się z mniejszym
rozrzutem reszt ln(obs) - ln(przewid.) modelu stałego T (bez dopasowania, beta=1, alpha=0), analogicznie do
tego, jak silniejsze tłumienie w układzie fizycznym redukuje amplitudę fluktuacji wokół stanu równowagi.

Statystyki: korelacja Spearmana |reszta| vs |T|; podział po medianie |T| na dwie grupy po 18 par, test F i
bootstrap (ziarno SEED, NBOOT replik) dla ilorazu wariancji reszt slabe/silne; test permutacyjny (NPERM
przetasowań) dla różnicy wariancji z hipotezą kierunkową "slabe > silne"; korelacja czastkowa Spearmana
|T|-|reszta| z kontrola delta (znanego juz w REPORT_B skorelowanego czynnika), zeby odróznic ewentualny nowy
efekt od juz opisanej zaleznosci reszty od delta.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import rankdata, spearmanr

SEED = 20260927
NBOOT = 20000
NPERM = 20000
R = "results/test3/"


def primary_valid_rows(geom):
    return [r for r in geom["rows"]
            if r["status"] == "collided" and r.get("ratio_pred") is not None and math.isfinite(r["ratio_pred"])]


def partial_spearman(a, b, c):
    """Korelacja Spearmana miedzy a i b po liniowym usunieciu (na rangach) zaleznosci od c."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)

    def resid_lin(y, x):
        A = np.vstack([x, np.ones_like(x)]).T
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta

    ea, eb = resid_lin(ra, rc), resid_lin(rb, rc)
    return float(np.corrcoef(ea, eb)[0, 1])


def cmd_run(a):
    geom = json.load(open(os.path.join(R, "geom.json")))
    rows = primary_valid_rows(geom)
    n = len(rows)
    if n != 36:
        raise SystemExit(f"oczekiwano 36 par pierwotnych z x<1, znaleziono {n}")

    T = np.array([r["T"] for r in rows])
    delta = np.array([r["delta"] for r in rows])
    x = np.array([r["x"] for r in rows])
    resid = np.array([math.log(r["ratio_obs"]) - math.log(r["ratio_pred"]) for r in rows])
    gaps = [r["gap"] for r in rows]

    # kontrola zgodnosci z geom.json: ten sam model stalego T bez dopasowania jak w M4_x_exact.param_free
    ref = geom["analyses"]["primary"]["M4_x_exact"]["param_free"]
    check = {
        "mean_resid": float(resid.mean()), "mean_resid_ref": ref["mean_resid"],
        "rmse": float(math.sqrt((resid ** 2).mean())), "rmse_ref": ref["rmse"],
        "sd_resid": float(resid.std(ddof=1)), "sd_resid_ref": ref["sd_resid"],
    }
    for k in ("mean_resid", "rmse", "sd_resid"):
        if abs(check[k] - check[f"{k}_ref"]) > 1e-6:
            raise SystemExit(f"niezgodnosc z geom.json: {k}={check[k]} vs ref={check[f'{k}_ref']}")

    absT, absR = np.abs(T), np.abs(resid)
    rho, p_rho = spearmanr(absT, absR)

    med = float(np.median(absT))
    order = np.argsort(absT)
    weak_idx, strong_idx = order[:18], order[18:]
    r_weak, r_strong = resid[weak_idx], resid[strong_idx]
    var_weak, var_strong = float(r_weak.var(ddof=1)), float(r_strong.var(ddof=1))

    F = var_weak / var_strong
    df1, df2 = len(r_weak) - 1, len(r_strong) - 1
    p_F_one = float(f_dist.sf(F, df1, df2))  # H1 kierunkowa: var(slabe) > var(silne)
    p_F_two = float(2 * min(p_F_one, 1 - p_F_one))

    rng = np.random.default_rng(SEED)
    boot_ratio = np.empty(NBOOT)
    for i in range(NBOOT):
        bw = rng.choice(r_weak, size=len(r_weak), replace=True)
        bs = rng.choice(r_strong, size=len(r_strong), replace=True)
        boot_ratio[i] = bw.var(ddof=1) / bs.var(ddof=1)
    ci_lo, ci_hi = (float(v) for v in np.percentile(boot_ratio, [2.5, 97.5]))
    p_boot_le1 = float((boot_ratio <= 1).mean())

    obs_diff = var_weak - var_strong
    all_r = np.concatenate([r_weak, r_strong])
    n1 = len(r_weak)
    cnt = 0
    for i in range(NPERM):
        perm = rng.permutation(all_r)
        d = perm[:n1].var(ddof=1) - perm[n1:].var(ddof=1)
        if d >= obs_diff:
            cnt += 1
    p_perm = (cnt + 1) / (NPERM + 1)

    rho_T_delta, p_T_delta = spearmanr(absT, delta)
    rho_T_x, p_T_x = spearmanr(absT, x)
    rho_R_delta, p_R_delta = spearmanr(delta, resid)
    partial_rho = partial_spearman(absT, absR, delta)

    out = {
        "seed": SEED, "n_boot": NBOOT, "n_perm": NPERM, "n": n, "gaps": gaps,
        "check_vs_geom_M4_param_free": check,
        "spearman_absT_absresid": {"rho": float(rho), "p": float(p_rho)},
        "median_split": {
            "median_absT": med,
            "weak": {"n": len(r_weak), "gaps": [gaps[i] for i in weak_idx], "var_resid": var_weak,
                     "absT_range": [float(absT[weak_idx].min()), float(absT[weak_idx].max())]},
            "strong": {"n": len(r_strong), "gaps": [gaps[i] for i in strong_idx], "var_resid": var_strong,
                       "absT_range": [float(absT[strong_idx].min()), float(absT[strong_idx].max())]},
            "F_weak_over_strong": float(F), "df1": df1, "df2": df2,
            "p_F_one_sided_weak_gt_strong": p_F_one, "p_F_two_sided": p_F_two,
            "bootstrap_ci95_ratio_weak_over_strong": [ci_lo, ci_hi],
            "p_bootstrap_ratio_le_1": p_boot_le1,
            "p_permutation_one_sided_weak_gt_strong": p_perm,
        },
        "confound_check_vs_delta": {
            "spearman_absT_delta": {"rho": float(rho_T_delta), "p": float(p_T_delta)},
            "spearman_absT_x": {"rho": float(rho_T_x), "p": float(p_T_x)},
            "spearman_resid_delta_ref_report_B": {"rho": float(rho_R_delta), "p": float(p_R_delta)},
            "partial_spearman_absT_absresid_given_delta": partial_rho,
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"n={n}; kontrola vs geom.json OK (mean_resid {check['mean_resid']:.6f}, rmse {check['rmse']:.6f})")
    print(f"Spearman |resid| vs |T|: rho={rho:.4f} p={p_rho:.4f}")
    print(f"mediana |T|={med:.4f}; var(resid) slabe={var_weak:.5f} silne={var_strong:.5f} "
          f"(F={F:.4f}, p_dwustronne={p_F_two:.4f}, p_perm(slabe>silne)={p_perm:.4f})")
    print(f"bootstrap 95% CI ilorazu wariancji slabe/silne: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"korelacja czastkowa |T|-|resid| po kontroli delta: {partial_rho:.4f} "
          f"(surowa {rho:.4f}; |T| vs delta rho={rho_T_delta:.4f})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
