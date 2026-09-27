"""Test 3, reanaliza: czy N_eff(wysokosc) wiaze sie z |T| albo z jakoscia dopasowania (results/test3/REPORT_B.md).

    python -m src.test3_neff_check --out results/test3/neff_check.json

Czysta reanaliza results/test3/geom.json (bez ponownego liczenia H_t). Proba: te same 36 par pierwotnych
(status "collided", x < 1) co w src/test3_fluctdiss.py. Wysokosc gamma pary = (z_lo+z_hi)/4, bo z_lo/z_hi w
geom.json to z = 2*gamma (zmienna heat-flow). N_eff(gamma) = ln(gamma/2pi)/sqrt(12*Lambda), Lambda = 1.57314,
ta sama definicja i stala co w src/test1.py (Bogomolny i in. 2006).

Motywacja: N_eff rosnie z wysokoscia (Test 2). Pytanie: czy w tej probie (pierwsze ~114 zer, gdzie N_eff
zmienia sie niemal wcale) widac zaleznosc |T| albo |reszta| od N_eff? Oczekiwanie z gory: N_eff w tym zakresie
jest niemal stale, wiec zaleznosci raczej nie bedzie widac - to nie jest test hipotezy N_eff w ogole, tylko
sprawdzenie, czy dostepna, tania proba cokolwiek o niej mowi.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr

LAM = 1.57314
R = "results/test3/"


def n_eff(gamma):
    return math.log(gamma / (2 * math.pi)) / math.sqrt(12 * LAM)


def cmd_run(a):
    geom = json.load(open(os.path.join(R, "geom.json")))
    rows = [r for r in geom["rows"]
            if r["status"] == "collided" and r.get("ratio_pred") is not None and math.isfinite(r["ratio_pred"])]
    n = len(rows)
    if n != 36:
        raise SystemExit(f"oczekiwano 36 par pierwotnych z x<1, znaleziono {n}")

    gamma_mid = np.array([(r["z_lo"] + r["z_hi"]) / 4 for r in rows])
    neff = np.array([n_eff(g) for g in gamma_mid])
    T = np.array([r["T"] for r in rows])
    resid = np.array([math.log(r["ratio_obs"]) - math.log(r["ratio_pred"]) for r in rows])
    gaps = [r["gap"] for r in rows]

    rho_T, p_T = spearmanr(neff, np.abs(T))
    rho_R, p_R = spearmanr(neff, np.abs(resid))

    out = {
        "n": n, "gaps": gaps,
        "gamma_mid_range": [float(gamma_mid.min()), float(gamma_mid.max())],
        "N_eff_range": [float(neff.min()), float(neff.max())],
        "N_eff_span_over_range": float((neff.max() - neff.min()) / (neff.max())),
        "spearman_Neff_absT": {"rho": float(rho_T), "p": float(p_T)},
        "spearman_Neff_absresid": {"rho": float(rho_R), "p": float(p_R)},
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"n={n}; gamma_mid in [{gamma_mid.min():.2f}, {gamma_mid.max():.2f}]")
    print(f"N_eff in [{neff.min():.4f}, {neff.max():.4f}] (rozpietosc {out['N_eff_span_over_range']*100:.1f}% zakresu)")
    print(f"Spearman N_eff vs |T|:     rho={rho_T:.4f} p={p_T:.4f}")
    print(f"Spearman N_eff vs |resid|: rho={rho_R:.4f} p={p_R:.4f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
