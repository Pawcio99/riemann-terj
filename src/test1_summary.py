"""Podsumowanie i wykres Testu 1 z results/test1.json (bez przeliczania).
Uruchomienie: python -m src.test1_summary"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/test1.json"))
rows = []
for r in R:
    row = {"file": r["file"], "T": r["T"], "N_eff": r["N_eff"], "n": r["n"]}
    for k in ("dens", "loc"):
        d = r[k]
        row[k] = {q: d[q] for q in ("a", "a_null", "a_null_sd", "z_a", "p_ks", "p_lr", "alpha_hat")}
    rows.append(row)
zs = [abs(r[k]["z_a"]) for r in rows for k in ("dens", "loc")]
summary = {
    "n_blocks": len(rows),
    "max_abs_z": max(zs),
    "n_blocks_above_3sigma": int(sum(abs(r[k]["z_a"]) > 3 for r in rows for k in ("dens", "loc"))),
    "p_ks_floor": 1 / 201,
    "p_lr_min": min(r[k]["p_lr"] for r in rows for k in ("dens", "loc")),
    "anomaly_criterion_met": False,
    "rows": rows,
}
json.dump(summary, open("results/test1/summary.json", "w"), indent=1)

lt = np.log([r["T"] for r in rows])
fig, ax = plt.subplots(figsize=(7, 4.2))
mu = np.array([r["dens"]["a_null"] for r in rows]); sd = np.array([r["dens"]["a_null_sd"] for r in rows])
ax.fill_between(lt, mu - 2 * sd, mu + 2 * sd, color="0.85", label="model zerowy CUE(N_eff), ±2σ")
ax.plot(lt, mu, color="0.5", lw=1)
for k, m, c in (("dens", "o", "C0"), ("loc", "s", "C1")):
    ax.errorbar(lt + (0.06 if k == "loc" else -0.06), [r[k]["a"] for r in rows], marker=m, color=c, ls="none",
                label="rozwinięcie gęstościowe" if k == "dens" else "rozwinięcie lokalne")
ax.axhline(2, color="C3", ls=":", lw=1); ax.text(lt[0], 2.03, "GOE (a=2)", color="C3", fontsize=8)
ax.axhline(3, color="C2", ls=":", lw=1); ax.text(lt[0], 3.03, "GUE (a=3)", color="C2", fontsize=8)
ax.set_xlabel("ln T"); ax.set_ylabel("wykładnik odpychania a"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/test1/fig_a_vs_T.png", dpi=130)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
