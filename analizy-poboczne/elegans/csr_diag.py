# Uruchom: python csr_diag.py  (ziarno 7; N = 272: 300 realizacji, N = 1000: 60 realizacji)
# Diagnostyka (tylko symulacje): wplyw brzegow na <r>, <cos>. Punkty "bulk": |l| < 0.6 i Im l > 0.15, sasiedzi z calego zbioru.
import os; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import math, numpy as np
from scipy.spatial import cKDTree
rng = np.random.default_rng(7)
def stats(pts, mask):
    _, idx = cKDTree(np.c_[pts.real, pts.imag]).query(np.c_[pts.real, pts.imag], k=3)
    z = (pts[idx[:, 1]] - pts) / (pts[idx[:, 2]] - pts)
    z = z[mask]
    return np.abs(z).mean(), (z.real / np.abs(z)).mean()
for N in (272, 1000):
    for kind in ("ginibre", "poisson"):
        acc = {"upper_all": [], "full_plane_all": [], "bulk_only": []}
        for _ in range(300 if N == 272 else 60):
            if kind == "ginibre":
                ev = np.linalg.eigvals(rng.standard_normal((N, N)) / math.sqrt(N))
            else:
                ev = np.sqrt(rng.uniform(0, 1, N)) * np.exp(2j * math.pi * rng.uniform(0, 1, N))
            up = ev[ev.imag > 1e-8]
            acc["upper_all"].append(stats(up, np.ones(len(up), bool)))
            acc["full_plane_all"].append(stats(ev, np.ones(len(ev), bool)))
            acc["bulk_only"].append(stats(ev, (np.abs(ev) < 0.6) & (ev.imag > 0.15)))
        print(N, kind, "  ".join(f"{k}: r {np.mean([a[0] for a in v]):.3f} cos {np.mean([a[1] for a in v]):+.3f}" for k, v in acc.items()))
