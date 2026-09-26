"""Test 2: continuous effective dimension N-hat of CUE(N) fitted to the zero spacings.

    python -m src.test2 table  --out results/test2/cue_table.npz [--ns 2000000]
    python -m src.test2 check  --table results/test2/cue_table.npz --out results/test2/interp_check.json
    python -m src.test2 calib  --table ... --out results/test2/calib.json [--nrep 300]
    python -m src.test2 fit    --table ... --calib ... --out results/test2/fits.json [--nboot 1000]

CDFs of CUE(N) spacings for integer N = 2..40 (+ anchors 48..128) are tabulated on a grid in s and
interpolated (natural cubic spline) in u = 1/N^2 -- one smooth family instead of a mixture of
neighbouring N. Beyond the largest node (u < u_min) the CDF is extrapolated linearly in u.
N-hat = argmin_u KS(ECDF, F(.;u)); N-hat <= 2 (u-hat on the edge u = 1/4) is reported as censored.
Prediction (Bogomolny et al. 2006): N_eff = ln(T/2pi)/sqrt(12*Lambda).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import CubicSpline

from src import stats
from src.test1 import EDGES, LAM, W, cue_batch, local_norm, neff

NODES = list(range(2, 41)) + [48, 64, 96, 128]
SG = np.round(np.arange(0, 4.0001, 0.005), 6)
NBLK = 50  # moving-block length for the bootstrap
FILES = [f"data/z_1e{e}_10k.npz" for e in range(3, 9)] + [f"data/odl_1e{e}.npz" for e in (12, 21, 22)]
EDGE_IDX = np.array([int(round(e / 0.005)) if np.isfinite(e) else -1 for e in EDGES])


def stream(n, nsp, rng):
    """nsp spacings from CUE(n), independent matrices, shuffled."""
    m_chunk = max(1, int(4e6 // (n * n)))
    parts, got = [], 0
    while got < nsp:
        m = min(m_chunk, -(-(nsp - got) // n))
        parts.append(cue_batch(n, m, rng))
        got += m * n
    s = np.concatenate(parts)[:nsp]
    rng.shuffle(s)
    return s


def ecdf_grid(s):
    return np.searchsorted(np.sort(s), SG, side="right") / len(s)


def _table_row(args):
    n, ns, seed = args
    rng = np.random.default_rng(seed + n)
    s = stream(n, ns + 2 * W, rng)
    return ecdf_grid(s), ecdf_grid(local_norm(s))


def cmd_table(a):
    with Pool(6) as p:
        rows = p.map(_table_row, [(n, a.ns, 5000) for n in NODES[::-1]], chunksize=1)  # big N first
    rows = rows[::-1]
    tab = np.array([[r[0] for r in rows], [r[1] for r in rows]])  # (2, nN, ng): dens, loc
    np.savez(a.out, nodes=np.array(NODES), s=SG, cdf=tab, ns=a.ns)
    print(f"table {tab.shape} ns={a.ns} -> {a.out}")


class Family:
    """F(s; u) for u = 1/N^2, natural cubic spline in u over the tabulated nodes."""

    def __init__(self, path, which):
        z = np.load(path)
        u = 1.0 / z["nodes"].astype(float) ** 2
        o = np.argsort(u)
        self.u, self.F = u[o], np.maximum.accumulate(z["cdf"][which][o], axis=1)
        self.cs = CubicSpline(self.u, self.F, bc_type="natural")
        self.d0 = self.cs(self.u[0], 1)

    def __call__(self, us):
        us = np.atleast_1d(np.asarray(us, float))
        out = self.cs(np.clip(us, self.u[0], None))
        low = us < self.u[0]
        out[low] += np.outer(us[low] - self.u[0], self.d0)
        return np.clip(np.maximum.accumulate(out, axis=1), 0, 1)


def ugrid():
    ns = np.arange(40.0, 1.9999, -0.05)  # from big N to 2 (last entry = edge)
    tail = np.linspace(-0.01, 1 / 40**2 - 2e-5, 400)
    return np.concatenate([tail, 1 / ns**2])


UG = ugrid()


def fit_ks(fam_grid, s):
    """(index of best u on UG, KS_min) for one sample."""
    d = np.abs(fam_grid - ecdf_grid(s)).max(axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def fit_lik(fam_grid, s):
    h = np.histogram(s, bins=EDGES)[0]
    ce = np.concatenate([fam_grid[:, EDGE_IDX[:-1]], np.ones((len(fam_grid), 1))], axis=1)
    p = np.maximum(np.diff(ce, axis=1), 1e-12)
    return int(np.argmax(np.log(p) @ h))


def n_of(u):
    return float(u) ** -0.5 if u > 1e-9 else float("inf")


def cmd_check(a):
    out = {}
    for w, tag in ((0, "dens"), (1, "loc")):
        z = np.load(a.table)
        nodes, cdf = z["nodes"], z["cdf"][w]
        loo = {}
        for k in range(3, 31):
            keep = nodes != k
            tmp = a.out + ".tmp.npz"
            np.savez(tmp, nodes=nodes[keep], cdf=np.array([cdf[keep]] * 2))
            fam = Family(tmp, 0)
            os.remove(tmp)
            loo[k] = float(np.abs(fam(1 / k**2)[0] - cdf[nodes == k][0]).max())
        fam = Family(a.table, w)
        big = nodes >= 15
        u = 1 / nodes[big].astype(float) ** 2
        d = cdf[big] - cdf[nodes == 128]
        coef = (u @ d) / (u @ u)
        res = np.abs(d - np.outer(u, coef)).max()
        out[tag] = {"loo_max": max(loo.values()), "loo_worst_k": max(loo, key=loo.get), "loo": loo,
                    "lin_resid_max_N_ge_15": float(res), "noise_expect": 0.5 / np.sqrt(float(z["ns"]))}
    json.dump(out, open(a.out, "w"), indent=1)
    for t, r in out.items():
        print(f"{t}: LOO max|dF|={r['loo_max']:.4f} (k={r['loo_worst_k']}), lin(1/N^2) resid N>=15 {r['lin_resid_max_N_ge_15']:.4f}, "
              f"noise~{r['noise_expect']:.4f}")


def _calib_one(args):
    ntrue, nrep, tag, table, seed = args
    fam = Family(table, 0 if tag == "dens" else 1)
    fg = fam(UG)
    rng = np.random.default_rng(seed + ntrue)
    us, ks = [], []
    for _ in range(nrep):
        s = stream(ntrue, 9999 + 2 * W, rng)
        s = s[:9999] if tag == "dens" else local_norm(s)[:9799]
        i, k = fit_ks(fg, s)
        us.append(UG[i])
        ks.append(k)
    return ntrue, tag, np.array(us), np.array(ks)


def cmd_calib(a):
    jobs = [(n, a.nrep, t, a.table, 7000) for n in (2, 3, 4, 6, 8, 11, 16) for t in ("dens", "loc")]
    with Pool(6) as p:
        res = p.map(_calib_one, jobs, chunksize=1)
    out = {}
    for n, t, us, ks in res:
        nn = np.array([n_of(u) for u in us])
        out[f"{t}_{n}"] = {"u_mean": float(us.mean()), "u_sd": float(us.std()), "u_true": 1 / n**2,
                           "N_median": float(np.median(nn)), "N_q16": float(np.quantile(nn, 0.16)),
                           "N_q84": float(np.quantile(nn, 0.84)), "frac_edge": float(np.mean(us >= 0.2499)),
                           "ks": ks.tolist()}
        print(f"{t:4s} N={n:2d}: u-hat {us.mean():.4f}±{us.std():.4f} (true {1/n**2:.4f}); N med {np.median(nn):.2f} "
              f"[{np.quantile(nn, .16):.2f},{np.quantile(nn, .84):.2f}]; edge {np.mean(us >= .2499):.2f}; KSmin med {np.median(ks):.4f}")
    json.dump(out, open(a.out, "w"))


def boot_samples(s, nboot, rng):
    n = len(s)
    nb = -(-n // NBLK)
    st = rng.integers(0, n - NBLK + 1, size=(nboot, nb))
    idx = (st[:, :, None] + np.arange(NBLK)).reshape(nboot, -1)[:, :n]
    return s[idx]


def wfit(x, y, sd, intercept=False):
    w = 1 / sd**2
    if intercept:
        A = np.stack([x, np.ones_like(x)], axis=1)
        return np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    return np.array([(w * x * y).sum() / (w * x * x).sum()])


def cmd_fit(a):
    cal = json.load(open(a.calib))
    fams = {"dens": Family(a.table, 0), "loc": Family(a.table, 1)}
    fgs = {t: f(UG) for t, f in fams.items()}
    rng = np.random.default_rng(31)
    blocks = []
    for f in [f for f in FILES if os.path.exists(f)]:
        base, x, _ = stats.load(f)
        s_d = stats.unfold(base, x)
        t = float(base) + float(np.mean(x))
        b = {"file": os.path.basename(f), "T": t, "L": float(np.log(t / (2 * np.pi))), "N_eff": float(neff(t))}
        for tag, s in (("dens", s_d), ("loc", local_norm(s_d))):
            i, ks = fit_ks(fgs[tag], s)
            il = fit_lik(fgs[tag], s)
            ub = np.empty(a.nboot)
            for j, sb in enumerate(boot_samples(s, a.nboot, rng)):
                ub[j] = UG[fit_ks(fgs[tag], sb)[0]]
            nb = np.array([min(n_of(u), 1e6) for u in ub])
            near = min((2, 3, 4, 6, 8, 11, 16), key=lambda n: abs(n - min(b["N_eff"], 16)) if UG[i] < 0.24 else abs(n - 2))
            ref = np.array(cal[f"{tag}_{near}"]["ks"])
            b[tag] = {"u": float(UG[i]), "N": n_of(UG[i]), "censored": bool(UG[i] >= 0.2499), "ks_min": ks,
                      "N_lik": n_of(UG[il]), "u_q": np.quantile(ub, [0.025, 0.16, 0.5, 0.84, 0.975]).tolist(),
                      "N_q": np.quantile(nb, [0.025, 0.16, 0.5, 0.84, 0.975]).tolist(),
                      "frac_edge": float(np.mean(ub >= 0.2499)), "u_sd": float(ub.std()), "gof_ref_N": near,
                      "p_gof": float((1 + np.sum(ref >= ks)) / (len(ref) + 1)), "ub": ub.tolist()}
        blocks.append(b)
        d, l = b["dens"], b["loc"]
        print(f"{b['file']:16s} N_eff={b['N_eff']:5.2f} | dens N={d['N']:6.2f} [{d['N_q'][1]:.2f},{d['N_q'][3]:.2f}]"
              f"{' cens' if d['censored'] else ''} pGOF={d['p_gof']:.2f} | loc N={l['N']:6.2f} [{l['N_q'][1]:.2f},{l['N_q'][3]:.2f}]"
              f"{' cens' if l['censored'] else ''} pGOF={l['p_gof']:.2f}", flush=True)
    res = {"blocks": blocks, "beta0": 1 / np.sqrt(12 * LAM), "k0": 12 * LAM}
    for tag in ("dens", "loc"):
        use = [b for b in blocks if not b[tag]["censored"] and b[tag]["u_sd"] > 0]
        x = np.array([1 / b["L"] ** 2 for b in use])
        y = np.array([b[tag]["u"] for b in use])
        sd = np.array([b[tag]["u_sd"] for b in use])
        L = np.array([b["L"] for b in use])
        ub = np.array([b[tag]["ub"] for b in use])  # (blocks, B)
        ncap = np.minimum(np.where(ub > 1e-9, np.maximum(ub, 1e-12) ** -0.5, 60.0), 60.0)
        sdn = ncap.std(axis=1)
        y_n = np.array([min(b[tag]["N"], 60.0) for b in use])
        k = wfit(x, y, sd)[0]
        kb = np.array([wfit(x, ub[:, j], sd)[0] for j in range(ub.shape[1])])
        beta = wfit(L, y_n, sdn)[0]
        bb = np.array([wfit(L, ncap[:, j], sdn)[0] for j in range(ub.shape[1])])
        ai = wfit(L, y_n, sdn, intercept=True)
        aib = np.array([wfit(L, ncap[:, j], sdn, intercept=True) for j in range(ub.shape[1])])
        res[tag] = {"n_used": len(use), "k": float(k), "k_sd": float(kb.std()), "k_q": np.quantile(kb, [.025, .975]).tolist(),
                    "beta": float(beta), "beta_sd": float(bb.std()), "beta_q": np.quantile(bb, [.025, .975]).tolist(),
                    "beta_int": float(ai[0]), "beta_int_sd": float(aib[:, 0].std()), "c_int": float(ai[1]), "c_int_sd": float(aib[:, 1].std()),
                    "z_beta": float((beta - res["beta0"]) / bb.std()), "z_k": float((k - res["k0"]) / kb.std())}
        r = res[tag]
        print(f"{tag}: blocks used {r['n_used']}; k={r['k']:.2f}±{r['k_sd']:.2f} (k0={res['k0']:.2f}, z={r['z_k']:+.1f}); "
              f"beta={r['beta']:.3f}±{r['beta_sd']:.3f} (0.230, z={r['z_beta']:+.1f}); "
              f"beta+c: {r['beta_int']:.3f}±{r['beta_int_sd']:.3f}, c={r['c_int']:.2f}±{r['c_int_sd']:.2f}")
    json.dump(res, open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "check", "calib", "fit"])
    ap.add_argument("--table", default="results/test2/cue_table.npz")
    ap.add_argument("--calib", default="results/test2/calib.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", type=int, default=2_000_000)
    ap.add_argument("--nrep", type=int, default=300)
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    {"table": cmd_table, "check": cmd_check, "calib": cmd_calib, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
