"""Consecutive nontrivial zeros of zeta via python-flint (Arb ball arithmetic).

Output .npz: base (int as str), x (float64 offsets, height = base + x),
start_index (int as str). Finished chunks are kept in <out>.parts/ so an
interrupted run resumes where it stopped.

    python -m src.zeros --start 1 --count 20000 --workers 6 --out data/z_1_20k.npz
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

CHUNK = 1000


def n_smooth(t):
    """Smooth part of the Riemann-von Mangoldt counting function."""
    t = np.asarray(t, dtype=float)
    return t / (2 * np.pi) * (np.log(t / (2 * np.pi)) - 1) + 7 / 8


def _worker(job):
    start, num, prec, base = job
    import flint
    flint.ctx.prec = prec
    b = flint.arb(base)
    zs = flint.acb.zeta_zeros(start, num)
    x = np.array([float((z.imag - b).mid()) for z in zs])
    rad = max(float(z.imag.rad()) for z in zs)
    return start, x, rad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="index of first zero (1 = 14.1347...)")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prec", type=int, default=0, help="working precision in bits, 0 = auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import flint
    first = float(flint.acb.zeta_zeros(a.start, 1)[0].imag.mid())
    base = int(first) if first > 1e6 else 0
    prec = a.prec or 64 + int(np.log2(max(first, 2.0)))
    parts = a.out + ".parts"
    os.makedirs(parts, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    jobs = []
    for s in range(a.start, a.start + a.count, CHUNK):
        num = min(CHUNK, a.start + a.count - s)
        if not os.path.exists(os.path.join(parts, f"{s}.npy")):
            jobs.append((s, num, prec, base))
    print(f"height ~{first:.6g}, base={base}, prec={prec} bits, {len(jobs)} chunks to compute", flush=True)

    t0, worst = time.time(), 0.0
    if jobs:
        with Pool(a.workers) as pool:
            for i, (s, x, rad) in enumerate(pool.imap_unordered(_worker, jobs), 1):
                np.save(os.path.join(parts, f"{s}.npy"), x)
                worst = max(worst, rad)
                if i % 5 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} chunks, {time.time() - t0:.0f} s", flush=True)

    starts = range(a.start, a.start + a.count, CHUNK)
    x = np.concatenate([np.load(os.path.join(parts, f"{s}.npy")) for s in starts])
    assert len(x) == a.count, (len(x), a.count)
    assert np.all(np.diff(x) > 0), "zeros are not strictly increasing"
    np.savez(a.out, base=str(base), x=x, start_index=str(a.start))

    msg = f"saved {len(x)} zeros, heights {base + x[0]:.6f} .. {base + x[-1]:.6f} -> {a.out}"
    if worst:
        msg += f"; max ball radius {worst:.1e}"
    print(msg)
    if base + x[-1] < 1e15:
        n = np.arange(a.start, a.start + a.count)
        d = n - 0.5 - n_smooth(base + x)
        print(f"index check: mean(n - 1/2 - N_smooth) = {d.mean():+.3f}, max|S| = {np.abs(d).max():.2f}")
        if abs(d.mean()) > 0.25:
            print("WARNING: index check failed (missing or duplicated zeros?)")


if __name__ == "__main__":
    main()
