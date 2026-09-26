"""Quick environment check (about a minute). Run from the project root:

    python tests/baseline_check.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.explicit import prime_powers, spectrum_peaks  # noqa: E402
from src.rmt import rp_sample  # noqa: E402
from src.stats import unfold  # noqa: E402

KNOWN = [14.134725141734693, 21.022039638771555, 25.010857580145688, 30.424876125859513, 32.935061587739189]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


t0 = time.time()
import flint  # noqa: E402

flint.ctx.prec = 64
g = np.array([float(z.imag.mid()) for z in flint.acb.zeta_zeros(1, 1000)])
check("python-flint: first five zeros", np.allclose(g[:5], KNOWN, atol=1e-9), f"({time.time() - t0:.1f} s)")

s = unfold(0, g)
check("unfolded mean spacing ~ 1", abs(s.mean() - 1) < 0.02, f"(mean {s.mean():.4f})")

peaks = spectrum_peaks(g[:300])
pp = [q for q, _, _ in prime_powers(32)]
primes = [q for q, _, k in prime_powers(32) if k == 1]
spurious = [round(v, 2) for v in peaks if min(abs(q - v) for q in pp) >= 0.1]
found_primes = all(any(abs(v - p) < 0.1 for v in peaks) for p in primes)
check("spectroscopy: all primes <= 31 found, nothing spurious", found_primes and not spurious,
      f"(spurious: {spurious})")

rng = np.random.default_rng(1)


def p01(lam, kind, n=150, r=40):
    s, pr = zip(*(rp_sample(n, lam, kind, rng) for _ in range(r)))
    return float(np.mean(np.concatenate(s) < 0.1)), float(np.concatenate(pr).mean())


a, _ = p01(0, "gue")
check("model H + lambda*G, lambda = 0 ~ Poisson", 0.05 < a < 0.13, f"(p01 {a:.4f}; Poisson 0.095)")
b, prb = p01(None, "gue")
check("pure complex G ~ GUE", b < 0.006 and 0.44 < prb < 0.56, f"(p01 {b:.4f}, PR/N {prb:.3f})")
c, prc = p01(None, "goe")
check("pure real G ~ GOE", 0.003 < c < 0.016 and 0.28 < prc < 0.39, f"(p01 {c:.4f}, PR/N {prc:.3f})")

print(f"{sum(results)}/{len(results)} checks passed in {time.time() - t0:.0f} s")
sys.exit(0 if all(results) else 1)
