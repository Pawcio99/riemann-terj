"""Heat-flow engine H_t(x) of the de Bruijn-Newman family, no I/O.

    Phi(u) = sum_n (2 pi^2 n^4 e^{9u} - 3 pi n^2 e^{5u}) exp(-pi n^2 e^{4u})
    H_t(x) = int_0^inf e^{t u^2} Phi(u) cos(x u) du

Production rule: trapezoid on an even, analytic integrand (strip |Im u| < pi/8), evaluated in mpmath
with working precision dps = digits + 0.171 zmax + 0.13 |tmin| + 15 (the result is ~ e^{-pi x/8}).
Step h < 2 pi / (2 zmax + 8 D ln10 / pi) follows from aliasing error ~ H(2 pi/h - x) [DO WERYFIKACJI:
checked empirically by halving h and doubling dps, see src.test3]. Derivatives are analytic in the
integrand: d/dx -> -u sin, d^2/dx^2 -> -u^2 cos, d/dt -> u^2 (so H_t = -H_xx).
"""
import math
from typing import Any

import mpmath as mp

LN10 = math.log(10.0)


def phi_nodes(us, dps):
    """Phi(u_k) for u_k >= 0 (list of mpf), series truncated at the working precision."""
    with mp.workdps(dps):
        nmax = int(math.sqrt((dps + 12) * LN10 / math.pi)) + 3
        pi = mp.pi
        out = []
        for u in us:
            e4 = mp.exp(4 * u)
            e5, e9 = mp.exp(5 * u), mp.exp(9 * u)
            s = mp.mpf(0)
            for n in range(1, nmax + 1):
                n2 = n * n
                s += (2 * pi * pi * n2 * n2 * e9 - 3 * pi * n2 * e5) * mp.exp(-pi * n2 * e4)
            out.append(s)
        return out


def phi_direct(u):
    """Phi(u) at the current mp precision (for the mp.quad reference)."""
    nmax = int(math.sqrt((mp.mp.dps + 12) * LN10 / math.pi)) + 3
    e4, e5, e9 = mp.exp(4 * u), mp.exp(5 * u), mp.exp(9 * u)
    s = mp.mpf(0)
    for n in range(1, nmax + 1):
        n2 = n * n
        s += (2 * mp.pi ** 2 * n2 * n2 * e9 - 3 * mp.pi * n2 * e5) * mp.exp(-mp.pi * n2 * e4)
    return s


class HeatFlow:
    def __init__(self, zmax, digits=20, tmin=-50.0, dps_factor=1.0, h_factor=1.0):
        self.zmax, self.digits, self.tmin = float(zmax), digits, float(tmin)
        base = digits + 0.171 * self.zmax + 0.13 * abs(self.tmin) + 15
        self.dps = int(math.ceil(dps_factor * base))
        d_eff = digits + 0.13 * abs(self.tmin)
        self.h = 2 * math.pi / (2 * self.zmax + 1.25 * 8 * d_eff * LN10 / math.pi) / h_factor
        umax = 0.25 * math.log((self.dps + 5) * LN10 / math.pi)
        self.K = int(math.ceil(umax / self.h))
        with mp.workdps(self.dps):
            self.u = [mp.mpf(k) * mp.mpf(self.h) for k in range(self.K + 1)]
            self.phi = phi_nodes(self.u, self.dps)
            self.u2 = [x * x for x in self.u]
        self._t = None
        self._w = None
        self.nev = 0

    def _weights(self, t):
        if self._t != t:
            with mp.workdps(self.dps):
                tt = mp.mpf(t)
                w = [p * mp.exp(tt * q) for p, q in zip(self.phi, self.u2)]
                w[0] = w[0] / 2
                self._w = [x * self.h for x in w]
            self._t = t
        assert self._w is not None
        return self._w

    def derivs(self, x, t):
        """(H, H_x, H_xx) at real x, time t (mpf results at working precision)."""
        self.nev += 1
        w = self._weights(t)
        with mp.workdps(self.dps):
            xx = mp.mpf(x)
            h0 = h1 = h2 = mp.mpf(0)
            for wk, u, q in zip(w, self.u, self.u2):
                c, s = mp.cos_sin(xx * u)
                a = wk * c
                h0 += a
                h1 -= wk * u * s
                h2 -= a * q
            return h0, h1, h2

    def H(self, x, t):
        return self.derivs(x, t)[0]


def h_quad(z, t, dps) -> Any:
    """Reference: mp.quad of the defining integral, split at k*pi/z."""
    with mp.workdps(dps):
        z = mp.mpf(z)
        umax = 0.25 * mp.log((dps + 5) * LN10 / mp.pi)
        nseg = int(umax * z / mp.pi) + 1
        pts = [mp.pi * k / z for k in range(nseg) if mp.pi * k / z < umax] + [umax]
        f = lambda u: mp.exp(t * u * u) * phi_direct(u) * mp.cos(z * u)
        return mp.quad(f, pts)
