"""Random-matrix models: null and alternative models for the zeros, and the
TERJ model H + lambda*G (Rosenzweig-Porter).

Normalisation: off-diagonal E|G_ij|^2 = 1; the diagonal H has uniform levels
with mean spacing 1, so lambda is measured in units of the mean level spacing.
"""
import numpy as np


def goe(n, rng):
    a = rng.normal(size=(n, n))
    return (a + a.T) / np.sqrt(2)


def gue(n, rng):
    a = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    return (a + a.conj().T) / np.sqrt(2)


def local_unfold(e, w=8):
    """Spacings from the central half of a spectrum, divided by a moving local mean."""
    e = np.sort(e)
    n = len(e)
    out = [(e[i + 1] - e[i]) / ((e[i + 1 + w] - e[i - w]) / (2 * w + 1))
           for i in range(max(w, n // 4), min(n - w - 1, 3 * n // 4))]
    return np.array(out)


def rp_sample(n, lam, kind, rng):
    """One realisation of H + lam*G. lam=None means pure G (the lambda -> infinity limit).

    Returns (unfolded spacings, participation ratio / n of central eigenvectors).
    """
    g = gue(n, rng) if kind == "gue" else goe(n, rng)
    h = g if lam is None else np.diag(np.sort(rng.uniform(0, n, n))) + lam * g
    e, v = np.linalg.eigh(h)
    c = np.abs(v[:, n // 4: 3 * n // 4]) ** 2
    return local_unfold(e), 1 / np.sum(c ** 2, axis=0) / n


def cue_spacings(n, rng):
    """Eigenphase spacings of a Haar unitary n x n matrix (Mezzadri's QR recipe), mean 1."""
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    th = np.sort(np.angle(np.linalg.eigvals(q)))
    return np.diff(np.concatenate([th, [th[0] + 2 * np.pi]])) * n / (2 * np.pi)


def crossover_sample(n, alpha, rng):
    """Pandey-Mehta GOE -> GUE crossover: S + i*alpha*A, S GOE, A real antisymmetric."""
    b = rng.normal(size=(n, n))
    a = (b - b.T) / np.sqrt(2)
    return local_unfold(np.linalg.eigvalsh(goe(n, rng) + 1j * alpha * a))
