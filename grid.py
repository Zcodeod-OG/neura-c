"""Electrotactile grid model: max-pool a drawing onto N x N electrodes, few intensity levels, noise."""
import numpy as np


def to_grid(img, n, levels=4, thresh=0.15, noise=0.03, rng=None):
    h, w = img.shape
    c = h // n
    pooled = img[: c * n, : c * n].reshape(n, c, n, c)
    cov = (pooled > 0).mean((1, 3))           # fraction of the cell covered
    peak = pooled.max((1, 3))
    g = peak * np.clip(cov * c / 2, 0, 1)      # a single stray pixel drives weakly
    if rng is not None:
        g = g + rng.normal(0, noise, g.shape)
    g = np.where(g < thresh, 0, g)
    return np.ceil(np.clip(g, 0, 1) * (levels - 1)) / (levels - 1)


def raw_grid(gray, n, levels=4):
    h, w = gray.shape
    c = h // n
    g = gray[: c * n, : c * n].reshape(n, c, n, c).mean((1, 3))
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    return np.round(g * (levels - 1)) / (levels - 1)


def boundary_grid(inst, n):
    """Cells containing a boundary between the objects we care about (box, person, ball) and anything else."""
    obj = inst >= 2
    b = np.zeros_like(obj)
    b[:-1] |= (inst[:-1] != inst[1:]) & (obj[:-1] | obj[1:])
    b[:, :-1] |= (inst[:, :-1] != inst[:, 1:]) & (obj[:, :-1] | obj[:, 1:])
    h, _ = b.shape
    c = h // n
    return b[: c * n, : c * n].reshape(n, c, n, c).any((1, 3))


class GridSmoother:
    """Temporal smoothing with hysteresis so electrodes don't toggle on noise.

    Each cell tracks an exponential moving average; it switches on above `on` and only
    switches off below `off`. alpha=0.5 adds roughly one frame of lag.
    """

    def __init__(self, alpha=0.5, on=0.3, off=0.12, levels=4):
        self.alpha, self.on, self.off, self.levels = alpha, on, off, levels
        self.s = None
        self.lit = None

    def __call__(self, g):
        if self.s is None:
            self.s, self.lit = g.astype(np.float32), g > self.on
        else:
            self.s = self.alpha * g + (1 - self.alpha) * self.s
            self.lit = np.where(self.lit, self.s > self.off, self.s > self.on)
        out = np.where(self.lit, np.clip(self.s, 1 / (self.levels - 1), 1), 0)
        return np.ceil(out * (self.levels - 1)) / (self.levels - 1)
