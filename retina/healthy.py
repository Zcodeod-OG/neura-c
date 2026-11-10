"""Healthy retina models: what ON and OFF parasol-like ganglion cells would fire for a movie.

Two models share one cell mosaic so they can be compared cell by cell.
  HealthyLN       - the training target. Each cell = difference-of-Gaussians receptive field, a temporal
                    filter and a softplus output. Filters and outputs are the fits from fit_ln.py
                    (recorded cells 0 = OFF and 2 = ON). The receptive field sizes are assumptions.
  HealthySubunit  - the held-out target, never used for training. Different recorded cells (1 = OFF, 3 = ON),
                    small rectifying subunits inside each receptive field centre, and contrast gain control.

Units: 1 pixel = 10 um of retina, 1 time bin = 1/120 s. Movies are Weber contrast, [b, T, H, W].
"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["ON", "OFF"]
SIGMA = {"ON": 2.0, "OFF": 1.8}        # receptive field centre sigma, pixels (20 and 18 um)
SPACING = {"ON": 4.5, "OFF": 4.0}      # mosaic spacing, pixels
SURROUND = (3.0, 0.5)                  # surround sigma as a multiple of the centre, and its weight
RATE_CAP = 300.0


def mosaic(size=48, seed=0):
    """Jittered hexagonal mosaics. Returns centres [n, 2] (x, y) and a type index [n] (0 = ON, 1 = OFF)."""
    g = torch.Generator().manual_seed(seed)
    pts, kinds = [], []
    for ti, t in enumerate(TYPES):
        s = SPACING[t]
        off = torch.rand(2, generator=g) * s
        rows = int(size / (s * math.sqrt(3) / 2)) + 2
        for r in range(-1, rows):
            for c in range(-1, int(size / s) + 2):
                p = torch.tensor([c * s + (s / 2 if r % 2 else 0.0), r * s * math.sqrt(3) / 2]) + off
                p = p + 0.12 * s * torch.randn(2, generator=g)
                if 0 <= p[0] <= size - 1 and 0 <= p[1] <= size - 1:
                    pts.append(p); kinds.append(ti)
    return torch.stack(pts), torch.tensor(kinds)


def gaussians(centres, sigma, size):
    """[n, size*size] Gaussian weights, each row summing to 1. `sigma` is [n]."""
    ax = torch.arange(size).float()
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    d2 = (xx[None] - centres[:, 0, None, None]) ** 2 + (yy[None] - centres[:, 1, None, None]) ** 2
    w = torch.exp(-d2 / (2 * sigma[:, None, None] ** 2)).flatten(1)
    return w / w.sum(1, keepdim=True)


def lowpass(x, tau_bins):
    """Causal exponential low-pass along dim 1."""
    a = math.exp(-1.0 / tau_bins)
    out, s = [], torch.zeros_like(x[:, 0])
    for t in range(x.shape[1]):
        s = a * s + (1 - a) * x[:, t]
        out.append(s)
    return torch.stack(out, 1)


class _Base(nn.Module):
    def __init__(self, cells, size=48, seed=0, fits=None):
        super().__init__()
        f = json.load(open(fits or os.path.join(HERE, "ln_fits.json")))
        by_type = {f["cells"][c]["type"]: f["cells"][c] for c in cells}
        centres, kinds = mosaic(size, seed)
        sig = torch.tensor([SIGMA[TYPES[k]] for k in kinds.tolist()])
        centre, surround = gaussians(centres, sig, size), gaussians(centres, sig * SURROUND[0], size)
        self.size = size
        self.register_buffer("centres", centres)
        self.register_buffer("kinds", kinds)
        self.register_buffer("sigma", sig)
        self.register_buffer("Wc", centre)
        self.register_buffer("Ws", surround)
        self.register_buffer("k", torch.stack([torch.tensor(by_type[t]["filter"]) for t in TYPES]))      # [2, lags]
        self.register_buffer("bias", torch.tensor([by_type[t]["bias"] for t in TYPES])[kinds])
        self.register_buffer("gain", torch.tensor([by_type[t]["gain"] for t in TYPES])[kinds])

    def temporal(self, x, k):
        """Filter [b, T, n] along time with k [lags] (k[j] weights the input j bins ago)."""
        b, T, n = x.shape
        y = F.conv1d(F.pad(x.permute(0, 2, 1).reshape(b * n, 1, T), (len(k) - 1, 0)), k.flip(0)[None, None])
        return y.reshape(b, n, T).permute(0, 2, 1)

    def output(self, gen):
        return (self.gain * F.softplus(gen + self.bias)).clamp(max=RATE_CAP)       # spikes per second, [b, T, n]


class HealthyLN(_Base):
    def __init__(self, **kw):
        super().__init__(cells=(0, 2), **kw)
        self.register_buffer("W", (self.Wc - SURROUND[1] * self.Ws) / (1 - SURROUND[1]))

    def forward(self, movie):
        s = movie.flatten(2) @ self.W.T                                             # [b, T, n]
        gen = torch.where(self.kinds == 0, self.temporal(s, self.k[0]), self.temporal(s, self.k[1]))
        return self.output(gen)


class HealthySubunit(_Base):
    RECT = 0.3          # how much of the non-preferred sign a subunit passes
    SUB_SIGMA = 0.6     # subunit blur, pixels
    GC_TAU, GC_SIGMA = 18.0, 1.5   # gain control: 150 ms memory, half-strength at this generator size

    def __init__(self, **kw):
        super().__init__(cells=(1, 3), **kw)
        r = 2
        ax = torch.arange(-r, r + 1).float()
        g = torch.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * self.SUB_SIGMA ** 2))
        self.register_buffer("blur", (g / g.sum())[None, None])

    def forward(self, movie):
        b, T, H, W = movie.shape
        x = F.conv2d(movie.reshape(b * T, 1, H, W), self.blur, padding=2).reshape(b, T, H * W)
        gens = []
        for ti in range(2):
            y = self.temporal(x, self.k[ti])                                        # filtered per pixel
            sub = F.relu(y) + self.RECT * (-F.relu(-y))                             # rectifying subunits
            gens.append((sub @ self.Wc.T - SURROUND[1] * (y @ self.Ws.T)) / (1 - SURROUND[1]))
        gen = torch.where(self.kinds == 0, gens[0], gens[1])
        gen = gen / (1 + lowpass(gen.abs(), self.GC_TAU) / self.GC_SIGMA)
        return self.output(gen)
