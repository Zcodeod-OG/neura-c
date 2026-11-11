"""Encoders: camera movie -> binary projector frames.

Input is a contrast movie on the 120 Hz simulation clock, [b, T, H, W]. The camera and the projector both
run at 60 Hz, so every encoder first takes every second bin. Output is [b, T/2, H/pitch, W/pitch] in {0, 1}.
All are causal: frame t uses the camera up to frame t only.
  threshold  - pixel on where the scene is brighter than its mean (plain projection)
  dither     - plain projection with ordered dithering, so the lit fraction follows brightness
  event      - pixel on where brightness changed since the last frame (the GS030 goggles use an event camera)
  handmade   - blur, subtract background, clip, scale by two, then dither. The hand-designed modification of
               Ramakrishna and Gollisch (bioRxiv 2026), which they applied to still images.
  mapped     - like handmade, but it knows where the ON and OFF cells are: pixels next to an ON cell show the
               image, pixels next to an OFF cell show the inverted image, other pixels stay dark
  Learned    - space-time network trained through the treated-retina simulator; with `cell_maps` it is also
               given the positions of the ON and OFF cells
The hand-designed ones take a `delay` in frames, tuned like their other settings on training clips.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BAYER = torch.tensor([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]).float() / 16 + 1 / 32


def camera(movie, pitch):
    """120 Hz contrast movie -> 60 Hz frames on the projector grid, [b, Tf, h, w]."""
    return F.avg_pool2d(movie[:, ::2], pitch)


def _delay(x, d):
    return x if d == 0 else torch.cat([torch.zeros_like(x[:, :d]), x[:, :-d]], 1)


def _dither(level):
    """Level in [0, 1] -> binary, by an ordered-dither mask that shifts every frame."""
    b, T, h, w = level.shape
    m = BAYER.repeat(h // 4 + 1, w // 4 + 1)[:h, :w]
    masks = torch.stack([m.roll((t % 4, (t * 2) % 4), (0, 1)) for t in range(T)])
    return (level > masks).float()


def threshold(movie, pitch, delay=0, level=0.0):
    return _delay((camera(movie, pitch) > level).float(), delay)


def dither(movie, pitch, delay=0, gain=1.0):
    return _delay(_dither((0.5 + 0.5 * gain * camera(movie, pitch)).clamp(0, 1)), delay)


def event(movie, pitch, delay=0, level=0.05, hold=1):
    c = camera(movie, pitch)
    ev = (torch.cat([torch.zeros_like(c[:, :1]), (c[:, 1:] - c[:, :-1]).abs()], 1) > level).float()
    out = ev.clone()
    for k in range(1, hold):
        out = torch.maximum(out, _delay(ev, k))
    return _delay(out, delay)


def handmade(movie, pitch, delay=0, sigma=3.0, sub=0.08, invert=False):
    b, T, H, W = movie.shape
    x = -movie if invert else movie
    r = int(3 * sigma) + 1
    ax = torch.arange(-r, r + 1).float()
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = x[:, ::2].reshape(-1, 1, H, W)
    x = F.conv2d(F.pad(x, (r, r, r, r), mode="replicate"), g[None, None, :, None])
    x = F.conv2d(x, g[None, None, None, :]).reshape(b, -1, H, W)
    level = (2 * (x - sub)).clamp(0, 1)                         # contrast above background, clipped, times two
    return _delay(_dither(F.avg_pool2d(level, pitch)), delay)


def sign_map(centres, kinds, pitch, radius, size=48):
    """Per projector pixel: +1 if the nearest cell is ON, -1 if OFF, 0 if no cell lies within `radius` pixels."""
    n = size // pitch
    ax = (torch.arange(n).float() + 0.5) * pitch - 0.5
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    d = ((xx[..., None] - centres[:, 0]) ** 2 + (yy[..., None] - centres[:, 1]) ** 2).sqrt()
    dmin, idx = d.min(-1)
    s = torch.where(kinds[idx] == 0, 1.0, -1.0)
    return torch.where(dmin <= radius, s, torch.zeros_like(s))


def cell_maps(centres, kinds, size=48, sigma=1.0):
    """[2, size, size]: a blob at every ON cell and at every OFF cell."""
    ax = torch.arange(size).float()
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-((xx[None] - centres[:, 0, None, None]) ** 2 + (yy[None] - centres[:, 1, None, None]) ** 2) / (2 * sigma ** 2))
    return torch.stack([g[kinds == 0].sum(0), g[kinds == 1].sum(0)]).clamp(max=1)


def mapped(movie, pitch, sign, delay=0, sigma=1.0, sub=0.08):
    b, T, H, W = movie.shape
    r = int(3 * sigma) + 1
    ax = torch.arange(-r, r + 1).float()
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = movie[:, ::2].reshape(-1, 1, H, W)
    x = F.conv2d(F.pad(x, (r, r, r, r), mode="replicate"), g[None, None, :, None])
    x = F.conv2d(x, g[None, None, None, :]).reshape(b, -1, H, W)
    level = (2 * (F.avg_pool2d(x, pitch) * sign - sub)).clamp(0, 1)
    return _delay(_dither(level) * (sign != 0), delay)


class _Causal(nn.Module):
    """3D convolution that looks only backwards in time."""

    def __init__(self, cin, cout, kt, ks):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv3d(cin, cout, (kt, ks, ks), padding=(0, ks // 2, ks // 2))

    def forward(self, x):
        return self.conv(F.pad(x, (0, 0, 0, 0, self.kt - 1, 0)))


class Learned(nn.Module):
    """Sees the last 12 camera frames (200 ms). Binary output with a straight-through gradient."""

    def __init__(self, pitch, width=32, cell_maps=None):
        super().__init__()
        self.pitch = pitch
        self.register_buffer("maps", cell_maps)
        self.a = _Causal(1 if cell_maps is None else 3, width, 6, 5)
        self.b = _Causal(width, width, 4, 3)
        self.c = _Causal(width, width, 4, 3)
        self.out = nn.Conv3d(width, 1, 1)

    def logits(self, movie):
        x = movie[:, None, ::2]
        if self.maps is not None:
            x = torch.cat([x, self.maps[None, :, None].expand(x.shape[0], 2, x.shape[2], -1, -1)], 1)
        x = F.relu(self.b(F.relu(self.a(x))))
        if self.pitch > 1:
            x = F.avg_pool3d(x, (1, self.pitch, self.pitch))
        return self.out(F.relu(self.c(x)))[:, 0]

    def forward(self, movie, noise=0.0):
        """`noise` > 0 adds logistic noise to the logits (training only), which keeps pixels from all
        settling on or off before the network has learned when to flash them."""
        z = 3 * torch.tanh(self.logits(movie) / 3)       # bounded, so a pixel can always be flipped back
        if noise > 0:
            u = torch.rand_like(z).clamp(1e-6, 1 - 1e-6)
            z = z + noise * (u.log() - (1 - u).log())
        p = torch.sigmoid(z)
        return p + ((p > 0.5).float() - p).detach()
