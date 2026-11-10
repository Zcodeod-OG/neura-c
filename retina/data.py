"""Real video clips for the retina experiments.

Source: the 30 DAVIS clips packaged with TAP-Vid (storage.googleapis.com/dm-tapnet/tapvid_davis.zip), 480p at
24 frames per second. `prepare` converts them to greyscale at quarter size and stores one array per clip.
`Clips` then cuts random space-time crops and resamples them to the 120 Hz simulation clock.

The pickle is read with an unpickler that only accepts numpy arrays.

Usage: python3 data.py <tapvid_davis.pkl> <out.npz>
"""
import importlib
import pickle
import sys
import numpy as np
import torch
import torch.nn.functional as F

SIM_HZ = 120
VIDEO_HZ = 24
UP = SIM_HZ // VIDEO_HZ                      # 5 simulation bins per video frame
TEST = ["blackswan", "car-roundabout", "dogs-jump", "judo", "parkour", "soapbox"]
_OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
       ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


class _ArraysOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _OK:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def prepare(pkl, out, shrink=4):
    d = _ArraysOnly(open(pkl, "rb")).load()
    arrays = {}
    for name, v in d.items():
        g = torch.tensor(v["video"]).float().mean(-1)[:, None] / 255          # [T,1,H,W]
        g = F.avg_pool2d(g, shrink)
        arrays[name] = (g[:, 0] * 255).round().byte().numpy()
        print(name, arrays[name].shape, flush=True)
    np.savez_compressed(out, **arrays)


class Clips:
    """Random crops: `sample(b)` returns contrast movies [b, T, size, size] on the simulation clock."""

    def __init__(self, npz, split, size=48, seconds=1.0, seed=0):
        d = np.load(npz)
        names = [n for n in d.files if (n in TEST) == (split == "test")]
        self.videos = [torch.tensor(d[n]).float() / 255 for n in names]
        self.names, self.size, self.frames = names, size, int(round(seconds * VIDEO_HZ)) + 1
        self.T = (self.frames - 1) * UP
        self.rng = np.random.default_rng(seed)

    def sample(self, b, augment=True):
        out = []
        for _ in range(b):
            v = self.videos[self.rng.integers(len(self.videos))]
            t0 = self.rng.integers(v.shape[0] - self.frames + 1)
            y0 = self.rng.integers(v.shape[1] - self.size + 1)
            x0 = self.rng.integers(v.shape[2] - self.size + 1)
            c = v[t0:t0 + self.frames, y0:y0 + self.size, x0:x0 + self.size]
            if augment and self.rng.random() < 0.5:
                c = c.flip(2)
            out.append(c)
        return to_contrast(upsample_time(torch.stack(out)))


def upsample_time(x):
    """[b, F, H, W] at 24 Hz -> [b, (F-1)*UP, H, W] at 120 Hz by linear interpolation."""
    b, f, h, w = x.shape
    y = F.interpolate(x.permute(0, 2, 3, 1).reshape(b, h * w, f), size=(f - 1) * UP + 1, mode="linear", align_corners=True)
    return y.reshape(b, h, w, -1).permute(0, 3, 1, 2)[:, :-1]


def to_contrast(x):
    """Luminance in [0,1] -> Weber contrast about each clip's mean, clipped to [-1, 1]."""
    m = x.mean((1, 2, 3), keepdim=True).clamp(min=0.05)
    return ((x - m) / m).clamp(-1, 1)


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2])
