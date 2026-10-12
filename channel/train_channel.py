"""Can a low-resolution electrotactile grid carry enough to tell objects apart?

For each grid size N (6..20) and each encoder, images are turned into an N x N pattern with 4
intensity levels plus noise (the channel), and a small "reader" network learns to name the object
from the pattern alone. Encoders:
  raw      - image area-averaged onto the grid (what image-based devices do)
  edges    - Sobel edge magnitude area-averaged onto the grid
  learned  - a CNN trained jointly with the reader, mean electrode drive capped at 20% of maximum
Reference: the same reader on the full 28 x 28 image.

Usage: python3 train_channel.py <fashion_mnist_dir> [out_json]
"""
import gzip
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
LEVELS = 4
NOISE = 0.05          # electrode/perception noise, in units of full intensity
SPARSITY = 0.20       # learned encoder: target mean electrode drive (fraction of max)
EPOCHS = 4
SIZES = [6, 8, 10, 14, 20]
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load(d):
    def imgs(f):
        with gzip.open(f"{d}/{f}") as fh:
            return np.frombuffer(fh.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255

    def labs(f):
        with gzip.open(f"{d}/{f}") as fh:
            return np.frombuffer(fh.read(), np.uint8, offset=8).astype(np.int64)

    return (torch.tensor(imgs("train-images-idx3-ubyte.gz")), torch.tensor(labs("train-labels-idx1-ubyte.gz")),
            torch.tensor(imgs("t10k-images-idx3-ubyte.gz")), torch.tensor(labs("t10k-labels-idx1-ubyte.gz")))


def quantize(x):
    """4 intensity levels with a straight-through gradient."""
    q = torch.round(x.clamp(0, 1) * (LEVELS - 1)) / (LEVELS - 1)
    return x + (q - x).detach()


def channel(g):
    return quantize(g + NOISE * torch.randn_like(g))


def normalise(g):
    lo = g.amin((2, 3), keepdim=True)
    hi = g.amax((2, 3), keepdim=True)
    return (g - lo) / (hi - lo + 1e-6)


SOBEL = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)


def raw_grid(x, n):
    return normalise(F.adaptive_avg_pool2d(x, n))


def edge_grid(x, n):
    gx = F.conv2d(x, SOBEL, padding=1)
    gy = F.conv2d(x, SOBEL.transpose(2, 3), padding=1)
    return normalise(F.adaptive_avg_pool2d(torch.sqrt(gx ** 2 + gy ** 2), n))


class Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x):
        return torch.sigmoid(F.adaptive_avg_pool2d(self.net(x), self.n) * 4)


class Reader(nn.Module):
    """Small CNN that names the object from a grid pattern (stands in for the trained user)."""

    def __init__(self, n):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, g):
        return self.fc(self.conv(F.interpolate(g, size=max(g.shape[-1], 8), mode="nearest")))


def run(kind, n, Xtr, Ytr, Xte, Yte):
    reader = Reader(n)
    enc = Encoder(n) if kind == "learned" else None
    params = list(reader.parameters()) + (list(enc.parameters()) if enc else [])
    opt = torch.optim.Adam(params, 1e-3)

    def encode(x):
        if kind == "full":
            return x
        if kind == "raw":
            return channel(raw_grid(x, n))
        if kind == "edges":
            return channel(edge_grid(x, n))
        return channel(enc(x))

    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            g = encode(Xtr[idx])
            loss = F.cross_entropy(reader(g), Ytr[idx])
            if enc is not None:
                loss = loss + 0.5 * F.relu(g.mean() - SPARSITY)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        correct, lit, conf = 0, [], np.zeros((10, 10), int)
        for i in range(0, len(Xte), 1000):
            g = encode(Xte[i:i + 1000])
            pred = reader(g).argmax(1)
            correct += (pred == Yte[i:i + 1000]).sum().item()
            lit.append((g > 0).float().mean().item())
            for t, p in zip(Yte[i:i + 1000].tolist(), pred.tolist()):
                conf[t, p] += 1
        examples = encode(Xte[:20]).squeeze(1).numpy().round(3).tolist() if kind != "full" else None
    return {"accuracy": correct / len(Xte), "electrodes_lit": float(np.mean(lit)),
            "bits_per_frame": n * n * 2 if kind != "full" else None, "confusion": conf.tolist(), "examples": examples}


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "results/channel_results.json"
    results = {"classes": CLASSES, "levels": LEVELS, "noise": NOISE, "sparsity_target": SPARSITY, "epochs": EPOCHS,
               "test_labels_first20": Yte[:20].tolist(), "test_images_first20": Xte[:20].squeeze(1).numpy().round(3).tolist(), "runs": {}}
    t0 = time.time()
    results["runs"]["full_28"] = run("full", 28, Xtr, Ytr, Xte, Yte)
    print("full", results["runs"]["full_28"]["accuracy"], f"{time.time()-t0:.0f}s", flush=True)
    for n in SIZES:
        for kind in ("raw", "edges", "learned"):
            r = run(kind, n, Xtr, Ytr, Xte, Yte)
            results["runs"][f"{kind}_{n}"] = r
            print(kind, n, round(r["accuracy"], 4), "lit", round(r["electrodes_lit"], 3), f"{time.time()-t0:.0f}s", flush=True)
            json.dump(results, open(out, "w"))
