"""Semantic code: tell the user what is there, where, and how near, instead of sending a picture.

Scenes: one of the 10 objects from scenes3d.py placed at a random direction (within the camera's 60 degree
view) and distance in front of a head-height depth camera. A small CNN ("slot network") reads the full
64 x 64 depth image and outputs three slots:
  what   - which of the 10 objects
  where  - one of 3 directions: left, ahead, right (20 degree sectors; 5 sectors were too fine to hear)
  near   - nearer or further than 3 m
That is log2(10 * 3 * 2) = 5.9 bits per message, small enough for speech, spatial audio or a few tactors.

For the human test the script also exports, per test scene, the slot network's own answers (so the test
is end to end, mistakes included) and the 8 x 8 outline grid blurred to skin resolution for comparison.

Usage: python3 semantic3d.py <out_json> [patterns_json]
"""
import json
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import scenes3d as s3
import train_channel as tc

DIRECTIONS = ["Left", "Ahead", "Right"]
NEAR_M = 3.0
SECTOR = 20.0                       # degrees per direction sector, 3 sectors = 60 degree view
ND = len(DIRECTIONS)
PER_CLASS_TRAIN, PER_CLASS_TEST, EPOCHS = 1600, 200, 20
EXPORT_PER_CLASS = 12
SKIN_BLUR = 0.7                     # Gaussian blur, in electrodes, standing in for skin spatial acuity


def scene(cls, b):
    sdf, size = s3.BUILDERS[cls](b)
    yaw = s3.U(b, 0, 2 * math.pi)[:, 0]
    lo, hi = (3.0, 9.0) if cls == 4 else (1.5, 6.0)                 # cars are met further away
    dist = lo + (hi - lo) * torch.rand(b)
    az = (torch.rand(b) - 0.5) * ND * SECTOR                         # degrees, uniform over the sectors
    x, z = dist * torch.sin(torch.deg2rad(az)), dist * torch.cos(torch.deg2rad(az))
    fwd = F.normalize(torch.tensor([[0.0, 0.7 - s3.CAM_H, -6.0]]), dim=1).expand(b, 3)
    cam = torch.stack([-x, torch.full((b,), s3.CAM_H), z], 1)        # object at origin, camera moved instead
    img, th, cover = s3.view(sdf, yaw[:, None], cam, fwd)
    where = ((az + ND * SECTOR / 2) // SECTOR).clamp(0, ND - 1).long()
    near = (dist < NEAR_M).long()
    return img, th, cover, where, near


def make(per_class, batch=200):
    X, Y, W, N, T = [], [], [], [], []
    for c in range(10):
        done = 0
        while done < per_class:
            b = min(batch, per_class - done)
            img, th, cover, w, n = scene(c, b)
            keep = cover > 0.002
            X.append(img[keep].half()); T.append((th[keep] * 255).byte())
            Y.append(torch.full((int(keep.sum()),), c)); W.append(w[keep]); N.append(n[keep])
            done += int(keep.sum())
    return torch.cat(X), torch.cat(Y), torch.cat(W), torch.cat(N), torch.cat(T)


class SlotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 256), nn.ReLU(), nn.Linear(256, 10 + ND + 2))

    def forward(self, x):
        o = self.fc(self.conv(x))
        return o[:, :10], o[:, 10:10 + ND], o[:, 10 + ND:]


def skin_grid(x, n=8):
    """The old picture code for comparison: outline on an n x n grid, blurred to skin resolution, 4 levels."""
    g = F.adaptive_max_pool2d(x[:, 1:2].float(), n)
    k = torch.arange(-2, 3, dtype=torch.float32)
    k = torch.exp(-k ** 2 / (2 * SKIN_BLUR ** 2)); k = k / k.sum()
    g = F.conv2d(g, k.view(1, 1, 1, 5), padding=(0, 2))
    g = F.conv2d(g, k.view(1, 1, 5, 1), padding=(2, 0))
    return tc.channel(tc.normalise(g))[:, 0]


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = sys.argv[1]
    pat_out = sys.argv[2] if len(sys.argv) > 2 else None
    t0 = time.time()
    Xtr, Ytr, Wtr, Ntr, _ = make(PER_CLASS_TRAIN)
    Xte, Yte, Wte, Nte, Tte = make(PER_CLASS_TEST)
    print("rendered", len(Xtr), len(Xte), f"{time.time()-t0:.0f}s", flush=True)
    net = SlotNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            a, b, c = net(Xtr[idx].float())
            loss = F.cross_entropy(a, Ytr[idx]) + F.cross_entropy(b, Wtr[idx]) + F.cross_entropy(c, Ntr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        print("epoch", ep, f"{time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        preds = [torch.cat(p) for p in zip(*[[h.argmax(1) for h in net(Xte[i:i + 500].float())] for i in range(0, len(Xte), 500)])]
    pa, pw, pn = preds
    ok_a, ok_w, ok_n = pa == Yte, pw == Wte, pn == Nte
    res = {"classes": s3.CLASSES, "directions": DIRECTIONS, "near_m": NEAR_M, "n_train": len(Xtr), "n_test": len(Xte),
           "what": ok_a.float().mean().item(), "where": ok_w.float().mean().item(), "near": ok_n.float().mean().item(),
           "all_three": (ok_a & ok_w & ok_n).float().mean().item(),
           "where_within_one_sector": ((pw - Wte).abs() <= 1).float().mean().item(),
           "bits_per_message": math.log2(10 * ND * 2)}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, list)}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
    if pat_out:
        idx = [i for c in range(10) for i in (Yte == c).nonzero().flatten()[:EXPORT_PER_CLASS].tolist()]
        torch.manual_seed(1)
        grids = (skin_grid(Xte[idx]) * 3).round().int().tolist()
        json.dump({"classes": s3.CLASSES, "directions": DIRECTIONS,
                   "truth": [[int(Yte[i]), int(Wte[i]), int(Nte[i])] for i in idx],
                   "said": [[int(pa[i]), int(pw[i]), int(pn[i])] for i in idx],
                   "grids": grids, "thumbs": [bytes(Tte[i].flatten().tolist()).hex() for i in idx]},
                  open(pat_out, "w"), separators=(",", ":"))
