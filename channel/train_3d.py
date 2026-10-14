"""Grid codes for 3D objects seen by a depth camera (data from scenes3d.py).

Same channel as the Fashion-MNIST experiment (N x N grid, 4 levels, noise 0.05) and the same reader
network standing in for the user. Codes:
  fill     - object nearness area-averaged onto the grid (nearer = stronger)
  outline  - depth-edge strength area-averaged onto the grid (the outline idea from the simulator)
  learned  - CNN encoder on nearness + edges, trained with the reader; alpha > 0 keeps it close to `fill`
  full     - reference: the reader on the full 64 x 64 nearness + edge images, no grid
Also exports test patterns per class for the human reading test.

Usage: python3 train_3d.py <objects3d.pt> [out_json] [patterns_json]
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_channel as tc

torch.set_num_threads(4)
EPOCHS = 10
RUNS = [("full", 64, None), ("fill", 8, None), ("outline", 8, None), ("learned", 8, 0.0), ("learned", 8, 4.0),
        ("fill", 16, None), ("outline", 16, None), ("learned", 16, 4.0)]
EXPORT = ["fill_8", "outline_8", "learned_8_a4"]
PER_CLASS_EXPORT = 50


class Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x):
        return torch.sigmoid(F.adaptive_avg_pool2d(self.net(x), self.n) * 4)


class Reader(nn.Module):
    def __init__(self, cin=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, g):
        return self.fc(self.conv(F.interpolate(g, size=max(g.shape[-1], 8), mode="nearest")))


def run(kind, n, alpha, Xtr, Ytr, Xte, Yte, export_idx):
    torch.manual_seed(0)
    reader = Reader(2 if kind == "full" else 1)
    enc = Encoder(n) if kind == "learned" else None
    opt = torch.optim.Adam(list(reader.parameters()) + (list(enc.parameters()) if enc else []), 1e-3)
    fill = lambda x: tc.raw_grid(x[:, :1], n)

    def encode(x):
        if kind == "full":
            return x
        if kind == "fill":
            return tc.channel(fill(x))
        if kind == "outline":
            return tc.channel(tc.raw_grid(x[:, 1:], n))
        return tc.channel(enc(x))

    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            x = Xtr[idx].float()
            g = encode(x)
            loss = F.cross_entropy(reader(g), Ytr[idx])
            if enc is not None:
                loss = loss + 0.5 * F.relu(g.mean() - tc.SPARSITY) + alpha * F.mse_loss(g, fill(x))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        correct, conf = 0, np.zeros((10, 10), int)
        for i in range(0, len(Xte), 500):
            x = Xte[i:i + 500].float()
            pred = reader(encode(x)).argmax(1)
            correct += (pred == Yte[i:i + 500]).sum().item()
            for t, p in zip(Yte[i:i + 500].tolist(), pred.tolist()):
                conf[t, p] += 1
        torch.manual_seed(1)
        patterns = None if kind == "full" else (encode(Xte[export_idx].float()).squeeze(1) * 3).round().int().tolist()
    return {"accuracy": correct / len(Xte), "confusion": conf.tolist()}, patterns


if __name__ == "__main__":
    d = torch.load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "../results/objects3d_results.json"
    pat_out = sys.argv[3] if len(sys.argv) > 3 else "../reading-test/patterns3d.json"
    Xtr, Ytr, Xte, Yte = d["Xtr"], d["Ytr"], d["Xte"], d["Yte"]
    export_idx = [i for c in range(10) for i in (Yte == c).nonzero().flatten()[:PER_CLASS_EXPORT].tolist()]
    res = {"classes": d["classes"], "epochs": EPOCHS, "n_train": len(Xtr), "n_test": len(Xte), "runs": {}}
    pats = {"classes": d["classes"], "labels": Yte[export_idx].tolist(),
            "thumbs": [bytes(t.flatten().tolist()).hex() for t in d["Tte"][export_idx]], "runs": {}}
    t0 = time.time()
    for kind, n, alpha in RUNS:
        key = f"{kind}_{n}" + ("" if alpha is None else f"_a{alpha:g}")
        r, p = run(kind, n, alpha or 0.0, Xtr, Ytr, Xte, Yte, export_idx)
        res["runs"][key] = r
        if key in EXPORT:
            pats["runs"][key] = p
        print(key, round(r["accuracy"], 4), f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"))
    json.dump(pats, open(pat_out, "w"), separators=(",", ":"))
