"""Pattern bank for the learning-curve reading test.

Trains the shape-keeping 8x8 encoder (alpha 4, same recipe as train_shaped.py) and exports, for each of
raw_8 and learned_8_a4, grid patterns of PER_CLASS test images per class that the first reading test did
not use, plus a 14x14 thumbnail of each original image to show as feedback.

Usage: python3 export_learning.py <fashion_mnist_dir> [out_json]
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
import train_channel as tc
import train_shaped as ts

PER_CLASS = 50
SKIP = 6  # the first 6 per class were used in the first reading test


def train_shaped_encoder(Xtr, Ytr, n=8, alpha=4.0):
    torch.manual_seed(0)
    reader, enc = tc.Reader(n), tc.Encoder(n)
    opt = torch.optim.Adam(list(reader.parameters()) + list(enc.parameters()), 1e-3)
    for ep in range(ts.EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            g = tc.channel(enc(Xtr[idx]))
            loss = (F.cross_entropy(reader(g), Ytr[idx]) + 0.5 * F.relu(g.mean() - tc.SPARSITY)
                    + alpha * F.mse_loss(g, tc.raw_grid(Xtr[idx], n)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch", ep, flush=True)
    return enc


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = tc.load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "../reading-test/learning_patterns.json"
    idx = [i for c in range(10) for i in (Yte == c).nonzero().flatten()[SKIP:SKIP + PER_CLASS].tolist()]
    x = Xte[idx]
    enc = train_shaped_encoder(Xtr, Ytr)
    torch.manual_seed(1)
    with torch.no_grad():
        runs = {"raw_8": tc.channel(tc.raw_grid(x, 8)), "learned_8_a4": tc.channel(enc(x))}
        thumbs = F.avg_pool2d(x, 2).squeeze(1)
    res = {"classes": tc.CLASSES, "labels": Yte[idx].tolist(),
           "thumbs": [bytes((t * 255).clamp(0, 255).byte().flatten().tolist()).hex() for t in thumbs],
           "runs": {k: (v.squeeze(1).numpy() * 3).round().astype(int).tolist() for k, v in runs.items()}}
    json.dump(res, open(out, "w"), separators=(",", ":"))
    print("wrote", out, len(idx), "patterns per code")
