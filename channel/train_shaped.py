"""Shape-keeping learned encoder.

Same setup as train_channel.py, but the learned encoder is also penalised for drifting away from the
object's own downsampled silhouette (mean squared difference to the raw grid, weight ALPHA). ALPHA = 0
is the unconstrained encoder; larger values trade machine accuracy for patterns a person can read as a shape.
Also exports a fixed set of test patterns per method for the human reading test.

Usage: python3 train_shaped.py <fashion_mnist_dir> [out_json]
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import train_channel as tc

EPOCHS = 8
RUNS = [("raw", 8, None), ("learned", 8, 0.0), ("learned", 8, 1.0), ("learned", 8, 4.0),
        ("raw", 20, None), ("learned", 20, 0.0), ("learned", 20, 4.0)]


def run(kind, n, alpha, Xtr, Ytr, Xte, Yte, export_idx):
    torch.manual_seed(0)
    reader = tc.Reader(n)
    enc = tc.Encoder(n) if kind == "learned" else None
    opt = torch.optim.Adam(list(reader.parameters()) + (list(enc.parameters()) if enc else []), 1e-3)

    def encode(x):
        return tc.channel(tc.raw_grid(x, n)) if kind == "raw" else tc.channel(enc(x))

    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            g = encode(Xtr[idx])
            loss = F.cross_entropy(reader(g), Ytr[idx])
            if enc is not None:
                loss = loss + 0.5 * F.relu(g.mean() - tc.SPARSITY) + alpha * F.mse_loss(g, tc.raw_grid(Xtr[idx], n))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        correct, lit, corr = 0, [], []
        for i in range(0, len(Xte), 1000):
            x = Xte[i:i + 1000]
            g = encode(x)
            correct += (reader(g).argmax(1) == Yte[i:i + 1000]).sum().item()
            lit.append((g > 0).float().mean().item())
            a = g.flatten(1) - g.flatten(1).mean(1, keepdim=True)
            b = tc.raw_grid(x, n).flatten(1)
            b = b - b.mean(1, keepdim=True)
            corr.append(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)).mean().item())
        patterns = encode(Xte[export_idx]).squeeze(1).numpy().round(3).tolist()
    return {"accuracy": correct / len(Xte), "electrodes_lit": float(np.mean(lit)),
            "shape_correlation": float(np.mean(corr)), "patterns": patterns}


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = tc.load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "../results/shaped_results.json"
    # 6 test images per class for the reading test, fixed across methods
    export_idx = [i for c in range(10) for i in (Yte == c).nonzero().flatten()[:6].tolist()]
    res = {"classes": tc.CLASSES, "epochs": EPOCHS, "export_labels": Yte[export_idx].tolist(), "runs": {}}
    t0 = time.time()
    for kind, n, alpha in RUNS:
        key = f"{kind}_{n}" + ("" if alpha is None else f"_a{alpha:g}")
        r = run(kind, n, alpha or 0.0, Xtr, Ytr, Xte, Yte, export_idx)
        res["runs"][key] = r
        print(key, round(r["accuracy"], 4), "lit", round(r["electrodes_lit"], 3), "shape", round(r["shape_correlation"], 3),
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"))
