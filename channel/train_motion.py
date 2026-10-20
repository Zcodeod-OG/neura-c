"""Can a grid carry motion? Moving person / dog / car sequences from motion3d.py.

Each frame goes through the same channel as before (N x N, 4 levels, noise 0.05). The reader names both
the motion (approaching, moving away, crossing left, crossing right, standing still) and the object.
Per-frame codes, all keeping absolute nearness (a nearer object gives stronger electrodes):
  outline  - depth edges, strongest value per electrode
  trail    - outline plus fading copies of the last two frames (speed streaks)
  learned  - CNN on this frame and the previous one, kept close to the outline (alpha)
Readers:
  frame    - sees only the newest frame, as a person feeling one snapshot would
  seq      - sees all 6 frames (1 s)
Reference: the seq reader on the full 64 x 64 depth frames.

Usage: python3 train_motion.py <motion3d.pt> [out_json] [patterns_json]
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
EPOCHS = 15
RUNS = [("full", "seq", 64), ("outline", "frame", 8), ("trail", "frame", 8), ("learned", "frame", 8),
        ("outline", "seq", 8), ("trail", "seq", 8), ("learned", "seq", 8),
        ("outline", "frame", 16), ("trail", "frame", 16), ("outline", "seq", 16)]
ALPHA = 1.0
EXPORT = ["outline_seq_8", "trail_seq_8"]
PER_COMBO_EXPORT = 20


class Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x):
        return torch.sigmoid(F.adaptive_avg_pool2d(self.net(x), self.n) * 4)


class Reader(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, 48, 3, padding=1), nn.ReLU(), nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(), nn.Linear(128, 5 + 3))

    def forward(self, g):
        out = self.fc(self.conv(F.interpolate(g, size=max(g.shape[-1], 8), mode="nearest")))
        return out[:, :5], out[:, 5:]


def grids(kind, x, n, enc=None):
    """x: [B,T,2,64,64] -> per-frame grids [B,T,n,n] after the channel."""
    B, T = x.shape[:2]
    e = x[:, :, 1]
    if kind == "outline":
        g = F.adaptive_max_pool2d(e, n)
    elif kind == "trail":
        prev1 = torch.cat([e[:, :1], e[:, :-1]], 1)
        prev2 = torch.cat([e[:, :1], e[:, :1], e[:, :-2]], 1)
        g = F.adaptive_max_pool2d(torch.maximum(e, torch.maximum(0.5 * prev1, 0.25 * prev2)), n)
    else:
        prev = torch.cat([x[:, :1], x[:, :-1]], 1)
        g = enc(torch.cat([x, prev], 2).flatten(0, 1)).reshape(B, T, n, n)
    return tc.channel(g)


def run(kind, reader_kind, n, Xtr, Mtr, Otr, Xte, Mte, Ote, export_idx):
    torch.manual_seed(0)
    T = Xtr.shape[1]
    cin = 2 * T if kind == "full" else (T if reader_kind == "seq" else 1)
    reader = Reader(cin)
    enc = Encoder(n) if kind == "learned" else None
    opt = torch.optim.Adam(list(reader.parameters()) + (list(enc.parameters()) if enc else []), 1e-3)

    def forward(x):
        if kind == "full":
            return reader(x.flatten(1, 2)), None
        g = grids(kind, x, n, enc)
        return reader(g if reader_kind == "seq" else g[:, -1:]), g

    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            x = Xtr[idx].float()
            (pm, po), g = forward(x)
            loss = F.cross_entropy(pm, Mtr[idx]) + F.cross_entropy(po, Otr[idx])
            if enc is not None:
                loss = loss + 0.5 * F.relu(g.mean() - tc.SPARSITY) + ALPHA * F.mse_loss(g, grids("outline", x, n))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pm_all, po_all = [], []
        for i in range(0, len(Xte), 200):
            (pm, po), _ = forward(Xte[i:i + 200].float())
            pm_all.append(pm.argmax(1)); po_all.append(po.argmax(1))
        pm, po = torch.cat(pm_all), torch.cat(po_all)
        conf = np.zeros((5, 5), int)
        for t, p in zip(Mte.tolist(), pm.tolist()):
            conf[t, p] += 1
        appr_t, appr_p = Mte == 0, pm == 0
        torch.manual_seed(1)
        patterns = None
        if kind != "full":
            patterns = (grids(kind, Xte[export_idx].float(), n, enc) * 3).round().int().tolist()
    return {"motion_accuracy": (pm == Mte).float().mean().item(), "object_accuracy": (po == Ote).float().mean().item(),
            "approach_recall": (appr_t & appr_p).sum().item() / appr_t.sum().item(),
            "approach_false_alarm": (~appr_t & appr_p).sum().item() / (~appr_t).sum().item(),
            "motion_confusion": conf.tolist()}, patterns


if __name__ == "__main__":
    d = torch.load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "../results/motion_results.json"
    pat_out = sys.argv[3] if len(sys.argv) > 3 else "../reading-test/patterns_motion.json"
    Xtr, Mtr, Otr, Xte, Mte, Ote = d["Xtr"], d["Mtr"], d["Otr"], d["Xte"], d["Mte"], d["Ote"]
    combo = Ote * 5 + Mte
    export_idx = [i for c in range(15) for i in (combo == c).nonzero().flatten()[:PER_COMBO_EXPORT].tolist()]
    res = {"objects": d["objects"], "motions": d["motions"], "epochs": EPOCHS, "frames": Xtr.shape[1],
           "n_train": len(Xtr), "n_test": len(Xte), "runs": {}}
    pats = {"objects": d["objects"], "motions": d["motions"], "object_labels": Ote[export_idx].tolist(),
            "motion_labels": Mte[export_idx].tolist(),
            "thumbs": [bytes(t.flatten().tolist()).hex() for t in d["Tte"][export_idx]], "runs": {}}
    t0 = time.time()
    for kind, reader_kind, n in RUNS:
        key = f"{kind}_{reader_kind}_{n}"
        r, p = run(kind, reader_kind, n, Xtr, Mtr, Otr, Xte, Mte, Ote, export_idx)
        res["runs"][key] = r
        if key in EXPORT:
            pats["runs"][key] = p
        print(key, "motion", round(r["motion_accuracy"], 3), "object", round(r["object_accuracy"], 3),
              "approach recall", round(r["approach_recall"], 3), "false alarm", round(r["approach_false_alarm"], 3),
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"))
    json.dump(pats, open(pat_out, "w"), separators=(",", ":"))
