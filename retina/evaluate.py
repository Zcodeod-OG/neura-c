"""Score every encoder on held-out test videos, against the training target and the held-out healthy model.

For each treatment target (opsin in ganglion cells, opsin in ON bipolar cells):
  - hand-designed encoders, settings tuned on training clips
  - the learned encoders from train.py: trained on firing match only ("learned"), and with the decoder
    term added ("learned+dec")
  - oracle (ganglion-cell target only): each cell gets its own binary light, optimised per clip
Scores:
  r, fve     - firing match to the healthy model (score.py), per cell type
  identify   - share of test clips whose treated response is closest to its own healthy response
  decode     - linear decoders fitted on healthy firing (all cells, ON only, OFF only) read the treated firing;
               correlation with what changed in the movie (score.py)
Everything is reported against HealthyLN (training target) and HealthySubunit (held-out).

Usage: python3 evaluate.py <clips.npz> <ckpt_dir> <out_json> [pitch]
"""
import itertools
import json
import os
import sys
import torch
import torch.nn.functional as F
import data
import encoders as E
import healthy
import score
import treated

GRIDS = {"threshold": (E.threshold, {"delay": [0, 1, 2, 3, 4], "level": [-0.2, 0.0, 0.2]}),
         "dither": (E.dither, {"delay": [0, 1, 2, 3, 4], "gain": [1, 2]}),
         "event": (E.event, {"delay": [0, 1, 2, 3], "level": [0.02, 0.05, 0.1], "hold": [1, 2, 3]}),
         "handmade": (E.handmade, {"delay": [0, 1, 2, 3, 4], "sigma": [1.5, 3, 6], "sub": [0.0, 0.08, 0.16],
                                   "invert": [False, True]})}


def tune(fn, grid, x, tr, target_rates, kinds, pitch):
    best = None
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        r = score.match(tr(fn(x, pitch, **kw)), target_rates, kinds)["r"]
        if best is None or r > best[0]:
            best = (r, kw)
    return best[1]


def identify(tr_rates, h_rates):
    a = tr_rates[:, score.WARM:].flatten(1); b = h_rates[:, score.WARM:].flatten(1)
    a = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
    b = (b - b.mean(1, keepdim=True)) / (b.std(1, keepdim=True) + 1e-6)
    return ((a @ b.T).argmax(1) == torch.arange(len(a))).float().mean().item()


def oracle(tr, target_rates, kinds, steps=300):
    logits = torch.zeros(target_rates.shape[0], target_rates.shape[1] // 2, target_rates.shape[2], requires_grad=True)
    opt = torch.optim.Adam([logits], 0.1)
    for _ in range(steps):
        p = torch.sigmoid(logits)
        l = score.loss(tr.per_cell(p + ((p > 0.5).float() - p).detach()), target_rates, kinds)
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        return tr.per_cell((logits > 0).float())


if __name__ == "__main__":
    npz, ckpt_dir, out = sys.argv[1:4]
    pitch = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tune_x = data.Clips(npz, "train", seed=5).sample(48)
    dec_x = data.Clips(npz, "train", seed=6).sample(160)
    test_x = data.Clips(npz, "test", seed=123).sample(120, augment=False)
    models = {"LN (training target)": healthy.HealthyLN(), "subunit (held-out)": healthy.HealthySubunit()}
    kinds = models["LN (training target)"].kinds
    with torch.no_grad():
        tune_h = models["LN (training target)"](tune_x)
        test_h = {k: m(test_x) for k, m in models.items()}
        decoders = {k: score.fit_decoders(m(dec_x), dec_x, kinds) for k, m in models.items()}
    res = {"pitch": pitch, "n_test_clips": len(test_x), "cells": {"ON": int((kinds == 0).sum()), "OFF": int((kinds == 1).sum())},
           "healthy": {k: {n: v.item() for n, v in score.decode_all(decoders[k], test_h[k], test_x, kinds).items()} for k in models},
           "healthy_models_agree": score.match(test_h["subunit (held-out)"], test_h["LN (training target)"], kinds),
           "targets": {}}
    print("healthy decode", res["healthy"], "models agree r", round(res["healthy_models_agree"]["r"], 3), flush=True)

    def report(name, rates, extra=None):
        row = dict(extra or {})
        for k in models:
            m = score.match(rates, test_h[k], kinds)
            m["identify"] = identify(rates, test_h[k]); m["decode"] = {n: v.item() for n, v in score.decode_all(decoders[k], rates, test_x, kinds).items()}
            row[k] = m
        a, b = row["LN (training target)"], row["subunit (held-out)"]
        print(f"  {name:10s} LN: r {a['r']:.2f} (ON {a['ON']['r']:.2f} OFF {a['OFF']['r']:.2f}) id {a['identify']:.2f} dec {a['decode']['all']:.2f} (ON {a['decode']['ON']:.2f} OFF {a['decode']['OFF']:.2f})"
              f" | held-out: r {b['r']:.2f} (ON {b['ON']['r']:.2f} OFF {b['OFF']['r']:.2f}) id {b['identify']:.2f} dec {b['decode']['all']:.2f} (ON {b['decode']['ON']:.2f} OFF {b['decode']['OFF']:.2f})"
              f" | {extra or ''}", flush=True)
        return row

    for target in ("rgc", "bipolar"):
        print(target, flush=True)
        tr = treated.Treated(models["LN (training target)"], target, pitch=pitch)
        rows = {}
        with torch.no_grad():
            for name, (fn, grid) in GRIDS.items():
                kw = tune(fn, grid, tune_x, tr, tune_h, kinds, pitch)
                fr = fn(test_x, pitch, **kw)
                rows[name] = report(name, tr(fr), {"settings": kw, "lit": fr.mean().item()})
            for tag, label in (("", "learned"), ("_dec", "learned+dec")):
                path = os.path.join(ckpt_dir, f"{target}_p{pitch}{tag}.pt")
                if os.path.exists(path):
                    ck = torch.load(path)
                    enc = E.Learned(pitch); enc.load_state_dict(ck["state"])
                    fr = torch.cat([enc(test_x[i:i + 24]) for i in range(0, len(test_x), 24)])
                    rows[label] = report(label, tr(fr), {"lit": fr.mean().item(), "train_steps": ck["log"][-1]["step"]})
        if target == "rgc":
            rows["oracle"] = report("oracle", oracle(tr, test_h["LN (training target)"], kinds))
        res["targets"][target] = rows
        json.dump(res, open(out, "w"), indent=1)
