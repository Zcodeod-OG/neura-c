"""Fit linear-nonlinear (LN) models to recorded primate ganglion cells.

Data: 4 cells responding to full-field binary white noise at ~120 Hz for 20 minutes, provided by
EJ Chichilnisky and distributed with the Pillow lab GLM tutorial
(github.com/pillowlab/GLMspiketraintutorial_python, folder data_RGCs). Cells 0-1 are OFF, 2-3 are ON
(checked here from the sign of the spike-triggered average).

Each cell gets a temporal filter (LAGS bins) and a softplus output, fitted by Poisson maximum likelihood on
the first 80% of the recording and scored on the last 20%. The fits give the temporal part of the healthy
retina model in healthy.py.

Usage: python3 fit_ln.py <data_RGCs dir> [out_json]
"""
import json
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

LAGS = 30          # 250 ms at 120 Hz
SMOOTH = 1e-2      # penalty on the second difference of the filter


def load(d):
    stim = sio.loadmat(f"{d}/Stim.mat")["Stim"].flatten()
    t = sio.loadmat(f"{d}/stimtimes.mat")["stimtimes"].flatten()
    sp = [c.flatten() for c in sio.loadmat(f"{d}/SpTimes.mat")["SpTimes"].flatten()]
    dt = float(np.median(np.diff(t)))
    edges = np.concatenate([t - dt, t[-1:]])
    counts = np.stack([np.histogram(s, edges)[0] for s in sp], 1).astype(np.float32)
    return stim.astype(np.float32), counts, dt


def design(stim):
    pad = np.concatenate([np.zeros(LAGS - 1, np.float32), stim])
    return np.stack([pad[i:i + len(stim)] for i in range(LAGS)], 1)[:, ::-1].copy()   # column j = stimulus j bins ago


def fit(X, y, dt, steps=600):
    X, y = torch.tensor(X), torch.tensor(y)
    k = torch.zeros(LAGS, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    g = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([k, b, g], max_iter=steps, line_search_fn="strong_wolfe")

    def rate():
        return F.softplus(g) * F.softplus(X @ k + b)       # spikes per second

    def closure():
        opt.zero_grad()
        r = rate() * dt + 1e-8
        loss = (r - y * torch.log(r)).mean() + SMOOTH * (k[2:] - 2 * k[1:-1] + k[:-2]).pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return k.detach(), b.item(), F.softplus(g).item()


if __name__ == "__main__":
    stim, counts, dt = load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "ln_fits.json"
    X = design(stim)
    n = int(0.8 * len(stim))
    res = {"dt": dt, "lags": LAGS, "stim_contrast": float(np.abs(stim).max()), "cells": []}
    box = np.ones(6, np.float32) / 6                                      # 50 ms window for scoring
    for c in range(counts.shape[1]):
        k, b, g = fit(X[:n], counts[:n, c], dt)
        with torch.no_grad():
            pred = (g * F.softplus(torch.tensor(X[n:]) @ k + b)).numpy()
        obs = np.convolve(counts[n:, c] / dt, box, "same")
        r = float(np.corrcoef(np.convolve(pred, box, "same"), obs)[0, 1])
        sta = (X[:n] * counts[:n, c:c + 1]).sum(0) / counts[:n, c].sum()
        kind = "ON" if sta[np.abs(sta).argmax()] > 0 else "OFF"
        res["cells"].append({"cell": c, "type": kind, "filter": k.tolist(), "bias": b, "gain": g,
                             "mean_rate_hz": float(counts[:, c].mean() / dt), "pred_mean_rate_hz": float(pred.mean()),
                             "heldout_r_50ms": r})
        print(c, kind, "mean rate", round(counts[:, c].mean() / dt, 1), "Hz, held-out r (50 ms)", round(r, 3),
              "peak lag", int(np.abs(k.numpy()).argmax()), flush=True)
    json.dump(res, open(out, "w"))
