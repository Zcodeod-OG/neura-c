"""Scores for how close treated-retina firing is to healthy firing. Rates are [b, T, n]; the first WARM bins are dropped."""
import torch

WARM = 36       # 300 ms: filters and the encoder's memory fill up


def match(treated, healthy, kinds):
    """Per cell type: correlation over cells and time, and the fraction of healthy variance explained."""
    out = {}
    for ti, name in enumerate(["ON", "OFF"]):
        a, b = treated[:, WARM:, kinds == ti].flatten(), healthy[:, WARM:, kinds == ti].flatten()
        out[name] = {"r": torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
                     "fve": (1 - (a - b).pow(2).mean() / b.var()).item()}
    out["r"] = (out["ON"]["r"] + out["OFF"]["r"]) / 2
    out["fve"] = (out["ON"]["fve"] + out["OFF"]["fve"]) / 2
    return out


def loss(treated, healthy, kinds):
    """Training loss: squared error normalised by the healthy variance of each cell type."""
    total = 0.0
    for ti in range(2):
        a, b = treated[:, WARM:, kinds == ti], healthy[:, WARM:, kinds == ti]
        total = total + (a - b).pow(2).mean() / b.var().detach()
    return total / 2
