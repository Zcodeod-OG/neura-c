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


def loss(treated, healthy, kinds, mse_weight=0.2):
    """Training loss per cell type: one minus the correlation with healthy firing, plus a smaller squared-error
    term (normalised by the healthy variance) that sets the overall firing level. Squared error alone makes
    "all pixels off" a trap, because early random flashes fire the cells more wrongly than silence does."""
    total = 0.0
    for ti in range(2):
        a, b = treated[:, WARM:, kinds == ti], healthy[:, WARM:, kinds == ti]
        ac, bc = a - a.mean(), b - b.mean()
        r = (ac * bc).mean() / (ac.pow(2).mean().sqrt() * bc.pow(2).mean().sqrt() + 1e-3)
        total = total + (1 - r) + mse_weight * (a - b).pow(2).mean() / b.var().detach()
    return total / 2
