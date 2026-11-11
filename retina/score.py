"""Scores for how close treated-retina firing is to healthy firing. Rates are [b, T, n]; the first WARM bins are dropped."""
import torch
import torch.nn.functional as F

WARM = 36       # 300 ms: filters and the encoder's memory fill up
LAGS, SHIFT, RIDGE = 6, 4, 10.0     # decoder: 6 bins of firing predict the frame 4 bins before the newest


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


# A linear decoder fitted on healthy firing stands in for a brain that expects the healthy code.
def features(rates):
    """[b, T, n] -> rows of LAGS consecutive bins, [b * (T - WARM - LAGS + 1), n * LAGS]."""
    r = rates[:, WARM:]
    return r.unfold(1, LAGS, 1).flatten(2).flatten(0, 1)


def frames_for(movie):
    m = F.avg_pool2d(movie[:, WARM:], 2)
    T = m.shape[1]
    return m[:, LAGS - 1 - SHIFT:T - SHIFT].flatten(2).flatten(0, 1)


def fit_decoder(rates, movie):
    X, Y = features(rates), frames_for(movie)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    A = Xn.T @ Xn + RIDGE * len(Xn) * torch.eye(Xn.shape[1])
    W = torch.linalg.solve(A, Xn.T @ (Y - Y.mean(0)))
    return mu, sd, W, Y.mean(0)


def decode_corr(dec, rates, movie):
    """Correlation between the decoded movie and the real one (differentiable)."""
    mu, sd, W, y0 = dec
    pred = (((features(rates) - mu) / sd) @ W + y0).flatten()
    true = frames_for(movie).flatten()
    p, t = pred - pred.mean(), true - true.mean()
    return (p * t).mean() / (p.pow(2).mean().sqrt() * t.pow(2).mean().sqrt() + 1e-6)


def decode_r(dec, rates, movie):
    return decode_corr(dec, rates, movie).item()
