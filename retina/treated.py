"""Treated retina: what the cells fire when goggles project light onto a retina made light-sensitive by an opsin.

Written from published device and opsin numbers, with no learned parts, so an encoder trained through it
cannot exploit a fitted model's errors. Everything is differentiable.

Chain: binary projector frames at 60 Hz (each lit for the whole 16.6 ms frame) -> optical blur ->
opsin activation (saturating in irradiance, fast on, tau_off 15.8 ms for ChrimsonR) -> spikes.
  target "rgc"      - opsin in the ganglion cells. Every cell, ON or OFF, fires when lit.
  target "bipolar"  - opsin in ON bipolar cells. ON ganglion cells keep their sign; OFF cells get the
                      opposite sign through the AII amacrine pathway; a transient synapse and a surround follow.

Assumptions (not measured for this device): half-saturating irradiance, optical blur, how far a cell's
opsin-bearing membrane extends, the spike threshold and peak rate, the bipolar synapse time course.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import healthy as hl

DT_MS = 1000.0 / 120


class Treated(nn.Module):
    def __init__(self, model, target="rgc", pitch=2, irradiance=1e16, ec50=10 ** 15.5, tau_on_ms=2.0, tau_off_ms=15.8,
                 optical_sigma=0.7, membrane=0.75, threshold=0.08, peak_rate=250.0, expression_sd=0.3, seed=1):
        super().__init__()
        self.target, self.pitch, self.size = target, pitch, model.size
        self.sat = (irradiance / ec50) / (1 + irradiance / ec50)        # opsin activation under a lit pixel
        self.a_on, self.a_off = math.exp(-DT_MS / tau_on_ms), math.exp(-DT_MS / tau_off_ms)
        self.threshold, self.peak = threshold, peak_rate
        r = 3
        ax = torch.arange(-r, r + 1).float()
        g = torch.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * optical_sigma ** 2))
        self.register_buffer("psf", (g / g.sum())[None, None])
        self.register_buffer("kinds", model.kinds)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("expr", torch.exp(expression_sd * torch.randn(len(model.kinds), generator=gen)))
        if target == "rgc":
            self.register_buffer("W", hl.gaussians(model.centres, model.sigma * membrane, model.size))
        else:
            self.register_buffer("W", (model.Wc - hl.SURROUND[1] * model.Ws) / (1 - hl.SURROUND[1]))
            self.register_buffer("bias", model.bias)
            self.register_buffer("gain", model.gain)
            self.syn_tau, self.syn_keep, self.syn_gain = 80.0 / DT_MS, 0.1, 6.0

    def activation(self, frames):
        """Binary (or relaxed) projector frames [b, Tf, h, w] at 60 Hz -> opsin activation [b, T, H*W] at 120 Hz."""
        b, Tf, h, w = frames.shape
        x = frames.repeat_interleave(2, 1).reshape(b * Tf * 2, 1, h, w)
        x = F.interpolate(x, scale_factor=self.pitch, mode="nearest")
        x = F.conv2d(x, self.psf, padding=3).reshape(b, Tf * 2, -1)
        return self.kinetics(self.sat * x)      # lit pixels sit at one irradiance, so blur mixes lit and dark area

    def kinetics(self, drive):
        """Steady-state opsin activation [b, T, ...] -> activation over time (fast rise, slow decay)."""
        a, out = torch.zeros_like(drive[:, 0]), []
        for t in range(drive.shape[1]):
            d = drive[:, t]
            k = torch.where(d > a, self.a_on, self.a_off)
            new = d + (a - d) * k
            out.append(0.5 * (a + new))
            a = new
        return torch.stack(out, 1)

    def per_cell(self, light):
        """Oracle for target "rgc": every cell has its own light, [b, Tf, n] in [0, 1] at 60 Hz."""
        g = self.kinetics(self.sat * light.repeat_interleave(2, 1))
        return (self.peak * self.expr * F.softplus((g - self.threshold) * 20) / 20).clamp(max=hl.RATE_CAP)

    def forward(self, frames):
        a = self.activation(frames)
        if self.target == "rgc":
            g = a @ self.W.T
            return (self.peak * self.expr * F.softplus((g - self.threshold) * 20) / 20).clamp(max=hl.RATE_CAP)
        y = a - (1 - self.syn_keep) * hl.lowpass(a, self.syn_tau)       # transient bipolar-to-ganglion synapse
        g = self.syn_gain * self.expr * (y @ self.W.T)
        g = torch.where(self.kinds == 0, g, -g)                         # OFF cells: sign inverted by AII amacrine cells
        return (self.gain * F.softplus(g + self.bias)).clamp(max=hl.RATE_CAP)
