"""Train the learned encoder through the treated-retina simulator to reproduce healthy firing.

Target is HealthyLN only; HealthySubunit is kept for evaluate.py.

Training from scratch can settle on "all pixels off" (early random flashes fire the cells more wrongly than
silence does), and can end below a hand-designed encoder. So every run starts with WARM_STEPS of imitating a
hand-designed encoder, then trains through the simulator.

With dec_weight > 0 the loss also asks that a linear decoder fitted on healthy firing can rebuild the movie
from the treated firing (score.decode_corr).

Usage: python3 train.py <clips.npz> <rgc|bipolar> <out.pt> [pitch] [steps] [threads] [dec_weight]
"""
import json
import sys
import time
import torch
import data
import encoders
import healthy
import score
import treated

WARM_STEPS = 80
WARM_START = {"rgc": lambda x, p: encoders.event(x, p, delay=2, level=0.1, hold=1),
              "bipolar": lambda x, p: encoders.dither(x, p, delay=1, gain=2)}

if __name__ == "__main__":
    npz, target, out = sys.argv[1:4]
    pitch = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 4)
    dec_weight = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    torch.manual_seed(0)
    torch.set_flush_denormal(True)          # decaying opsin activation otherwise reaches very slow tiny numbers
    clips = data.Clips(npz, "train", seed=0)
    val = data.Clips(npz, "train", seed=99).sample(24)
    h = healthy.HealthyLN()
    tr = treated.Treated(h, target, pitch=pitch)
    enc = encoders.Learned(pitch)
    opt = torch.optim.Adam(enc.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    with torch.no_grad():
        val_h = h(val)
        dec_x = data.Clips(npz, "train", seed=6).sample(160)
        decoder = score.fit_decoder(h(dec_x), dec_x)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        x = clips.sample(12)
        with torch.no_grad():
            target_rates = h(x)
        if step <= WARM_STEPS:
            l = torch.nn.functional.binary_cross_entropy_with_logits(enc.logits(x), WARM_START[target](x, pitch))
        else:
            done = (step - WARM_STEPS) / (steps - WARM_STEPS)
            noise = max(0.0, 1.0 - done / 0.6)                  # logistic noise on the logits, annealed to zero
            rates = tr(enc(x, noise))
            l = score.loss(rates, target_rates, h.kinds)
            if dec_weight > 0:
                l = l + dec_weight * (1 - score.decode_corr(decoder, rates, x))
        opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                fr = enc(val)
                vr = tr(fr)
                m = score.match(vr, val_h, h.kinds)
                m["dec"] = score.decode_r(decoder, vr, val)
            log.append({"step": step, "loss": l.item(), "val_r": m["r"], "val_dec": m["dec"], "lit": fr.mean().item()})
            print(step, f"loss {l.item():.3f} val r {m['r']:.3f} (ON {m['ON']['r']:.2f} OFF {m['OFF']['r']:.2f}) "
                  f"dec {m['dec']:.2f} lit {fr.mean().item():.2f} {time.time()-t0:.0f}s", flush=True)
            torch.save({"state": enc.state_dict(), "pitch": pitch, "target": target, "dec_weight": dec_weight, "log": log}, out)
