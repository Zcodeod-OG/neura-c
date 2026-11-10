"""Train the learned encoder through the treated-retina simulator to reproduce healthy firing.

Target is HealthyLN only; HealthySubunit is kept for evaluate.py.

Usage: python3 train.py <clips.npz> <rgc|bipolar> <out.pt> [pitch] [steps] [threads]
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

if __name__ == "__main__":
    npz, target, out = sys.argv[1:4]
    pitch = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 4)
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
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        x = clips.sample(12)
        with torch.no_grad():
            target_rates = h(x)
        noise = max(0.0, 1.0 - step / (0.6 * steps))            # logistic noise on the logits, annealed to zero
        l = score.loss(tr(enc(x, noise)), target_rates, h.kinds)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                fr = enc(val)
                m = score.match(tr(fr), val_h, h.kinds)
            log.append({"step": step, "loss": l.item(), "val_r": m["r"], "val_fve": m["fve"], "lit": fr.mean().item()})
            print(step, f"loss {l.item():.3f} val r {m['r']:.3f} (ON {m['ON']['r']:.2f} OFF {m['OFF']['r']:.2f}) "
                  f"fve {m['fve']:.2f} lit {fr.mean().item():.2f} {time.time()-t0:.0f}s", flush=True)
            torch.save({"state": enc.state_dict(), "pitch": pitch, "target": target, "log": log}, out)
