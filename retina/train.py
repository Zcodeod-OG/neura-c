"""Train the learned encoder through the treated-retina simulator to reproduce healthy firing.

Target is HealthyLN only; HealthySubunit is kept for evaluate.py.

Training from scratch can settle on "all pixels off" (early random flashes fire the cells more wrongly than
silence does), and can end below a hand-designed encoder. So every run starts with WARM_STEPS of imitating a
hand-designed encoder, then trains through the simulator.

With dec_weight > 0 the loss also asks that linear decoders fitted on healthy firing, one reading ON cells
and one reading OFF cells, can rebuild what changed in the movie from the treated firing (score.decode_all).

Usage: python3 train.py <clips.npz> <rgc|bipolar> <out.pt> [pitch] [steps] [threads] [dec_weight] [membrane] [map]
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
MAP_RADIUS = 1.2
WARM_START = {"rgc": lambda x, p: encoders.event(x, p, delay=2, level=0.1, hold=1),
              "bipolar": lambda x, p: encoders.dither(x, p, delay=1, gain=2)}

if __name__ == "__main__":
    npz, target, out = sys.argv[1:4]
    pitch = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 4)
    dec_weight = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    membrane = float(sys.argv[8]) if len(sys.argv) > 8 else 0.75     # opsin spread; about 0.25 for a soma-targeted opsin
    use_map = len(sys.argv) > 9 and sys.argv[9] == "map"          # give the encoder the ON/OFF cell positions
    torch.manual_seed(0)
    torch.set_flush_denormal(True)          # decaying opsin activation otherwise reaches very slow tiny numbers
    clips = data.Clips(npz, "train", seed=0)
    val = data.Clips(npz, "train", seed=99).sample(24)
    h = healthy.HealthyLN()
    tr = treated.Treated(h, target, pitch=pitch, membrane=membrane)
    enc = encoders.Learned(pitch, cell_maps=encoders.cell_maps(h.centres, h.kinds) if use_map else None)
    warm = WARM_START[target]
    if use_map:
        sign = encoders.sign_map(h.centres, h.kinds, pitch, MAP_RADIUS)
        warm = lambda x, p: encoders.mapped(x, p, sign)
    opt = torch.optim.Adam(enc.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    with torch.no_grad():
        val_h = h(val)
        dec_x = data.Clips(npz, "train", seed=6).sample(160)
        decoder = score.fit_decoders(h(dec_x), dec_x, h.kinds)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        x = clips.sample(12)
        with torch.no_grad():
            target_rates = h(x)
        if step <= WARM_STEPS:
            l = torch.nn.functional.binary_cross_entropy_with_logits(enc.logits(x), warm(x, pitch))
        else:
            done = (step - WARM_STEPS) / (steps - WARM_STEPS)
            noise = max(0.0, 1.0 - done / 0.6)                  # logistic noise on the logits, annealed to zero
            rates = tr(enc(x, noise))
            l = score.loss(rates, target_rates, h.kinds, worst=0.5 if use_map else 0.0)
            if dec_weight > 0:
                l = l + dec_weight * (1 - score.decode_all(decoder, rates, x, h.kinds)["split"])
        opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                fr = enc(val)
                vr = tr(fr)
                m = score.match(vr, val_h, h.kinds)
                m["dec"] = score.decode_all(decoder, vr, val, h.kinds)["split"].item()
            log.append({"step": step, "loss": l.item(), "val_r": m["r"], "val_dec": m["dec"], "lit": fr.mean().item()})
            print(step, f"loss {l.item():.3f} val r {m['r']:.3f} (ON {m['ON']['r']:.2f} OFF {m['OFF']['r']:.2f}) "
                  f"dec {m['dec']:.2f} lit {fr.mean().item():.2f} {time.time()-t0:.0f}s", flush=True)
            torch.save({"state": enc.state_dict(), "pitch": pitch, "target": target, "dec_weight": dec_weight, "membrane": membrane, "map": use_map, "log": log}, out)
