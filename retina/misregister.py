"""How much does the map-aware encoder lose when the projected pattern is shifted on the retina?

The encoder plans its pattern for where it believes the cells are. Here the pattern lands `shift` pixels
(10 um each at pitch 1) to the side, as after an eye movement or an alignment error.

Usage: python3 misregister.py <clips.npz> <ckpt_dir> <out_json>
"""
import json
import sys
import torch
import data
import encoders as E
import healthy
import score
import treated

if __name__ == "__main__":
    npz, ckpt_dir, out = sys.argv[1:4]
    torch.manual_seed(0)
    torch.set_num_threads(4)
    h = healthy.HealthyLN(); k = h.kinds
    tx = data.Clips(npz, "test", seed=123).sample(120, augment=False)
    dx = data.Clips(npz, "train", seed=6).sample(160)
    tr = treated.Treated(h, "rgc", pitch=1, membrane=0.25)
    res = {}
    with torch.no_grad():
        ht = h(tx); dec = score.fit_decoders(h(dx), dx, k)
        frames = {}
        for name, path, use_map in [("learned+map", "rgc_p1_dec_map_m25.pt", True), ("learned, no map", "rgc_p1_dec_m25.pt", False)]:
            enc = E.Learned(1, cell_maps=E.cell_maps(h.centres, k) if use_map else None)
            enc.load_state_dict(torch.load(f"{ckpt_dir}/{path}")["state"])
            frames[name] = torch.cat([enc(tx[i:i + 24]) for i in range(0, len(tx), 24)])
        frames["mapped (hand-designed)"] = E.mapped(tx, 1, E.sign_map(h.centres, k, 1, 1.6), 0, 1.0, 0.08)
        for name, fr in frames.items():
            res[name] = {}
            for shift in (0, 1, 2, 3, 5):
                rt = tr(fr.roll(shift, -1))
                m = score.match(rt, ht, k); d = score.decode_all(dec, rt, tx, k)
                res[name][shift * 10] = {"r": m["r"], "decode": d["all"].item(), "decode_ON": d["ON"].item(), "decode_OFF": d["OFF"].item()}
                print(f"{name:24s} shift {shift*10:3d} um  r {m['r']:.2f}  decode {d['all'].item():.2f} (ON {d['ON'].item():.2f} OFF {d['OFF'].item():.2f})", flush=True)
    json.dump(res, open(out, "w"), indent=1)
