"""Moving 3D objects seen by a fixed head-height depth camera (renderer from scenes3d.py).

A person, dog or car either approaches the viewer, moves away, crosses left, crosses right or stands
still, at realistic speeds (person 1-2 m/s, dog 1.5-4 m/s, car 3-7 m/s). Each sequence is 6 frames at
5 Hz (1 s), the window a walker has to notice something coming; the display itself can refresh faster.

Usage: python3 motion3d.py <out.pt> [per_combo_train] [per_combo_test]
"""
import math
import sys
import time
import torch
import scenes3d as s3

OBJECTS = [("Person", 0, (1.0, 2.0)), ("Dog", 1, (1.5, 4.0)), ("Car", 4, (3.0, 7.0))]
MOTIONS = ["Approaching", "Moving away", "Crossing left", "Crossing right", "Standing still"]
T, DT = 6, 0.2


def U(b, lo, hi):
    return lo + (hi - lo) * torch.rand(b)


def sequence(obj, motion, b):
    _, builder, (vlo, vhi) = OBJECTS[obj]
    sdf, size = s3.BUILDERS[builder](b)
    yaw = s3.U(b, 0, 2 * math.pi)
    k = 1.6 if obj == 2 else 1.0                       # cars are seen from further away
    v = U(b, vlo, vhi)
    travel = v * DT * (T - 1)
    if motion == 0:
        z0 = U(b, 4.5, 7.0) * k; z1 = z0 - travel; x0 = x1 = U(b, -0.4, 0.4)
    elif motion == 1:
        z0 = U(b, 2.5, 4.5) * k; z1 = z0 + travel; x0 = x1 = U(b, -0.4, 0.4)
    elif motion in (2, 3):
        z0 = z1 = U(b, 3.0, 5.5) * k; off = U(b, -0.3, 0.3)
        sgn = 1 if motion == 2 else -1                 # crossing left: x goes from right (+) to left (-)
        x0, x1 = off + sgn * travel / 2, off - sgn * travel / 2
    else:
        z0 = z1 = U(b, 2.5, 6.0) * k; x0 = x1 = U(b, -1.0, 1.0)
    z1 = z1.clamp(min=1.2 * k)
    fwd = torch.nn.functional.normalize(torch.tensor([[0.0, 0.7 - s3.CAM_H, -6.0]]), dim=1).expand(b, 3)
    frames, thumb = [], None
    for t in range(T):
        a = t / (T - 1)
        x, z = x0 + (x1 - x0) * a, z0 + (z1 - z0) * a
        # object stays at the origin; the camera moves the opposite way (same image, simpler renderer)
        cam = torch.stack([-x, torch.full((b,), s3.CAM_H), z], 1)
        img, th, _ = s3.view(sdf, yaw, cam, fwd)
        frames.append(img)
        thumb = th
    return torch.stack(frames, 1), thumb


def make(per_combo, batch=100):
    X, O, M, Th = [], [], [], []
    for o in range(len(OBJECTS)):
        for m in range(len(MOTIONS)):
            done = 0
            while done < per_combo:
                b = min(batch, per_combo - done)
                x, th = sequence(o, m, b)
                X.append(x.half()); Th.append((th * 255).byte())
                O.append(torch.full((b,), o)); M.append(torch.full((b,), m))
                done += b
    return torch.cat(X), torch.cat(O), torch.cat(M), torch.cat(Th)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(4)
    ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    nte = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    t0 = time.time()
    Xtr, Otr, Mtr, Ttr = make(ntr)
    print("train", tuple(Xtr.shape), f"{time.time()-t0:.0f}s", flush=True)
    Xte, Ote, Mte, Tte = make(nte)
    print("test", tuple(Xte.shape), f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"objects": [o[0] for o in OBJECTS], "motions": MOTIONS, "Xtr": Xtr, "Otr": Otr, "Mtr": Mtr,
                "Xte": Xte, "Ote": Ote, "Mte": Mte, "Ttr": Ttr, "Tte": Tte}, sys.argv[1])
