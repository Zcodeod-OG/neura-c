"""Procedural 3D objects seen by a head-height depth camera.

Ten everyday object classes a walking user would meet (person, dog, chair, table, car, stairs, doorway,
signpost, bin, sofa) are built from signed-distance primitives with random proportions and pose. Each
sample is ray-marched from a camera 1.6 m above the ground, at a random distance (scaled by object size),
viewing angle all the way round the object, and small sideways offset. The ground plane is rendered and
then removed, as a lidar ground-plane fit would.

Each sample gives:
  nearness  (64x64)  clip(1.5 / depth, 0, 1) on the object, 0 elsewhere
  edges     (64x64)  depth-discontinuity strength (Sobel of nearness): silhouette plus internal depth steps
  thumb     (32x32)  a shaded rendering, for feedback in the human reading test

Usage: python3 scenes3d.py <out.pt> [per_class_train] [per_class_test]
"""
import math
import sys
import time
import torch
import torch.nn.functional as F

CLASSES = ["Person", "Dog", "Chair", "Table", "Car", "Stairs", "Doorway", "Signpost", "Bin", "Sofa"]
RES, THUMB, FOV, CAM_H, STEPS = 64, 32, math.radians(60), 1.6, 96


def U(b, lo, hi):
    return lo + (hi - lo) * torch.rand(b, 1)


def v3(x, y, z):
    """Stack per-sample [B,1] (or float) coordinates into [B,1,3]."""
    b = next((t.shape[0] for t in (x, y, z) if torch.is_tensor(t)), 1)
    f = lambda t: t if torch.is_tensor(t) else torch.full((b, 1), float(t))
    return torch.stack([f(x), f(y), f(z)], -1)


def box(p, c, h):
    q = (p - c).abs() - h
    return q.clamp(min=0).norm(dim=-1) + q.amax(-1).clamp(max=0)


def capsule(p, a, b, r):
    pa, ba = p - a, b - a
    t = ((pa * ba).sum(-1, keepdim=True) / (ba * ba).sum(-1, keepdim=True)).clamp(0, 1)
    return (pa - ba * t).norm(dim=-1) - r


def sphere(p, c, r):
    return (p - c).norm(dim=-1) - r


def cyl(p, c, r, h, axis=1):
    q = p - c
    ax = [0, 1, 2]
    ax.remove(axis)
    d = torch.stack([q[..., ax].norm(dim=-1) - r, q[..., axis].abs() - h], -1)
    return d.amax(-1).clamp(max=0) + d.clamp(min=0).norm(dim=-1)


def union(*ds):
    return torch.stack(ds).amin(0)


# ---- object classes: each returns (sdf(p) for p [B,N,3] in object space, size in metres) ----
def person(b):
    H = U(b, 1.5, 1.9); s = U(b, -0.25, 0.25); a = U(b, -0.3, 0.3)
    hip, sh = 0.5 * H, 0.8 * H
    def f(p):
        return union(
            capsule(p, v3(0.1, 0.05, s), v3(0.1, hip, 0), 0.07), capsule(p, v3(-0.1, 0.05, -s), v3(-0.1, hip, 0), 0.07),
            capsule(p, v3(0, hip, 0), v3(0, sh, 0), 0.15), sphere(p, v3(0, 0.91 * H, 0), 0.1),
            capsule(p, v3(0.22, sh, 0), v3(0.28, hip + 0.05, a), 0.05), capsule(p, v3(-0.22, sh, 0), v3(-0.28, hip + 0.05, -a), 0.05))
    return f, H


def dog(b):
    L = U(b, 0.6, 1.0); hb = U(b, 0.35, 0.55); g = U(b, -0.08, 0.08)
    def f(p):
        legs = [capsule(p, v3(sx * 0.4 * L, hb, sz * 0.1), v3(sx * 0.4 * L + g * sx, 0.03, sz * 0.1), 0.045)
                for sx in (-1, 1) for sz in (-1, 1)]
        return union(capsule(p, v3(-L / 2, hb, 0), v3(L / 2, hb, 0), 0.13), sphere(p, v3(L / 2 + 0.12, hb + 0.16, 0), 0.11),
                     capsule(p, v3(-L / 2, hb, 0), v3(-L / 2 - 0.2, hb + 0.18, 0), 0.03), *legs)
    return f, L + 0.3


def chair(b):
    w = U(b, 0.2, 0.26); sh = U(b, 0.42, 0.5); bh = U(b, 0.35, 0.5)
    def f(p):
        legs = [box(p, v3(sx * (w - 0.02), sh / 2, sz * (w - 0.02)), v3(0.02, sh / 2, 0.02)) for sx in (-1, 1) for sz in (-1, 1)]
        return union(box(p, v3(0, sh, 0), v3(w, 0.025, w)), box(p, v3(0, sh + bh / 2, -w + 0.02), v3(w, bh / 2, 0.02)), *legs)
    return f, sh + bh


def table(b):
    w = U(b, 0.45, 0.8); d = U(b, 0.3, 0.5); h = U(b, 0.68, 0.78)
    def f(p):
        legs = [box(p, v3(sx * (w - 0.05), h / 2, sz * (d - 0.05)), v3(0.03, h / 2, 0.03)) for sx in (-1, 1) for sz in (-1, 1)]
        return union(box(p, v3(0, h, 0), v3(w, 0.025, d)), *legs)
    return f, 2 * w


def car(b):
    L = U(b, 1.8, 2.3); W = U(b, 0.8, 0.95); c = U(b, -0.4, 0.1)
    def f(p):
        wheels = [cyl(p, v3(sx * 0.65 * L, 0.33, sz * W), 0.33, 0.12, axis=2) for sx in (-1, 1) for sz in (-1, 1)]
        return union(box(p, v3(0, 0.6, 0), v3(L, 0.3, W)), box(p, v3(c, 1.12, 0), v3(0.55 * L, 0.25, 0.9 * W)), *wheels)
    return f, 1.4 * L


def stairs(b):
    n = 5; w = U(b, 0.5, 0.9); rise = U(b, 0.14, 0.2); run = U(b, 0.25, 0.32)
    def f(p):
        steps = [box(p, v3(0, (i + 1) * rise / 2, (n / 2 - i - 0.5) * run), v3(w, (i + 1) * rise / 2, run / 2)) for i in range(n)]
        return union(*steps)
    return f, n * run * 1.6


def doorway(b):
    ww = U(b, 1.1, 1.6); dw = U(b, 0.4, 0.5); dh = U(b, 0.95, 1.05)
    def f(p):
        wall = box(p, v3(0, 1.2, 0), v3(ww, 1.2, 0.1))
        hole = box(p, v3(0, dh, 0), v3(dw, dh, 0.3))
        return torch.maximum(wall, -hole)
    return f, 2 * ww


def signpost(b):
    h = U(b, 1.9, 2.3); r = U(b, 0.25, 0.35); sq = torch.rand(b, 1) < 0.5
    def f(p):
        pole = cyl(p, v3(0, h / 2, 0), 0.04, h / 2)
        disc = cyl(p, v3(0, h, 0.05), r, 0.02, axis=2)
        plate = box(p, v3(0, h, 0.05), v3(r, r * 0.7, 0.02))
        return union(pole, torch.where(sq, plate, disc))
    return f, h


def bin_(b):
    r = U(b, 0.22, 0.32); h = U(b, 0.35, 0.5)
    def f(p):
        return union(cyl(p, v3(0, h, 0), r, h), cyl(p, v3(0, 2 * h + 0.02, 0), r + 0.03, 0.025))
    return f, 2 * h + 0.1


def sofa(b):
    w = U(b, 0.8, 1.1); d = U(b, 0.4, 0.5)
    def f(p):
        return union(box(p, v3(0, 0.22, 0), v3(w, 0.2, d)), box(p, v3(0, 0.62, -d + 0.1), v3(w, 0.25, 0.1)),
                     box(p, v3(w - 0.1, 0.5, 0), v3(0.1, 0.12, d)), box(p, v3(-w + 0.1, 0.5, 0), v3(0.1, 0.12, d)))
    return f, 2 * w


BUILDERS = [person, dog, chair, table, car, stairs, doorway, signpost, bin_, sofa]


def render(cls, b):
    sdf, size = BUILDERS[cls](b)
    yaw = U(b, 0, 2 * math.pi)
    dist = size / (2 * math.tan(FOV / 2) * U(b, 0.45, 0.85))   # object spans 45-85% of the view
    look_y = (size * 0.35).clamp(0.3, 1.0)
    # camera on a circle around the object at head height; aim at the object with a small sideways offset
    cam = torch.cat([torch.zeros(b, 1), torch.full((b, 1), CAM_H), dist], 1)
    side = U(b, -0.12, 0.12) * dist
    target = torch.cat([side, look_y, torch.zeros(b, 1)], 1)
    fwd = F.normalize(target - cam, dim=1)
    right = F.normalize(torch.cross(fwd, torch.tensor([[0.0, 1.0, 0.0]]).expand(b, 3), dim=1), dim=1)
    up = torch.cross(right, fwd, dim=1)
    t = torch.linspace(-1, 1, RES) * math.tan(FOV / 2)
    yy, xx = torch.meshgrid(-t, t, indexing="ij")
    dirs = F.normalize(fwd[:, None] + xx.reshape(1, -1, 1) * right[:, None] + yy.reshape(1, -1, 1) * up[:, None], dim=-1)
    c, s = torch.cos(yaw)[:, :, None], torch.sin(yaw)[:, :, None]

    def obj(p):  # world -> object space (rotate about the vertical axis)
        q = torch.stack([c[..., 0] * p[..., 0] - s[..., 0] * p[..., 2], p[..., 1], s[..., 0] * p[..., 0] + c[..., 0] * p[..., 2]], -1)
        return sdf(q)

    o = cam[:, None].expand_as(dirs)
    depth = torch.full(dirs.shape[:2], 0.3)
    for _ in range(STEPS):
        d = obj(o + dirs * depth[..., None])
        depth = depth + d.clamp(min=0) * 0.9
    hit = obj(o + dirs * depth[..., None]) < 2e-3 * depth.clamp(min=1)
    far = 30.0
    t_ground = torch.where(dirs[..., 1] < -1e-4, -CAM_H / dirs[..., 1].clamp(max=-1e-4), torch.full_like(depth, far))
    obj_mask = hit & (depth < t_ground)
    zdepth = torch.where(obj_mask, depth, t_ground.clamp(max=far))
    # shading for the thumbnail
    p = o + dirs * zdepth[..., None]
    e = 1e-3
    n = torch.stack([obj(p + torch.tensor([e, 0, 0])) - obj(p - torch.tensor([e, 0, 0])),
                     obj(p + torch.tensor([0, e, 0])) - obj(p - torch.tensor([0, e, 0])),
                     obj(p + torch.tensor([0, 0, e])) - obj(p - torch.tensor([0, 0, e]))], -1)
    light = F.normalize(torch.tensor([0.4, 0.8, 0.5]) - 0 * n, dim=-1)
    shade = (F.normalize(n, dim=-1) * light).sum(-1).clamp(min=0) * 0.75 + 0.25
    ground_shade = (0.15 + 0.15 * (1.5 / zdepth).clamp(0, 1)) * (t_ground < far)
    img = torch.where(obj_mask, shade, ground_shade)
    near = torch.where(obj_mask, (1.5 / depth).clamp(0, 1), torch.zeros_like(depth))
    near = near.reshape(b, 1, RES, RES)
    sob = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
    gx, gy = F.conv2d(near, sob, padding=1), F.conv2d(near, sob.transpose(2, 3), padding=1)
    edges = (torch.sqrt(gx ** 2 + gy ** 2) / 2).clamp(0, 1)
    thumb = F.adaptive_avg_pool2d(img.reshape(b, 1, RES, RES), THUMB)
    return torch.cat([near, edges], 1), thumb[:, 0], obj_mask.float().mean(1)


def make(per_class, batch=200):
    X, Y, T = [], [], []
    for c in range(len(CLASSES)):
        done = 0
        while done < per_class:
            b = min(batch, per_class - done)
            x, th, cover = render(c, b)
            keep = cover > 0.004            # drop the rare view where the object is out of frame
            X.append(x[keep].half()); T.append((th[keep] * 255).byte()); Y.append(torch.full((int(keep.sum()),), c))
            done += int(keep.sum())
    return torch.cat(X), torch.cat(Y), torch.cat(T)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = sys.argv[1]
    ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    nte = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    t0 = time.time()
    Xtr, Ytr, Ttr = make(ntr)
    print("train", Xtr.shape, f"{time.time()-t0:.0f}s", flush=True)
    Xte, Yte, Tte = make(nte)
    print("test", Xte.shape, f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"classes": CLASSES, "Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte, "Ttr": Ttr, "Tte": Tte}, out)
