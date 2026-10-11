"""Render scene | depth | outline drawing | grid (ours) | grid (raw image), and score boundary F1 per grid size."""
import json
import numpy as np
import cv2
import imageio.v2 as imageio
from scene import Scene, H
from encoder import encode, rgb_edges
from grid import to_grid, raw_grid, boundary_grid
from run_helpers import dots, label

FRAMES = 100
SIZES = [16, 20, 24, 32]
SHOW_N = 24


def f1(active, truth):
    k = np.ones((3, 3), np.uint8)
    if active.sum() == 0 or truth.sum() == 0:
        return 0.0, 0.0, 0.0
    p = (active & cv2.dilate(truth.astype(np.uint8), k).astype(bool)).sum() / active.sum()
    r = (truth & cv2.dilate(active.astype(np.uint8), k).astype(bool)).sum() / truth.sum()
    return p, r, 2 * p * r / (p + r + 1e-9)


scene = Scene(seed=0)
rng = np.random.default_rng(1)
scores = {m: {n: [] for n in SIZES} for m in ("outline", "rgb_edges", "raw_image")}
frames = []
prev, _, _ = scene.frame(-1)
for t in range(FRAMES):
    gray, depth, inst = scene.frame(t)
    drawing, edges, streaks = encode(prev, gray, depth)
    rgbe = rgb_edges(gray).astype(np.float32)
    for n in SIZES:
        truth = boundary_grid(inst, n)
        for m, g in (("outline", to_grid(drawing, n, rng=rng)),
                     ("rgb_edges", to_grid(rgbe, n, rng=rng)),
                     ("raw_image", raw_grid(gray, n))):
            active = g > 0
            p, r, f = f1(active, truth)
            scores[m][n].append((p, r, f, active.mean()))
    if t % 2 == 0:
        dcol = cv2.applyColorMap(cv2.normalize(1 / depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)
        row = np.hstack([
            label((gray * 255).astype(np.uint8), "camera"),
            label(dcol, "lidar depth"),
            label((np.clip(drawing, 0, 1) * 255).astype(np.uint8), "outline + speed streaks"),
            label(dots(to_grid(drawing, SHOW_N, rng=rng)), f"grid {SHOW_N}x{SHOW_N}: outline"),
            label(dots(raw_grid(gray, SHOW_N)), f"grid {SHOW_N}x{SHOW_N}: raw image"),
        ])
        frames.append(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
    if t == 40:
        cv2.imwrite("results/still.png", row)
    prev = gray

imageio.mimsave("results/demo.gif", [cv2.resize(f, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA) for f in frames],
                duration=0.08, loop=0)

summary = {m: {n: dict(zip(("precision", "recall", "f1", "active_fraction"),
                           np.round(np.mean(v, 0), 3).tolist())) for n, v in d.items()} for m, d in scores.items()}
json.dump(summary, open("results/metrics.json", "w"), indent=2)
for m, d in summary.items():
    print(m, {n: (s["f1"], s["active_fraction"]) for n, s in d.items()})
