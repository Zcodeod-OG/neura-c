"""Run the outline + grid pipeline on a real RGB-D sequence (color/ and depth/ PNG folders, depth in mm).

Usage: python3 run_real.py <sequence_dir> [out_prefix]
No object labels in this data, so we report electrode usage, frame-to-frame flicker and
how many electrodes speed streaks light in a static scene (they should not, once camera motion is removed).
"""
import json
import os
import sys
import numpy as np
import cv2
import imageio.v2 as imageio
from encoder import encode_real, rgb_edges
from grid import to_grid, raw_grid, GridSmoother
from run_helpers import dots, label

S = 256
SIZES = [16, 20, 24, 32]
SHOW_N = 24
seq = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "real"
cs, ds = sorted(os.listdir(f"{seq}/color")), sorted(os.listdir(f"{seq}/depth"))


def load(i):
    c = cv2.imread(f"{seq}/color/{cs[i]}")
    d = cv2.imread(f"{seq}/depth/{ds[i]}", -1).astype(np.float32) / 1000.0
    h, w = d.shape
    x0 = (w - h) // 2
    c = cv2.resize(c[:, x0:x0 + h], (S, S), interpolation=cv2.INTER_AREA)
    d = cv2.resize(d[:, x0:x0 + h], (S, S), interpolation=cv2.INTER_NEAREST)
    return c, cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, d


# Kinect v1 intrinsics (fx=fy=525, cx=319.5, cy=239.5), adjusted for the centre crop and resize
_sc = S / 480
K = np.array([[525 * _sc, 0, (319.5 - 80) * _sc], [0, 525 * _sc, 239.5 * _sc], [0, 0, 1]], np.float64)
rng = np.random.default_rng(1)
methods = ("outline", "outline_smoothed", "rgb_edges", "raw_image")
smoothers = {n: GridSmoother() for n in SIZES}
active = {m: {n: [] for n in SIZES} for m in methods}
flicker = {m: {n: [] for n in SIZES} for m in methods}
last = {}
streak_cells = {"depth_ego_motion": [], "homography": [], "none": []}  # % of 24x24 electrodes lit by false streaks
frames = []
_, prev, prev_depth = load(0)
N = len(cs)
for i in range(1, N):
    color, gray, depth = load(i)
    drawing, edges, streaks = encode_real(prev, gray, depth, prev_depth_m=prev_depth, K=K)
    _, _, streaks_h = encode_real(prev, gray, depth)
    _, _, streaks_none = encode_real(prev, gray, depth, compensate=False)
    for k, s_ in (("depth_ego_motion", streaks), ("homography", streaks_h), ("none", streaks_none)):
        streak_cells[k].append((to_grid(s_, SHOW_N) > 0).mean())
    rgbe = rgb_edges(gray).astype(np.float32)
    for n in SIZES:
        g_out = to_grid(drawing, n, rng=rng)
        g_smooth = smoothers[n](g_out)
        if n == SHOW_N:
            show_smooth = g_smooth
        for m, g in (("outline", g_out), ("outline_smoothed", g_smooth), ("rgb_edges", to_grid(rgbe, n, rng=rng)),
                     ("raw_image", raw_grid(gray, n))):
            a = g > 0
            active[m][n].append(a.mean())
            if (m, n) in last:
                flicker[m][n].append((a != last[(m, n)]).mean())
            last[(m, n)] = a
    if i % 2 == 0:
        dv = depth.copy()
        dv[dv <= 0] = np.nan
        dcol = cv2.applyColorMap(cv2.normalize(np.nan_to_num(1 / dv), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)
        row = np.hstack([label(color, "camera"), label(dcol, "depth (Kinect)"),
                         label((np.clip(drawing, 0, 1) * 255).astype(np.uint8), "outline"),
                         label(dots(show_smooth), f"grid {SHOW_N}x{SHOW_N}: outline, smoothed"),
                         label(dots(raw_grid(gray, SHOW_N)), f"grid {SHOW_N}x{SHOW_N}: raw image")])
        frames.append(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
        if i == 120:
            cv2.imwrite(f"results/{prefix}_still.png", row)
    prev, prev_depth = gray, depth

imageio.mimsave(f"results/{prefix}_demo.gif",
                [cv2.resize(f, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA) for f in frames], duration=0.08, loop=0)
summary = {"frames": N - 1, "false_streak_electrodes_24": {k: round(float(np.mean(v)), 3) for k, v in streak_cells.items()},
           "per_grid": {m: {n: {"active_fraction": round(float(np.mean(active[m][n])), 3),
                                "flicker": round(float(np.mean(flicker[m][n])), 3)} for n in SIZES} for m in methods}}
json.dump(summary, open(f"results/{prefix}_metrics.json", "w"), indent=2)
print(json.dumps(summary, indent=1))
