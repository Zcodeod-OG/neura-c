"""Synthetic RGB-D street scene: textured wall, a static box, a walking person and a fast ball.

Stands in for camera + lidar frames until real recordings exist. Returns ground-truth
instance ids so we can score what the grid keeps.
"""
import numpy as np
import cv2

H = W = 256
BG, FLOOR, BOX, PERSON, BALL = 0, 1, 2, 3, 4


def _texture(rng, h, w, scale):
    t = rng.normal(0, 1, (h // scale + 2, w // scale + 2)).astype(np.float32)
    return cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC)


class Scene:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.wall_tex = _texture(rng, H, W, 6) * 0.5 + _texture(rng, H, W, 24)
        self.box_tex = _texture(rng, H, W, 4)
        self.person_tex = _texture(rng, H, W, 3)
        self.noise_rng = rng
        # posters on the wall: pure texture, no depth edge
        self.posters = [(30, 40, 70, 90), (170, 30, 230, 80), (110, 60, 150, 100)]

    def person_mask(self, t):
        m = np.zeros((H, W), np.uint8)
        x = int(20 + 2.2 * t)
        y_hip = 175
        swing = 14 * np.sin(t * 0.35)
        cv2.circle(m, (x, y_hip - 70), 11, 1, -1)                     # head
        cv2.rectangle(m, (x - 12, y_hip - 58), (x + 12, y_hip), 1, -1)  # torso
        for s in (1, -1):
            cv2.line(m, (x, y_hip), (int(x + s * swing), y_hip + 45), 1, 8)          # legs
            cv2.line(m, (x + s * 10, y_hip - 52), (int(x - s * swing * 0.8), y_hip - 15), 1, 6)  # arms
        return m.astype(bool)

    def ball_mask(self, t):
        m = np.zeros((H, W), np.uint8)
        x = int(250 - 5.5 * t) % (W + 40) - 20
        y = int(150 - 50 * abs(np.sin(t * 0.18)))
        cv2.circle(m, (x, y), 13, 1, -1)
        return m.astype(bool)

    def frame(self, t):
        yy = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
        depth = np.full((H, W), 6.0, np.float32)
        inst = np.full((H, W), BG, np.int32)
        floor = yy > 140
        depth[floor] = (6.0 * 116 / (yy[floor] - 140 + 116 / 6.0 * 1)).clip(1.0, 6.0)
        inst[floor] = FLOOR
        gray = 0.55 + 0.08 * self.wall_tex
        for (x0, y0, x1, y1) in self.posters:
            gray[y0:y1, x0:x1] = 0.25 + 0.2 * (self.wall_tex[y0:y1, x0:x1] > 0)
        gray[floor] = 0.35 + 0.05 * self.wall_tex[floor]

        def paint(mask, d, tex, base, iid):
            closer = mask & (d < depth)
            depth[closer] = d
            gray[closer] = base + 0.12 * tex[closer]
            inst[closer] = iid

        box = np.zeros((H, W), bool)
        box[120:200, 175:235] = True
        paint(box, 3.0, self.box_tex, 0.7, BOX)
        paint(self.person_mask(t), 2.5, self.person_tex, 0.3, PERSON)
        paint(self.ball_mask(t), 1.8, self.person_tex, 0.85, BALL)

        gray = np.clip(gray + self.noise_rng.normal(0, 0.02, gray.shape), 0, 1)
        # lidar noise, about 1% of range
        depth_noisy = depth * (1 + self.noise_rng.normal(0, 0.01, depth.shape)).astype(np.float32)
        return gray.astype(np.float32), depth_noisy, inst
