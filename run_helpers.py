"""Rendering helpers shared by run.py and run_real.py."""
import numpy as np
import cv2
from scene import H


def dots(g, size=H):
    img = np.zeros((size, size), np.uint8)
    n = g.shape[0]
    c = size / n
    for i in range(n):
        for j in range(n):
            ctr = (int((j + .5) * c), int((i + .5) * c))
            cv2.circle(img, ctr, max(1, int(c * .12)), 40, -1)  # idle electrode
            if g[i, j] > 0:
                cv2.circle(img, ctr, int(c * .45), int(80 + 175 * g[i, j]), -1)
    return img


def label(img, text):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out
