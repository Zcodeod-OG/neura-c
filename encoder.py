"""Scene -> sparse line drawing: depth outlines weighted by nearness, plus speed streaks."""
import numpy as np
import cv2


def depth_outlines(depth):
    inv = (1.0 / depth)
    inv8 = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv8 = cv2.GaussianBlur(inv8, (5, 5), 0)
    return cv2.Canny(inv8, 20, 50) > 0


def rgb_edges(gray):
    g8 = (gray * 255).astype(np.uint8)
    g8 = cv2.GaussianBlur(g8, (5, 5), 0)
    return cv2.Canny(g8, 20, 50) > 0


def nearness(depth):
    return np.clip(2.0 / depth, 0.35, 1.0)


def motion_streaks(prev_gray, gray, edges, gain=5.0, min_speed=1.0, step=6):
    """Draw lines trailing each moving outline point, length proportional to speed."""
    flow = cv2.calcOpticalFlowFarneback((prev_gray * 255).astype(np.uint8), (gray * 255).astype(np.uint8),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    speed = np.linalg.norm(flow, axis=2)
    moving = (speed > min_speed).astype(np.uint8)
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    streaks = np.zeros(gray.shape, np.float32)
    n, labels = cv2.connectedComponents(moving)
    for k in range(1, n):
        blob = labels == k
        if blob.sum() < 60:
            continue
        v = flow[blob].mean(0)
        ring = edges & cv2.dilate(blob.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ys, xs = np.nonzero(ring)
        if len(xs) == 0:
            continue
        # trail from the back edge of the object only
        back = (xs - xs.mean()) * v[0] + (ys - ys.mean()) * v[1] < 0
        for x, y in list(zip(xs[back], ys[back]))[::step]:
            cv2.line(streaks, (int(x), int(y)), (int(x - gain * v[0]), int(y - gain * v[1])), 0.6, 1)
    return streaks, flow


def encode(prev_gray, gray, depth):
    edges = depth_outlines(depth)
    drawing = edges * nearness(depth)
    streaks, flow = motion_streaks(prev_gray, gray, edges)
    return np.maximum(drawing, streaks).astype(np.float32), edges, streaks


def fill_depth(depth_m):
    """Fill lidar/Kinect holes (zeros) so they don't create fake outlines; returns filled depth and hole mask."""
    holes = depth_m <= 0
    d8 = cv2.normalize(np.where(holes, 0, 1.0 / np.maximum(depth_m, 1e-3)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filled = cv2.inpaint(d8, holes.astype(np.uint8), 5, cv2.INPAINT_NS).astype(np.float32)
    inv = filled / 255 * (1.0 / np.maximum(depth_m[~holes], 1e-3)).max() if (~holes).any() else filled
    return 1.0 / np.maximum(inv, 1e-3), holes


def residual_flow(flow):
    """Remove camera ego-motion: fit a homography to the flow field and keep what it can't explain."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    src = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    dst = src + flow[ys.ravel(), xs.ravel()]
    Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
    if Hm is None:
        return flow
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([gx, gy, np.ones_like(gx)], -1) @ Hm.T
    ego = pts[..., :2] / pts[..., 2:3] - np.stack([gx, gy], -1)
    return flow - ego


def residual_flow_3d(flow, prev_depth_m, K):
    """Remove camera ego-motion using depth: solve the 6-DoF camera move from static points, subtract its flow.

    A homography can't explain parallax from nearby objects, so it leaves false motion; depth fixes that.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[4:h:6, 4:w:6]
    ys, xs = ys.ravel(), xs.ravel()
    z = prev_depth_m[ys, xs]
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], 1).astype(np.float32)
    uv = (np.stack([xs, ys], 1) + flow[ys, xs]).astype(np.float32)
    okp, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, K, None, reprojectionError=1.5, iterationsCount=200)
    if not okp:
        return residual_flow(flow)
    # predicted flow for every pixel with depth; pixels without depth fall back to zero residual
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    zz = prev_depth_m
    P = np.stack([(gx - cx) * zz / fx, (gy - cy) * zz / fy, zz], -1).reshape(-1, 3)
    proj, _ = cv2.projectPoints(P, rvec, tvec, K, None)
    ego = proj.reshape(h, w, 2) - np.stack([gx, gy], -1)
    res = flow - ego
    res[zz <= 0] = 0
    return res


def encode_real(prev_gray, gray, depth_m, compensate=True, prev_depth_m=None, K=None):
    depth, holes = fill_depth(depth_m)
    edges = depth_outlines(depth)
    edges &= ~cv2.dilate(holes.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    drawing = edges * nearness(depth)
    flow = cv2.calcOpticalFlowFarneback((prev_gray * 255).astype(np.uint8), (gray * 255).astype(np.uint8),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if compensate and prev_depth_m is not None and K is not None:
        flow = residual_flow_3d(flow, prev_depth_m, K)
    elif compensate:
        flow = residual_flow(flow)
    streaks = streaks_from_flow(flow, edges)
    return np.maximum(drawing, streaks).astype(np.float32), edges, streaks


def streaks_from_flow(flow, edges, gain=5.0, min_speed=1.5, step=6):
    speed = np.linalg.norm(flow, axis=2)
    moving = cv2.morphologyEx((speed > min_speed).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    streaks = np.zeros(edges.shape, np.float32)
    n, labels = cv2.connectedComponents(moving)
    for k in range(1, n):
        blob = labels == k
        if blob.sum() < 60:
            continue
        v = flow[blob].mean(0)
        ring = edges & cv2.dilate(blob.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ys, xs = np.nonzero(ring)
        if len(xs) == 0:
            continue
        back = (xs - xs.mean()) * v[0] + (ys - ys.mean()) * v[1] < 0
        for x, y in list(zip(xs[back], ys[back]))[::step]:
            cv2.line(streaks, (int(x), int(y)), (int(x - gain * v[0]), int(y - gain * v[1])), 0.6, 1)
    return streaks
