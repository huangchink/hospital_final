"""VA gratings, VF point geometry, the adaptive staircase, and VA scoring.

Stateless numeric logic extracted verbatim from VA_center_opt.py (no behaviour change).
"""
import math

import cv2
import numpy as np


# ---------- VA scoring ----------
def get_va_result_cpd(cpd_value: float, *_ignored):
    """
    只依據 cpd 判定 VA 分數。你可依需求微調 thr_cpd。
    """
    thr_cpd = [18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0]
    scores  = [0.9,  0.8,  0.7,  0.6,  0.5,  0.4,  0.3, 0.2, 0.1]
    for t, s in zip(thr_cpd, scores):
        if cpd_value >= t:
            return s
    return "Unknown"


DO_BLUR    = True
BLUR_KSIZE = (5, 5)
BLUR_SIGMA = 1.0


# ---------- VF Test Helpers ----------
VF_ANGULAR_DIAMETERS = {
    "Goldmann II":   0.43,
    "Goldmann III":  0.64,
    "Goldmann IV":   0.86,
    "Goldmann V":    1.72
}
VF_PASS_DWELL_SEC = 2.0
VF_TIMEOUT_SEC    = 5.0


def vf_angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg / 2))
    return int(size_cm * px_per_cm)


def vf_get_quadrant(x, y, cx, cy):
    if x > cx and y < cy: return 1
    if x < cx and y < cy: return 2
    if x < cx and y > cy: return 3
    if x > cx and y > cy: return 4
    return None


def vf_generate_points(n, max_deg_horizon, max_deg_vertical):
    if n == 5:
        return [(0, 0), (max_deg_horizon, max_deg_vertical), (-max_deg_horizon, max_deg_vertical),
                (max_deg_horizon, -max_deg_vertical), (-max_deg_horizon, -max_deg_vertical)]
    if n == 9:
        return [(0, max_deg_vertical), (max_deg_horizon, max_deg_vertical), (-max_deg_horizon, max_deg_vertical),
                (max_deg_horizon, -max_deg_vertical), (-max_deg_horizon, -max_deg_vertical), (max_deg_horizon, 0),
                (-max_deg_horizon, 0), (0, -max_deg_vertical), (0, 0)]
    if n == 13:
        return [
            (0, 0), (max_deg_horizon, 0), (-max_deg_horizon, 0), (0, max_deg_vertical), (0, -max_deg_vertical),
            (max_deg_horizon, max_deg_vertical), (-max_deg_horizon, max_deg_vertical),
            (max_deg_horizon, -max_deg_vertical), (-max_deg_horizon, -max_deg_vertical),
            (max_deg_horizon, max_deg_vertical / 2), (-max_deg_horizon, max_deg_vertical / 2),
            (max_deg_horizon, -max_deg_vertical / 2), (-max_deg_horizon, -max_deg_vertical / 2)
        ]
    raise ValueError("VF stim_points must be 5, 9, or 13")


def vf_convert_positions_to_pixels(deg_pts, w, h, px_per_cm, dist_cm, diameter_px):
    d2p = lambda d: int(px_per_cm * math.tan(math.radians(d)) * dist_cm)
    raw = [(w // 2 + d2p(x), h // 2 - d2p(y)) for x, y in deg_pts]
    margin = diameter_px // 2 + 10
    right = w - margin
    return [(max(margin, min(x, right)), max(margin, min(y, h - margin))) for x, y in raw]


# ---------- VA Grating Helpers ----------
def prepare_patch_grid(rad):
    diam = 2 * rad
    yy, xx = np.mgrid[0:diam, 0:diam]
    xx = xx - rad
    yy = yy - rad
    circle_mask = (xx * xx + yy * yy) <= (rad * rad)
    return xx.astype(np.float32), yy.astype(np.float32), circle_mask


def generate_grating_oriented_patch(freq_cycles_per_screen, xx, yy, angle_deg, w_total_px,
                                    color_dark=(0,0,0), color_light=(255,255,255), do_blur=True):
    theta = np.deg2rad(angle_deg)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * freq_cycles_per_screen * u / float(w_total_px))
    gray = (g * 255).astype(np.uint8)
    if do_blur:
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, sigmaX=BLUR_SIGMA)
    a = gray.astype(np.float32) / 255.0
    light = np.array(color_light, dtype=np.float32)
    dark  = np.array(color_dark,  dtype=np.float32)
    out = (a[..., None] * light + (1 - a[..., None]) * dark).astype(np.uint8)
    return out


# ---------- Staircase (cpd) ----------
class Staircase:
    def __init__(self, start, step, minv, maxv):
        self.freq = float(start)
        self.step = float(step)
        self.minv, self.maxv = float(minv), float(maxv)
        self.reversals = []
        self.last_correct = None
        self.correct_streak = 0
        self.max_correct_streak = 0
        self.incorrect_streak = 0

    def update(self, correct):
        if self.last_correct is not None and correct != self.last_correct:
            self.reversals.append(self.freq)
        self.last_correct = correct

        if correct:
            self.correct_streak += 1
            self.max_correct_streak = max(self.max_correct_streak, self.correct_streak)
            self.incorrect_streak = 0
        else:
            self.correct_streak = 0
            self.incorrect_streak += 1

        delta = self.step if correct else -self.step
        self.freq = min(self.maxv, max(self.minv, self.freq + delta))

    def done(self):
        return (len(self.reversals) >= 4) or \
               (self.freq >= self.maxv and self.max_correct_streak >= 3) or \
               (self.incorrect_streak >= 4)
