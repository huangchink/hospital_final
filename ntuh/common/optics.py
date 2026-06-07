"""Pure unit / optics conversions shared by the VA, VF and calibration tools.

Extracted verbatim from VA_center_opt.py (no behaviour change).
"""
import math

import numpy as np


def to_rgb_tuple(rgb_like):
    return tuple(int(v) for v in rgb_like)


def screen_width_deg_from_cm(width_cm: float, dist_cm: float) -> float:
    if dist_cm <= 0 or width_cm <= 0:
        return 0.0
    return 2.0 * math.degrees(math.atan((width_cm / 2.0) / dist_cm))


def px_to_cm(px: float, scr_width_cm: float, scr_width_px: float) -> float:
    """Convert a pixel length to centimeters using the screen's physical width and resolution."""
    try:
        if scr_width_cm <= 0 or scr_width_px <= 0:
            return 0.0
        return float(px) * float(scr_width_cm) / float(scr_width_px)
    except (TypeError, ValueError):
        return 0.0


def cm_to_px(cm: float, scr_width_cm: float, scr_width_px: float) -> int:
    """Convert a centimeter length to pixels using the screen's physical width and resolution."""
    try:
        if scr_width_cm <= 0 or scr_width_px <= 0:
            return 0
        return int(round(float(cm) * float(scr_width_px) / float(scr_width_cm)))
    except (TypeError, ValueError):
        return 0


def mean_color_rgb(light, dark):
    l = np.array(light, dtype=np.uint16)
    d = np.array(dark,  dtype=np.uint16)
    return tuple(((l + d) // 2).astype(np.uint8))
