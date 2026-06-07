"""Webcam face-quality overlay (shared by the settings preview and the tester dashboard).

Green/red face & eye boxes + a head-shaped centering guide, eye open/closed (EAR) and
optional pixel-based occlusion detection. Extracted verbatim from VA_center_opt.py
(no behaviour change). OpenCV uses BGR colors.
"""
import cv2
import numpy as np


_Q_GREEN = (0, 200, 0)
_Q_RED = (0, 0, 255)
_Q_GRAY = (170, 170, 170)
_Q_YELLOW = (0, 210, 230)
_Q_WHITE = (255, 255, 255)


def _put_text(img, text, org, color, scale=0.6, thickness=1, center=False):
    """Draw text with a dark outline for readability. If center=True, org[0] is treated as the
    horizontal center and the text is centered on it."""
    x, y = int(org[0]), int(org[1])
    if center:
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thickness))
        x -= tw // 2
    org = (x, y)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _rect_int(rect):
    x, y, w, h = rect
    return int(x), int(y), int(w), int(h)


# Head-guide geometry, defined by the oval's BOTTOM point + height so the bottom stays put when the
# height changes. Fractions of the (640x480) webcam frame; overridable from the Webcam tab.
_GUIDE_OVAL_SIZE_FRAC = 0.30       # overall guide SIZE = oval height as a fraction of frame height
_GUIDE_ASPECT = 0.75               # FIXED oval width/height ratio (face-like); width derives from size
_GUIDE_OVAL_BOTTOM_X_FRAC = 0.50   # horizontal position of the oval bottom (0=left, 1=right)
_GUIDE_OVAL_BOTTOM_Y_FRAC = 0.84   # vertical position of the oval bottom (0=top, 1=bottom)
# Face-fit feedback, judged against the CURRENT guide oval (recomputed each frame from the live
# Size/position), so matching your face to the oval reads OK and live changes apply immediately.
_FACE_FIT_RATIO_MIN = 0.65   # face_width / guide_width below this => TOO FAR
_FACE_FIT_RATIO_MAX = 1.20   # above this => TOO CLOSE
_FACE_CENTER_MARGIN = 1.30   # face center allowed within this multiple of the guide radii
# Eye-occlusion detection. MediaPipe keeps reporting eye landmarks even when an eye is covered, so
# we inspect the eye-box pixels relative to a skin reference (cheeks/nose), making it robust to
# lighting and skin tone. A visible eye - including one behind glasses - shows either a DARK region
# (pupil / lashes / frame) OR a BRIGHT region (sclera / lens glare); a covering hand shows neither.
_EYE_OCCL_DARK_MARGIN = 40.0        # eye pixels this much darker than skin look like pupil/lashes/frame
_EYE_OCCL_BRIGHT_MARGIN = 45.0      # eye pixels this much brighter than skin look like sclera / lens glare
_EYE_OCCL_FEATURE_FRAC_MIN = 0.001  # need this fraction of dark-OR-bright "eye-like" pixels, else occluded
# Eye open/closed via Eye Aspect Ratio (EAR; Soukupova & Cech 2016) on the MediaPipe eye landmarks.
# EAR is the standard, person/pose-insensitive measure and is robust WITH glasses and across
# lighting. It reports open/closed reliably from landmarks; it does NOT by itself detect a hand
# over an open-positioned eye (for that, enable the pixel test below).
_EAR_CLOSED_THRESH = 0.18          # average eye EAR below this => eye closed (typical open ~0.25-0.35)
# 6-point EAR landmark indices on the 478-mesh, ordered P1(outer), P2, P3, P4(inner), P5, P6.
_EAR_IDX_LEFT_RECT = [33, 160, 158, 133, 153, 144]    # eye in face_info.left_rect  (image-left)
_EAR_IDX_RIGHT_RECT = [362, 385, 387, 263, 373, 380]  # eye in face_info.right_rect (image-right)
# Optional pixel-based hand/object occlusion test. Fragile with glasses (glare/tint), so OFF by
# default; set True to also flag a hand covering an open eye (accepting occasional misses).
_EYE_USE_PIXEL_OCCLUSION = False


def _draw_face_guide(img, cx, cy, rx, ry, color, thickness=3):
    """Draw a head-shaped centering guide (輪廓): an oval head outline with two ears."""
    cx, cy, rx, ry = int(cx), int(cy), int(rx), int(ry)
    th = max(1, int(thickness))
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, th, cv2.LINE_AA)
    ear_rx = max(3, int(rx * 0.22)); ear_ry = max(5, int(ry * 0.16))
    cv2.ellipse(img, (cx - rx, cy), (ear_rx, ear_ry), 0, 0, 360, color, th, cv2.LINE_AA)
    cv2.ellipse(img, (cx + rx, cy), (ear_rx, ear_ry), 0, 0, 360, color, th, cv2.LINE_AA)


def _skin_reference(frame_bgr, face_rect):
    """Median brightness of a mid-face (cheeks/nose) patch, used as an adaptive skin reference for
    eye-occlusion detection (robust to lighting and skin tone). Returns a float or None."""
    try:
        fx, fy, fw, fh = _rect_int(face_rect)
        if fw <= 0 or fh <= 0:
            return None
        H, W = frame_bgr.shape[:2]
        x0 = max(0, fx + int(fw * 0.30)); x1 = min(W, fx + int(fw * 0.70))
        y0 = max(0, fy + int(fh * 0.55)); y1 = min(H, fy + int(fh * 0.80))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        patch = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return float(np.median(patch))
    except Exception:
        return None


def eye_region_occluded(frame_bgr, eye_rect, skin_ref=None):
    """Detect whether an eye box is occluded by a hand/object. MediaPipe infers eye landmarks from
    the face outline, so they stay 'open' under a hand; we instead inspect the PIXELS. An open eye
    shows either a dark region (pupil/lashes/frame) or a bright region (sclera/lens glare) relative
    to facial skin; a covering hand shows neither. Returns True (occluded), False (looks like an
    eye), or None (cannot tell / box too small)."""
    try:
        x, y, w, h = _rect_int(eye_rect)
        if w < 10 or h < 6:
            return None
        H, W = frame_bgr.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, W - x); h = min(h, H - y)
        if w < 10 or h < 6:
            return None
        gray = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 40), interpolation=cv2.INTER_AREA)
        if skin_ref and skin_ref > 0:
            dark_thr = skin_ref - _EYE_OCCL_DARK_MARGIN
            bright_thr = skin_ref + _EYE_OCCL_BRIGHT_MARGIN
        else:
            dark_thr, bright_thr = 70.0, 190.0
        dark_frac = float(np.mean(gray < dark_thr))      # pupil / lashes / glasses frame
        bright_frac = float(np.mean(gray > bright_thr))   # sclera / lens glare
        # A visible eye shows a dark OR a bright region; a covering hand shows neither.
        looks_like_eye = (dark_frac >= _EYE_OCCL_FEATURE_FRAC_MIN) or (bright_frac >= _EYE_OCCL_FEATURE_FRAC_MIN)
        return not looks_like_eye
    except Exception:
        return None


def _ear_from_landmarks(landmarks, idx):
    """Eye Aspect Ratio from 6 ordered eye landmarks: (|P2-P6| + |P3-P5|) / (2*|P1-P4|).
    Scale-invariant; ~0.25-0.35 for an open eye, near 0 when closed."""
    try:
        p = [np.asarray(landmarks[i][:2], dtype=np.float64) for i in idx]
        horiz = np.linalg.norm(p[0] - p[3])
        if horiz <= 1e-3:
            return None
        vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        return float(vert / (2.0 * horiz))
    except Exception:
        return None


def eye_aspect_ratios(face_info):
    """Return (ear_for_left_rect, ear_for_right_rect) from the MediaPipe 478-landmark mesh, or
    (None, None) if landmarks are unavailable. Low EAR => closed eye. Robust with glasses."""
    lms = getattr(face_info, 'face_landmarks', None)
    if lms is None or len(lms) < 468:
        return None, None
    return (_ear_from_landmarks(lms, _EAR_IDX_LEFT_RECT),
            _ear_from_landmarks(lms, _EAR_IDX_RIGHT_RECT))


def evaluate_face_centering(face_info, oval_cx, oval_cy, oval_rx, oval_ry):
    """Decide whether the detected face matches the guide oval. Judged against the CURRENT oval
    geometry (passed in fresh each frame), so a face that fills the oval reads OK and live Size /
    position changes apply immediately. Returns (fits, reason)."""
    if not face_info or not getattr(face_info, 'status', False):
        return False, "NO FACE"
    try:
        fx, fy, fw, fh = _rect_int(face_info.face_rect)
    except Exception:
        return False, "NO FACE"
    if fw <= 0 or fh <= 0:
        # MediaPipe only fills the rects on a fully-usable detection; a zero box => partial face.
        return False, "ADJUST FACE"
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    nx = (cx - oval_cx) / max(1.0, _FACE_CENTER_MARGIN * float(oval_rx))
    ny = (cy - oval_cy) / max(1.0, _FACE_CENTER_MARGIN * float(oval_ry))
    if nx * nx + ny * ny > 1.0:
        return False, "OFF-CENTER"
    ratio = fw / max(1.0, 2.0 * float(oval_rx))   # face width vs guide width; ~1.0 when it fills the oval
    if ratio < _FACE_FIT_RATIO_MIN:
        return False, "TOO FAR"
    if ratio > _FACE_FIT_RATIO_MAX:
        return False, "TOO CLOSE"
    return True, "OK"


def draw_face_quality_overlay(frame_bgr, face_info, draw_oval=True,
                              oval_size_frac=_GUIDE_OVAL_SIZE_FRAC,
                              oval_bottom_x_frac=_GUIDE_OVAL_BOTTOM_X_FRAC,
                              oval_bottom_y_frac=_GUIDE_OVAL_BOTTOM_Y_FRAC):
    """Draw green/red face & eye boxes plus a head-shaped centering guide onto frame_bgr (modified
    in place). The guide keeps a FIXED width:height ratio (oval_size_frac scales the whole oval) and
    is anchored by its BOTTOM point, so changing the size keeps the bottom fixed.
    Returns (fits, reason, left_ok, right_ok) where left_ok/right_ok are the SUBJECT's left/right
    eye visibility (MediaPipe's left_rect is the image-left = subject's right eye, mapped here)."""
    h, w = frame_bgr.shape[:2]
    oval_ry = max(1, int(h * oval_size_frac / 2.0))
    oval_rx = max(1, int(_GUIDE_ASPECT * oval_ry))     # width derived from height (fixed face ratio)
    oval_cx = int(w * oval_bottom_x_frac)
    oval_cy = int(h * oval_bottom_y_frac - oval_ry)    # center sits half-height above the bottom

    fits, reason = evaluate_face_centering(face_info, oval_cx, oval_cy, oval_rx, oval_ry)
    guide_color = _Q_GREEN if fits else _Q_RED

    if draw_oval:
        _draw_face_guide(frame_bgr, oval_cx, oval_cy, oval_rx, oval_ry, guide_color)

    left_ok = right_ok = False
    if face_info and getattr(face_info, 'status', False):
        can_gaze = bool(getattr(face_info, 'can_gaze_estimation', False))
        try:
            fx, fy, fw, fh = _rect_int(face_info.face_rect)
            if fw > 0 and fh > 0:
                cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), guide_color, 2)
        except Exception:
            pass
        bad_reason = "EYES BLOCKED"
        left_ear, right_ear = eye_aspect_ratios(face_info)
        skin_ref = _skin_reference(frame_bgr, face_info.face_rect) if _EYE_USE_PIXEL_OCCLUSION else None
        ok_flags = {'L': False, 'R': False}
        for key, rect, ear in (('L', getattr(face_info, 'left_rect', None), left_ear),
                               ('R', getattr(face_info, 'right_rect', None), right_ear)):
            ok = False
            if rect is not None:
                try:
                    ex, ey, ew, eh = _rect_int(rect)
                    if ew > 0 and eh > 0:
                        if can_gaze:
                            closed = (ear is not None and ear < _EAR_CLOSED_THRESH)  # EAR: eye open/closed
                            occ = eye_region_occluded(frame_bgr, rect, skin_ref) if _EYE_USE_PIXEL_OCCLUSION else None
                            if closed:
                                bad_reason = "EYES CLOSED"
                            elif occ is True:
                                bad_reason = "EYES BLOCKED"
                            else:
                                ok = True
                        cv2.rectangle(frame_bgr, (ex, ey), (ex + ew, ey + eh), _Q_GREEN if ok else _Q_RED, 2)
                except Exception:
                    ok = False
            ok_flags[key] = ok
        # MediaPipe's left_rect is the IMAGE-left eye = the SUBJECT's RIGHT eye (non-mirror naming),
        # so map to subject-relative: "left eye" = subject's left = right_rect, "right" = left_rect.
        left_ok, right_ok = ok_flags['R'], ok_flags['L']
        if not (left_ok and right_ok):
            _put_text(frame_bgr, bad_reason, (oval_cx, oval_cy + oval_ry + 26), _Q_RED, scale=0.6, center=True)

    _put_text(frame_bgr, reason, (oval_cx, max(22, oval_cy - oval_ry - 12)),
              _Q_GREEN if fits else _Q_RED, scale=0.7, center=True)
    return fits, reason, left_ok, right_ok


def guide_kwargs_from_cfg(cfg):
    """Head-guide geometry overrides (height + bottom X/Y) from a cfg dict, for the tester
    dashboard. Falls back to the module defaults for any missing/invalid value."""
    cfg = cfg or {}
    def _f(key, default):
        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return default
    return dict(
        oval_size_frac=_f('webcam_oval_size', _GUIDE_OVAL_SIZE_FRAC),
        oval_bottom_x_frac=_f('webcam_oval_bottom_x', _GUIDE_OVAL_BOTTOM_X_FRAC),
        oval_bottom_y_frac=_f('webcam_oval_bottom_y', _GUIDE_OVAL_BOTTOM_Y_FRAC),
    )


def _quality_color(missing_pct):
    """Green < 20% missing, yellow < 50%, else red."""
    if missing_pct < 20:
        return _Q_GREEN
    if missing_pct < 50:
        return _Q_YELLOW
    return _Q_RED
