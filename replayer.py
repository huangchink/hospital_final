import sys
import os
import time
import json
from datetime import datetime
import cv2
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QStatusBar, QMenuBar, QGroupBox, QFrame,
    QLineEdit, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import (
    Qt, QTimer, QMimeData, pyqtSignal, QPoint,
)
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QFont, QDrag, QAction,
    QShortcut, QKeySequence,
)


# ---------------------------------------------------------------------------
# VideoController — kept from original (timestamp-based frame seeking)
# ---------------------------------------------------------------------------
class VideoController:
    def __init__(self, name, video_path, timestamp_path):
        self.name = name
        self.cap = cv2.VideoCapture(video_path)
        self.timestamps = np.array([])
        self.valid = False
        self.current_frame_idx = -1
        self.last_frame_img = None
        self.width = 0
        self.height = 0

        if self.cap.isOpened():
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if os.path.exists(timestamp_path):
                try:
                    df = pd.read_csv(timestamp_path)
                    self.timestamps = df['timestamp'].values
                    self.valid = True
                except Exception as e:
                    print(f"[{name}] Error loading timestamps: {e}")
            else:
                print(f"[{name}] No timestamp file found at {timestamp_path}")

    def normalize_timestamps(self, start_time):
        if self.valid:
            self.timestamps = self.timestamps - start_time

    def get_frame_at_time(self, t):
        if not self.valid or not self.timestamps.size:
            return None

        idx = np.searchsorted(self.timestamps, t, side='right') - 1
        idx = max(0, min(idx, len(self.timestamps) - 1))

        if idx != self.current_frame_idx:
            if abs(idx - self.current_frame_idx) > 5:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame_img = frame
                    self.current_frame_idx = idx
            elif idx > self.current_frame_idx:
                while self.current_frame_idx < idx:
                    ret = self.cap.grab()
                    if not ret:
                        break
                    self.current_frame_idx += 1
                ret, frame = self.cap.retrieve()
                if ret:
                    self.last_frame_img = frame
            elif idx < self.current_frame_idx:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame_img = frame
                    self.current_frame_idx = idx

        return self.last_frame_img

    def release(self):
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# PlaybackEngine — manages playback state, session loading, gaze data
# ---------------------------------------------------------------------------
class PlaybackEngine(QWidget):
    time_changed = pyqtSignal(float)
    session_loaded = pyqtSignal()
    playback_toggled = pyqtSignal(bool)   # True = playing
    speed_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_dir = None
        self.controllers: dict[str, VideoController] = {}
        self.webcam_gaze_df = None
        self.sol_gaze_df = None
        self.trial_events_df = None
        self.webcam_quality_df = None
        self.review = None  # human-in-the-loop review labels (see ReviewStore methods)
        # Gaze coordinate space (original screen resolution). The screen video may be
        # downscaled, so gaze overlays must be scaled from this to the video resolution.
        self.screen_width = None
        self.screen_height = None
        self.start_time = 0.0
        self.duration = 0.0
        self.master_clock = 0.0
        self.is_playing = False
        self.playback_speed = 1.0

        # Sol low-FPS cache
        self.last_valid_sol_gaze = None
        self.last_sol_gaze_time = None
        self.sol_gaze_timeout = 0.5

        self._last_real_time = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)

    # -- Session loading (ported from Replayer.load_session) ----------------

    def load_session(self, directory: str) -> bool:
        # Release previous
        for c in self.controllers.values():
            c.release()
        self.controllers.clear()
        self.webcam_gaze_df = None
        self.sol_gaze_df = None
        self.trial_events_df = None
        self.webcam_quality_df = None
        self.review = None
        self.screen_width = None
        self.screen_height = None
        self.master_clock = 0.0
        self.last_valid_sol_gaze = None
        self.last_sol_gaze_time = None

        self.session_dir = directory
        print(f"Loading session from: {directory}")

        # Video paths
        sc_vid = os.path.join(directory, "screen_record.mp4")
        sc_ts  = os.path.join(directory, "screen_video_timestamp.csv")
        wc_vid = os.path.join(directory, "webcam_video.mp4")
        wc_ts  = os.path.join(directory, "webcam_video_timestamp.csv")
        so_vid = os.path.join(directory, "sol_video.mp4")
        so_ts  = os.path.join(directory, "sol_video_timestamp.csv")

        webcam_csv = os.path.join(directory, "webcam_gaze_data.csv")
        sol_csv    = os.path.join(directory, "sol_gaze_data.csv")
        trial_csv  = os.path.join(directory, "trial_events.csv")

        if os.path.exists(sc_vid):
            self.controllers['screen'] = VideoController("Screen", sc_vid, sc_ts)
        if os.path.exists(wc_vid):
            self.controllers['webcam'] = VideoController("Webcam", wc_vid, wc_ts)
        if os.path.exists(so_vid):
            self.controllers['sol'] = VideoController("Sol", so_vid, so_ts)

        # Global start time
        start_times = []
        for c in self.controllers.values():
            if c.valid and len(c.timestamps) > 0:
                start_times.append(c.timestamps[0])
        if not start_times:
            print("Error: No valid timestamp data found.")
            return False

        self.start_time = min(start_times)

        # Normalize
        max_dur = 0.0
        for c in self.controllers.values():
            c.normalize_timestamps(self.start_time)
            if c.valid and len(c.timestamps) > 0:
                max_dur = max(max_dur, c.timestamps[-1])
        self.duration = max_dur

        # Gaze CSVs
        if os.path.exists(webcam_csv):
            try:
                self.webcam_gaze_df = pd.read_csv(webcam_csv)
                self.webcam_gaze_df['t_norm'] = self.webcam_gaze_df['timestamp'] - self.start_time
                print(f"Loaded webcam gaze data: {len(self.webcam_gaze_df)} samples")
            except Exception as e:
                print(f"Error loading webcam_gaze_data.csv: {e}")

        if os.path.exists(sol_csv):
            try:
                self.sol_gaze_df = pd.read_csv(sol_csv)
                self.sol_gaze_df['timestamp'] = self.sol_gaze_df['pc_timestamp_ms'] / 1000.0
                self.sol_gaze_df['t_norm'] = self.sol_gaze_df['timestamp'] - self.start_time
                print(f"Loaded Sol gaze data: {len(self.sol_gaze_df)} samples")
            except Exception as e:
                print(f"Error loading sol_gaze_data.csv: {e}")

        # Trial events
        if os.path.exists(trial_csv):
            try:
                self.trial_events_df = pd.read_csv(trial_csv)
                self.trial_events_df['start_norm'] = self.trial_events_df['start_timestamp'] - self.start_time
                self.trial_events_df['end_norm'] = self.trial_events_df['end_timestamp'] - self.start_time
                print(f"Loaded trial events: {len(self.trial_events_df)} trials")
            except Exception as e:
                print(f"Error loading trial_events.csv: {e}")

        # Webcam per-frame quality (face/eye validity) for the timeline + numbers
        wq_csv = os.path.join(directory, "webcam_quality.csv")
        if os.path.exists(wq_csv):
            try:
                self.webcam_quality_df = pd.read_csv(wq_csv)
                self.webcam_quality_df['t_norm'] = (
                    self.webcam_quality_df['pc_timestamp_ms'] / 1000.0 - self.start_time
                )
                print(f"Loaded webcam quality: {len(self.webcam_quality_df)} rows")
            except Exception as e:
                print(f"Error loading webcam_quality.csv: {e}")

        # Screen resolution metadata (gaze coordinate space). The screen video may be
        # downscaled; gaze overlays are scaled from screen_width/height to the video size.
        meta_path = os.path.join(directory, "screen_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.screen_width = int(meta.get("screen_width")) if meta.get("screen_width") else None
                self.screen_height = int(meta.get("screen_height")) if meta.get("screen_height") else None
                print(f"Screen meta: gaze coord space {self.screen_width}x{self.screen_height}, "
                      f"video {meta.get('screen_video_width')}x{meta.get('screen_video_height')}")
            except Exception as e:
                print(f"Error loading screen_meta.json: {e}")
        else:
            print("No screen_meta.json (older session) — gaze overlay assumes screen video is full resolution")

        # Human-in-the-loop review labels (pass/fail/discard per trial + keep/discard record)
        self._load_or_init_review(directory)

        print(f"Session loaded. Duration: {self.duration:.2f}s")
        self.session_loaded.emit()
        return True

    # -- Review labels (human-in-the-loop) ---------------------------------

    REVIEW_FILE = "review_labels.json"

    def _review_path(self):
        return os.path.join(self.session_dir, self.REVIEW_FILE) if self.session_dir else None

    def _load_or_init_review(self, directory):
        """Load review_labels.json if present, else initialise from the test's auto results.
        Per-trial labels pre-fill from trial_events 'result' (PASS->pass, FAIL->fail); the
        reviewer confirms or overrides. 'reviewed' flips True only on an explicit action."""
        path = os.path.join(directory, self.REVIEW_FILE)
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception as e:
                print(f"Error loading {self.REVIEW_FILE}: {e}")
                existing = {}

        ex_trials = existing.get("trials", {}) if isinstance(existing, dict) else {}
        ex_record = existing.get("record", {}) if isinstance(existing, dict) else {}

        trials = {}
        if self.trial_events_df is not None:
            for _, row in self.trial_events_df.iterrows():
                tnum = str(row.get("trial_number", "?"))
                auto = str(row.get("result", "")).upper()
                auto_label = "pass" if auto == "PASS" else ("fail" if auto == "FAIL" else "discard")
                prev = ex_trials.get(tnum, {})
                trials[tnum] = {
                    "label": prev.get("label", auto_label),
                    "auto_result": auto,
                    "cpd": row.get("cpd", None),
                    "side": row.get("side", None),
                    "note": prev.get("note", ""),
                    "reviewed": bool(prev.get("reviewed", False)),
                }

        self.review = {
            "schema_version": 1,
            "session": os.path.basename(directory.rstrip("/\\")),
            "reviewer": existing.get("reviewer", "") if isinstance(existing, dict) else "",
            "reviewed_at": existing.get("reviewed_at", "") if isinstance(existing, dict) else "",
            "record": {
                "label": ex_record.get("label", "keep"),
                "note": ex_record.get("note", ""),
                "reviewed": bool(ex_record.get("reviewed", False)),
            },
            "trials": trials,
        }
        n_done = sum(1 for t in trials.values() if t["reviewed"])
        print(f"Review labels: {n_done}/{len(trials)} trials reviewed"
              + (" (loaded existing)" if existing else " (initialised from test results)"))

    def save_review(self):
        """Write review labels to review_labels.json. Returns the save time string (HH:MM:SS) or None."""
        if not self.review or not self.session_dir:
            return None
        try:
            now = datetime.now()
            self.review["reviewed_at"] = now.isoformat(timespec="seconds")
            with open(self._review_path(), "w", encoding="utf-8") as f:
                json.dump(self.review, f, indent=2, ensure_ascii=False)
            return now.strftime("%H:%M:%S")
        except Exception as e:
            print(f"Error saving {self.REVIEW_FILE}: {e}")
            return None

    def set_trial_label(self, tnum, label):
        if not self.review:
            return None
        t = self.review["trials"].get(str(tnum))
        if t is None:
            return None
        t["label"] = label
        t["reviewed"] = True
        return self.save_review()

    def set_trial_note(self, tnum, note):
        if not self.review:
            return None
        t = self.review["trials"].get(str(tnum))
        if t is None:
            return None
        t["note"] = note
        return self.save_review()

    def set_record_label(self, label):
        if not self.review:
            return None
        self.review["record"]["label"] = label
        self.review["record"]["reviewed"] = True
        return self.save_review()

    def set_record_note(self, note):
        if not self.review:
            return None
        self.review["record"]["note"] = note
        return self.save_review()

    def set_reviewer(self, name):
        if not self.review:
            return None
        self.review["reviewer"] = name
        return self.save_review()

    def review_progress(self):
        """(reviewed_trial_count, total_trial_count)."""
        if not self.review:
            return (0, 0)
        trials = self.review["trials"]
        return (sum(1 for t in trials.values() if t.get("reviewed")), len(trials))

    # -- Validity (data quality) -------------------------------------------

    def _validity_arrays(self):
        """Per-source validity time series: {'sol': (t_norm, valid01), 'webcam': (...)} (None if absent)."""
        out = {'sol': None, 'webcam': None}
        try:
            df = self.sol_gaze_df
            if df is not None and len(df) and 'is_valid' in df.columns:
                t = df['t_norm'].to_numpy(dtype=float)
                v = (df['is_valid'].to_numpy() == 1).astype(float)
                out['sol'] = (t, v)
        except Exception as e:
            print(f"[validity] sol series error: {e}")
        try:
            df = self.webcam_quality_df
            if df is not None and len(df) and {'face_ok', 'left_eye_ok', 'right_eye_ok'}.issubset(df.columns):
                t = df['t_norm'].to_numpy(dtype=float)
                v = ((df['face_ok'] == 1) & (df['left_eye_ok'] == 1) & (df['right_eye_ok'] == 1)).to_numpy().astype(float)
                out['webcam'] = (t, v)
        except Exception as e:
            print(f"[validity] webcam series error: {e}")
        return out

    def validity_summary(self):
        """Overall and trial-only validity % per source: {'sol': {'overall','trial','n'}, ...}."""
        arrs = self._validity_arrays()
        windows = []
        if self.trial_events_df is not None:
            try:
                for _, r in self.trial_events_df.iterrows():
                    windows.append((float(r['start_norm']), float(r['end_norm'])))
            except Exception:
                pass
        summary = {}
        for key, ar in arrs.items():
            if ar is None:
                continue
            t, v = ar
            if len(v) == 0:
                continue
            overall = float(v.mean() * 100.0)
            trial = None
            if windows:
                mask = np.zeros(len(t), dtype=bool)
                for (a, b) in windows:
                    mask |= (t >= a) & (t <= b)
                if mask.any():
                    trial = float(v[mask].mean() * 100.0)
            summary[key] = {'overall': overall, 'trial': trial, 'n': int(len(v))}
        return summary

    # -- Gaze data lookup ---------------------------------------------------

    def get_data_at_time(self, t: float) -> dict:
        res = {}

        if self.webcam_gaze_df is not None:
            idx = self.webcam_gaze_df['t_norm'].searchsorted(t, side='right') - 1
            if 0 <= idx < len(self.webcam_gaze_df):
                res['webcam'] = self.webcam_gaze_df.iloc[idx]

        if self.sol_gaze_df is not None:
            idx = self.sol_gaze_df['t_norm'].searchsorted(t, side='right') - 1
            if 0 <= idx < len(self.sol_gaze_df):
                row = self.sol_gaze_df.iloc[idx]
                data_time = row['t_norm']
                time_diff = t - data_time
                if time_diff < self.sol_gaze_timeout:
                    res['sol'] = row
                    self.last_valid_sol_gaze = row
                    self.last_sol_gaze_time = data_time
                elif self.last_valid_sol_gaze is not None and self.last_sol_gaze_time is not None:
                    if t - self.last_sol_gaze_time < self.sol_gaze_timeout:
                        res['sol'] = self.last_valid_sol_gaze

        return res

    # -- Playback controls --------------------------------------------------

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self._last_real_time = time.time()
            self._timer.start()
            self.playback_toggled.emit(True)

    def pause(self):
        if self.is_playing:
            self.is_playing = False
            self._timer.stop()
            self.playback_toggled.emit(False)

    def toggle(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, t: float):
        self.master_clock = max(0.0, min(t, self.duration))
        self._last_real_time = time.time()
        self.time_changed.emit(self.master_clock)

    def set_speed(self, s: float):
        self.playback_speed = max(0.1, min(5.0, s))
        self.speed_changed.emit(self.playback_speed)

    def restart(self):
        self.seek(0.0)

    # -- Timer tick ---------------------------------------------------------

    def _tick(self):
        now = time.time()
        dt = now - self._last_real_time
        self._last_real_time = now

        self.master_clock += dt * self.playback_speed
        if self.master_clock >= self.duration:
            self.master_clock = self.duration
            self.pause()

        self.time_changed.emit(self.master_clock)


# ---------------------------------------------------------------------------
# VideoDisplayWidget — renders one video stream, supports drag-and-drop swap
# ---------------------------------------------------------------------------
MIME_STREAM = "application/x-stream-name"


class VideoDisplayWidget(QWidget):
    stream_swapped = pyqtSignal()  # emitted after a D&D swap so app can rewire

    def __init__(self, stream_name: str, label: str, parent=None):
        super().__init__(parent)
        self.stream_name = stream_name
        self.label = label
        self._qimage = None
        self._frame_bgr = None

        # Gaze overlay data (set externally each tick)
        self._gaze_points: list[tuple[float, float, QColor, str]] = []
        # Each entry: (x, y, colour, label) in gaze-source pixel coords (see _gaze_src_*)

        self._frame_w = 0
        self._frame_h = 0

        # Resolution the gaze coords are expressed in. If None, assume the displayed frame's
        # own resolution. For the screen display this is the original screen resolution, which
        # differs from the (possibly downscaled) screen video — so gaze must be rescaled.
        self._gaze_src_w = None
        self._gaze_src_h = None

        self.setAcceptDrops(True)
        self.setMinimumSize(120, 90)

        self._drag_highlight = False

    # -- Frame update -------------------------------------------------------

    def set_frame(self, bgr_frame):
        if bgr_frame is None:
            self._qimage = None
            self._frame_bgr = None
            self.update()
            return
        self._frame_bgr = bgr_frame
        h, w, ch = bgr_frame.shape
        self._frame_w = w
        self._frame_h = h
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def set_gaze_points(self, pts: list):
        self._gaze_points = pts

    def set_gaze_source_size(self, w, h):
        """Resolution the gaze coords are in (e.g. original screen res). Overlays are scaled
        from this to the displayed frame size. Pass None/0 to assume the frame's own size."""
        self._gaze_src_w = int(w) if w else None
        self._gaze_src_h = int(h) if h else None
        self.update()

    # -- Painting -----------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Background
        p.fillRect(rect, QColor(30, 30, 30))

        if self._qimage and not self._qimage.isNull():
            # Aspect-ratio preserving scale
            iw, ih = self._qimage.width(), self._qimage.height()
            scale = min(rect.width() / iw, rect.height() / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            x_off = (rect.width() - nw) // 2
            y_off = (rect.height() - nh) // 2

            scaled = self._qimage.scaled(nw, nh, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            p.drawImage(x_off, y_off, scaled)

            # Gaze overlays. Gaze coords are in gaze-source resolution (default: frame size);
            # scale to the frame, then to the widget. This corrects overlays when the screen
            # video was downscaled but gaze stayed in original screen coordinates.
            src_w = self._gaze_src_w or iw
            src_h = self._gaze_src_h or ih
            sx = iw / src_w
            sy = ih / src_h
            for gx, gy, color, lbl in self._gaze_points:
                vx = int(gx * sx * scale) + x_off
                vy = int(gy * sy * scale) + y_off
                pen = QPen(color, 2)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPoint(vx, vy), 10, 10)
                p.setFont(QFont("Arial", 8))
                p.drawText(vx + 13, vy + 4, lbl)
        else:
            p.setPen(QColor(100, 100, 100))
            p.setFont(QFont("Arial", 14))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"No {self.label}")

        # Label badge
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        fm = p.fontMetrics()
        p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        tw = fm.horizontalAdvance(self.label) + 12
        p.drawRoundedRect(4, 4, tw, 22, 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(10, 19, self.label)

        # Drag highlight
        if self._drag_highlight:
            p.setPen(QPen(QColor(0, 170, 255), 3))
            p.setBrush(QColor(0, 170, 255, 40))
            p.drawRect(rect.adjusted(1, 1, -1, -1))

        p.end()

    # -- Drag & Drop --------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(MIME_STREAM, self.stream_name.encode())
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_STREAM):
            src_name = bytes(event.mimeData().data(MIME_STREAM)).decode()
            if src_name != self.stream_name:
                event.acceptProposedAction()
                self._drag_highlight = True
                self.update()

    def dragLeaveEvent(self, event):
        self._drag_highlight = False
        self.update()

    def dropEvent(self, event):
        self._drag_highlight = False
        src_name = bytes(event.mimeData().data(MIME_STREAM)).decode()
        # Find the source widget — the drag originator
        src_widget = event.source()
        if isinstance(src_widget, VideoDisplayWidget) and src_widget is not self:
            # Swap stream names and labels
            src_widget.stream_name, self.stream_name = self.stream_name, src_widget.stream_name
            src_widget.label, self.label = self.label, src_widget.label
            self.stream_swapped.emit()
            src_widget.stream_swapped.emit()
        event.acceptProposedAction()
        self.update()


# ---------------------------------------------------------------------------
# TimelineWidget — custom painted timeline with trial markers
# ---------------------------------------------------------------------------
class TimelineWidget(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(84)
        self.setMouseTracking(True)

        self._duration = 0.0
        self._current_time = 0.0
        self._trials = []  # list of (start_norm, end_norm, result_str, trial_num, cpd)
        self._hover_trial = None
        self._scrubbing = False
        self._strip_N = 360
        self._sol_band = None   # np int array (0=no data, 1=green, 2=amber, 3=red) or None
        self._wc_band = None

    def set_duration(self, d: float):
        self._duration = d
        self.update()

    def set_time(self, t: float):
        self._current_time = t
        self.update()

    def set_trials(self, trials: list):
        # Normalise to 6-tuples (start, end, result, tnum, cpd, review_label); pad legacy 5-tuples.
        norm = []
        for t in trials:
            t = tuple(t)
            if len(t) == 5:
                t = t + (None,)
            norm.append(t)
        self._trials = norm
        self.update()

    def set_validity(self, sol, webcam):
        """sol/webcam: (t_norm_array, valid01_array) or None. Builds per-bucket validity bands."""
        self._sol_band = self._bucketize(sol)
        self._wc_band = self._bucketize(webcam)
        self.update()

    def _bucketize(self, arr):
        """Down-sample a validity time series to self._strip_N bands (0=no data,1=green,2=amber,3=red)."""
        if arr is None or self._duration <= 0:
            return None
        t, v = arr
        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(t) == 0:
            return None
        N = self._strip_N
        idx = np.clip((t / self._duration * N).astype(int), 0, N - 1)
        counts = np.bincount(idx, minlength=N).astype(float)
        sums = np.bincount(idx, weights=v, minlength=N).astype(float)
        band = np.zeros(N, dtype=int)  # 0 = no data (gray)
        nz = counts > 0
        frac = np.zeros(N)
        frac[nz] = sums[nz] / counts[nz]
        band[nz & (frac >= 0.8)] = 1
        band[nz & (frac >= 0.5) & (frac < 0.8)] = 2
        band[nz & (frac < 0.5)] = 3
        return band

    @staticmethod
    def _band_color(b):
        return {0: QColor(55, 55, 55), 1: QColor(76, 175, 80),
                2: QColor(255, 193, 7), 3: QColor(244, 67, 54)}[int(b)]

    def _draw_strip(self, p, band, y, sh, margin, usable, label):
        p.setPen(Qt.PenStyle.NoPen)
        if band is None:
            p.setBrush(QColor(48, 48, 48))
            p.drawRect(margin, y, usable, sh)
        else:
            N = len(band)
            i = 0
            while i < N:
                b = band[i]
                j = i + 1
                while j < N and band[j] == b:
                    j += 1
                x1 = margin + int(i / N * usable)
                x2 = margin + int(j / N * usable)
                p.setBrush(self._band_color(b))
                p.drawRect(x1, y, max(1, x2 - x1), sh)
                i = j
        # Source label (left edge, over the strip)
        p.setPen(QColor(235, 235, 235))
        p.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        p.drawText(margin + 3, y + sh - 2, label)

    # -- Coordinate helpers -------------------------------------------------

    def _time_to_x(self, t: float) -> int:
        if self._duration <= 0:
            return 0
        margin = 8
        usable = self.width() - 2 * margin
        return margin + int((t / self._duration) * usable)

    def _x_to_time(self, x: int) -> float:
        margin = 8
        usable = self.width() - 2 * margin
        t = ((x - margin) / usable) * self._duration if usable > 0 else 0.0
        return max(0.0, min(t, self._duration))

    # -- Painting -----------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(40, 40, 40))

        bar_y = 16
        bar_h = 13
        margin = 8
        usable = w - 2 * margin

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(70, 70, 70))
        p.drawRoundedRect(margin, bar_y, usable, bar_h, 4, 4)

        # Trial markers (coloured by review label, falling back to the test result)
        for (ts, te, result, tnum, cpd, label) in self._trials:
            x1 = self._time_to_x(ts)
            x2 = self._time_to_x(te)
            tw = max(x2 - x1, 3)
            if label == "discard":
                color = QColor(150, 150, 150, 150)
            elif label == "fail":
                color = QColor(244, 67, 54, 160)
            elif label == "pass":
                color = QColor(76, 175, 80, 160)
            else:
                color = QColor(76, 175, 80, 140) if result == "PASS" else QColor(244, 67, 54, 140)
            p.setBrush(color)
            p.drawRoundedRect(x1, bar_y, tw, bar_h, 2, 2)

        # Validity strips (Sol + Webcam), below the trial bar
        strip_y0 = bar_y + bar_h + 3
        self._draw_strip(p, self._sol_band, strip_y0, 10, margin, usable, "SOL")
        self._draw_strip(p, self._wc_band, strip_y0 + 12, 10, margin, usable, "CAM")

        # Progress fill
        if self._duration > 0:
            px = self._time_to_x(self._current_time)
            p.setBrush(QColor(0, 188, 212, 200))
            fill_w = px - margin
            if fill_w > 0:
                p.drawRoundedRect(margin, bar_y, fill_w, bar_h, 4, 4)

            # Playhead
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPoint(px, bar_y + bar_h // 2), 6, 6)

            # Playhead line through the validity strips (alignment)
            p.setPen(QPen(QColor(255, 255, 255, 170), 1))
            p.drawLine(px, bar_y, px, bar_y + bar_h + 3 + 22)

        # Time text
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Consolas", 9))
        cur = self._format_time(self._current_time)
        dur = self._format_time(self._duration)
        p.drawText(margin, h - 4, f"{cur} / {dur}")

        # Hover tooltip
        if self._hover_trial is not None:
            ts, te, result, tnum, cpd, label = self._hover_trial
            tip = f"Trial {tnum}  CPD:{cpd}  {result}" + (f" -> {label}" if label else "")
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 200))
            tw = p.fontMetrics().horizontalAdvance(tip) + 12
            mouse_x = self.mapFromGlobal(self.cursor().pos()).x()
            tx = min(mouse_x, w - tw - 4)
            p.drawRoundedRect(tx, 0, tw, 16, 3, 3)
            p.setPen(QColor(255, 255, 255))
            p.drawText(tx + 6, 12, tip)

        p.end()

    @staticmethod
    def _format_time(s: float) -> str:
        m = int(s) // 60
        sec = s - m * 60
        return f"{m}:{sec:05.2f}"

    # -- Mouse interaction --------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._scrubbing = True
            self.seek_requested.emit(self._x_to_time(event.pos().x()))

    def mouseMoveEvent(self, event):
        if self._scrubbing:
            self.seek_requested.emit(self._x_to_time(event.pos().x()))
        # Hover trial detection
        t = self._x_to_time(event.pos().x())
        found = None
        for trial in self._trials:
            if trial[0] <= t <= trial[1]:
                found = trial
                break
        if found != self._hover_trial:
            self._hover_trial = found
            self.update()

    def mouseReleaseEvent(self, event):
        self._scrubbing = False

    def leaveEvent(self, event):
        self._hover_trial = None
        self.update()


# ---------------------------------------------------------------------------
# TransportControls — play/pause, speed buttons
# ---------------------------------------------------------------------------
class TransportControls(QWidget):
    def __init__(self, engine: PlaybackEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(80)
        self.play_btn.clicked.connect(self.engine.toggle)

        self.speed_down = QPushButton("-")
        self.speed_down.setFixedWidth(32)
        self.speed_down.clicked.connect(lambda: self.engine.set_speed(self.engine.playback_speed - 0.25))

        self.speed_label = QLabel("1.00x")
        self.speed_label.setFixedWidth(52)
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.speed_up = QPushButton("+")
        self.speed_up.setFixedWidth(32)
        self.speed_up.clicked.connect(lambda: self.engine.set_speed(self.engine.playback_speed + 0.25))

        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.play_btn)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Speed:"))
        layout.addWidget(self.speed_down)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_up)
        layout.addStretch()
        layout.addWidget(self.time_label)

        # Connect signals
        self.engine.playback_toggled.connect(self._on_playback_toggled)
        self.engine.speed_changed.connect(self._on_speed_changed)

    def _on_playback_toggled(self, playing: bool):
        self.play_btn.setText("Pause" if playing else "Play")

    def _on_speed_changed(self, s: float):
        self.speed_label.setText(f"{s:.2f}x")

    def update_time(self, t: float, duration: float):
        cur = self._fmt(t)
        dur = self._fmt(duration)
        self.time_label.setText(f"{cur} / {dur}")

    @staticmethod
    def _fmt(s: float) -> str:
        m = int(s) // 60
        sec = s - m * 60
        return f"{m}:{sec:05.2f}"


# ---------------------------------------------------------------------------
# ConfigPanel — session loader, overlay toggles, trial list
# ---------------------------------------------------------------------------
class ConfigPanel(QWidget):
    open_folder_requested = pyqtSignal()
    overlay_changed = pyqtSignal()
    trial_selected = pyqtSignal(float)  # seek time
    reviewer_changed = pyqtSignal(str)
    record_label_changed = pyqtSignal(str)   # 'keep' | 'discard'
    record_note_changed = pyqtSignal(str)
    trial_label_set = pyqtSignal(str)        # 'pass' | 'fail' | 'discard' (applies to current trial)

    LABEL_COLORS = {"pass": "#4CAF50", "fail": "#F44336", "discard": "#9E9E9E"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self._trial_meta = {}  # str(tnum) -> (cpd, side), for refreshing rows

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Session loader
        grp_session = QGroupBox("Session")
        gl = QVBoxLayout(grp_session)
        self.open_btn = QPushButton("Open Folder...")
        self.open_btn.clicked.connect(self.open_folder_requested.emit)
        self.path_label = QLabel("No session loaded")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #aaa; font-size: 11px;")
        gl.addWidget(self.open_btn)
        gl.addWidget(self.path_label)
        layout.addWidget(grp_session)

        # Overlay toggles
        grp_overlay = QGroupBox("Gaze Overlays")
        ol = QVBoxLayout(grp_overlay)
        self.chk_webcam_gaze = QCheckBox("Webcam gaze (blue)")
        self.chk_webcam_gaze.setChecked(True)
        self.chk_sol_gaze = QCheckBox("Sol mapped gaze (green)")
        self.chk_sol_gaze.setChecked(True)
        self.chk_sol_raw = QCheckBox("Sol raw gaze (yellow)")
        self.chk_sol_raw.setChecked(True)
        for chk in (self.chk_webcam_gaze, self.chk_sol_gaze, self.chk_sol_raw):
            chk.stateChanged.connect(lambda _: self.overlay_changed.emit())
            ol.addWidget(chk)
        layout.addWidget(grp_overlay)

        # Data quality (validity %) summary
        grp_qual = QGroupBox("Data Quality (valid %)")
        ql = QVBoxLayout(grp_qual)
        self.qual_label = QLabel("No session loaded")
        self.qual_label.setWordWrap(True)
        self.qual_label.setTextFormat(Qt.TextFormat.RichText)
        self.qual_label.setStyleSheet("font-size: 11px;")
        ql.addWidget(self.qual_label)
        layout.addWidget(grp_qual)

        # Review (human-in-the-loop labeling)
        self._suppress = False  # guard against signals while populating programmatically
        grp_review = QGroupBox("Review")
        rv = QVBoxLayout(grp_review)

        rev_row = QHBoxLayout()
        rev_row.addWidget(QLabel("Reviewer:"))
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("name / id")
        self.reviewer_edit.editingFinished.connect(
            lambda: (not self._suppress) and self.reviewer_changed.emit(self.reviewer_edit.text().strip()))
        rev_row.addWidget(self.reviewer_edit)
        rv.addLayout(rev_row)

        rv.addWidget(QLabel("Whole record:"))
        rec_row = QHBoxLayout()
        self.record_keep_btn = QRadioButton("Keep")
        self.record_discard_btn = QRadioButton("Discard")
        self.record_group = QButtonGroup(self)
        self.record_group.addButton(self.record_keep_btn)
        self.record_group.addButton(self.record_discard_btn)
        self.record_keep_btn.toggled.connect(
            lambda on: on and not self._suppress and self.record_label_changed.emit("keep"))
        self.record_discard_btn.toggled.connect(
            lambda on: on and not self._suppress and self.record_label_changed.emit("discard"))
        rec_row.addWidget(self.record_keep_btn)
        rec_row.addWidget(self.record_discard_btn)
        rec_row.addStretch()
        rv.addLayout(rec_row)

        self.record_note_edit = QLineEdit()
        self.record_note_edit.setPlaceholderText("record note (optional)")
        self.record_note_edit.editingFinished.connect(
            lambda: (not self._suppress) and self.record_note_changed.emit(self.record_note_edit.text()))
        rv.addWidget(self.record_note_edit)

        self.review_progress_lbl = QLabel("Progress: – / –")
        self.review_progress_lbl.setStyleSheet("font-size: 11px;")
        rv.addWidget(self.review_progress_lbl)
        self.review_saved_lbl = QLabel("")
        self.review_saved_lbl.setStyleSheet("color: #4CAF50; font-size: 11px;")
        rv.addWidget(self.review_saved_lbl)
        layout.addWidget(grp_review)

        # Trial list + per-trial label buttons
        grp_trials = QGroupBox("Trials")
        tl = QVBoxLayout(grp_trials)
        self.trial_list = QListWidget()
        self.trial_list.itemClicked.connect(self._on_trial_clicked)
        tl.addWidget(self.trial_list)
        btn_row = QHBoxLayout()
        self.btn_pass = QPushButton("Pass (1)")
        self.btn_fail = QPushButton("Fail (2)")
        self.btn_discard = QPushButton("Discard (3)")
        self.btn_pass.clicked.connect(lambda: self.trial_label_set.emit("pass"))
        self.btn_fail.clicked.connect(lambda: self.trial_label_set.emit("fail"))
        self.btn_discard.clicked.connect(lambda: self.trial_label_set.emit("discard"))
        for b in (self.btn_pass, self.btn_fail, self.btn_discard):
            btn_row.addWidget(b)
        tl.addLayout(btn_row)
        layout.addWidget(grp_trials)

        layout.addStretch()

    def set_session_path(self, path: str):
        self.path_label.setText(os.path.basename(path))
        self.path_label.setToolTip(path)

    def populate_trials(self, trials_df, review=None):
        self.trial_list.clear()
        self._trial_meta = {}
        if trials_df is None:
            return
        for _, row in trials_df.iterrows():
            tnum = row.get('trial_number', '?')
            cpd = row.get('cpd', '?')
            side = row.get('side', '?')
            self._trial_meta[str(tnum)] = (cpd, side)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, float(row['start_norm']))
            try:
                item.setData(Qt.ItemDataRole.UserRole + 1, int(tnum))
            except (ValueError, TypeError):
                item.setData(Qt.ItemDataRole.UserRole + 1, tnum)
            self._apply_trial_item(item, tnum, cpd, side, review)
            self.trial_list.addItem(item)

    def _apply_trial_item(self, item, tnum, cpd, side, review):
        s = str(side)[:1].upper() if side is not None else "?"
        tr = (review or {}).get("trials", {}).get(str(tnum), {})
        label = tr.get("label", "?")
        auto = tr.get("auto_result", "")
        mark = "✓" if tr.get("reviewed") else "·"  # ✓ / ·
        item.setText(f"#{tnum}  {cpd}cpd {s}   {auto}→{label} {mark}")
        item.setForeground(QColor(self.LABEL_COLORS.get(label, "#bbbbbb")))

    def refresh_trial(self, tnum, review):
        for i in range(self.trial_list.count()):
            item = self.trial_list.item(i)
            if str(item.data(Qt.ItemDataRole.UserRole + 1)) == str(tnum):
                cpd, side = self._trial_meta.get(str(tnum), ('?', '?'))
                self._apply_trial_item(item, tnum, cpd, side, review)
                return

    def current_trial_number(self):
        item = self.trial_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole + 1) if item is not None else None

    def set_review_state(self, review, progress):
        self._suppress = True
        try:
            rec = (review or {}).get("record", {})
            self.reviewer_edit.setText((review or {}).get("reviewer", ""))
            self.record_note_edit.setText(rec.get("note", ""))
            if rec.get("label", "keep") == "discard":
                self.record_discard_btn.setChecked(True)
            else:
                self.record_keep_btn.setChecked(True)
        finally:
            self._suppress = False
        self.set_progress(progress)
        self.review_saved_lbl.setText("")

    def set_progress(self, progress):
        done, total = progress
        self.review_progress_lbl.setText(f"Progress: {done} / {total} trials reviewed")

    def set_saved(self, time_str):
        if time_str:
            self.review_saved_lbl.setText(f"● Saved {time_str}")

    def set_validity_summary(self, summary):
        """summary: {'sol': {'overall','trial','n'}, 'webcam': {...}} (from PlaybackEngine.validity_summary)."""
        if not summary:
            self.qual_label.setText(
                "<span style='color:#888'>No validity data<br>"
                "(no Sol / webcam_quality.csv)</span>")
            return

        def band_color(p):
            if p is None:
                return "#888"
            if p >= 80:
                return "#4CAF50"
            if p >= 50:
                return "#FFC107"
            return "#F44336"

        def fmt(p):
            return f"{p:.0f}%" if p is not None else "N/A"

        rows = []
        for key, label in (('sol', 'Sol (combined gaze)'),
                           ('webcam', 'Webcam (face + eyes)')):
            s = summary.get(key)
            if not s:
                continue
            ov, tr = s.get('overall'), s.get('trial')
            rows.append(
                f"<b>{label}</b><br>"
                f"&nbsp;whole: <span style='color:{band_color(ov)}'><b>{fmt(ov)}</b></span>"
                f"&nbsp;&nbsp;trials: <span style='color:{band_color(tr)}'><b>{fmt(tr)}</b></span>"
            )
        self.qual_label.setText("<br>".join(rows) if rows else
                                "<span style='color:#888'>No validity data</span>")

    def _on_trial_clicked(self, item: QListWidgetItem):
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is not None:
            self.trial_selected.emit(t)


# ---------------------------------------------------------------------------
# ReplayerApp — main window
# ---------------------------------------------------------------------------
class ReplayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NTUH Eye Tracking Replayer")
        self.resize(1400, 800)

        # Engine
        self.engine = PlaybackEngine(self)

        # Trial-follow state: highlight the trial the playhead is in (set on session load)
        self._trial_start_norms = None
        self._follow_trial_row = -1

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # -- Menu bar -------------------------------------------------------
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Session...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_session)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # -- Main splitter (Config | Videos) --------------------------------
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Config panel
        self.config_panel = ConfigPanel()
        self.main_splitter.addWidget(self.config_panel)

        # Video area splitter (Main | Side stack)
        self.video_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.screen_display = VideoDisplayWidget("screen", "Screen")
        self.video_splitter.addWidget(self.screen_display)

        # Side stack (webcam on top, sol on bottom)
        self.side_splitter = QSplitter(Qt.Orientation.Vertical)
        self.webcam_display = VideoDisplayWidget("webcam", "Webcam")
        self.sol_display = VideoDisplayWidget("sol", "Sol")
        self.side_splitter.addWidget(self.webcam_display)
        self.side_splitter.addWidget(self.sol_display)
        self.video_splitter.addWidget(self.side_splitter)

        # Set initial proportions: main ~70%, side ~30%
        self.video_splitter.setSizes([700, 300])
        self.side_splitter.setSizes([250, 250])

        self.main_splitter.addWidget(self.video_splitter)
        self.main_splitter.setSizes([260, 1140])

        root_layout.addWidget(self.main_splitter, 1)

        # -- Transport controls ---------------------------------------------
        self.transport = TransportControls(self.engine)
        root_layout.addWidget(self.transport)

        # -- Timeline -------------------------------------------------------
        self.timeline = TimelineWidget()
        root_layout.addWidget(self.timeline)

        # -- Status bar -----------------------------------------------------
        self.statusBar().showMessage("Ready — Open a session folder to begin")

        # -- Wiring ---------------------------------------------------------
        self.engine.time_changed.connect(self._on_time_changed)
        self.engine.session_loaded.connect(self._on_session_loaded)
        self.timeline.seek_requested.connect(self.engine.seek)
        self.config_panel.open_folder_requested.connect(self._open_session)
        self.config_panel.trial_selected.connect(self.engine.seek)
        self.config_panel.reviewer_changed.connect(self._on_reviewer_changed)
        self.config_panel.record_label_changed.connect(self._on_record_label_changed)
        self.config_panel.record_note_changed.connect(self._on_record_note_changed)
        self.config_panel.trial_label_set.connect(self._on_trial_label_set)

        # Video display widgets list for easy iteration
        self._displays = [self.screen_display, self.webcam_display, self.sol_display]
        for d in self._displays:
            d.stream_swapped.connect(self._on_streams_swapped)

        # -- Keyboard shortcuts ---------------------------------------------
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self.engine.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_A), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock - 1.0))
        QShortcut(QKeySequence(Qt.Key.Key_D), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock + 1.0))
        QShortcut(QKeySequence(Qt.Key.Key_S), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock - 5.0))
        QShortcut(QKeySequence(Qt.Key.Key_W), self).activated.connect(
            lambda: self.engine.seek(self.engine.master_clock + 5.0))
        QShortcut(QKeySequence(Qt.Key.Key_R), self).activated.connect(self.engine.restart)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self).activated.connect(
            lambda: self.engine.set_speed(self.engine.playback_speed - 0.25))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self).activated.connect(
            lambda: self.engine.set_speed(self.engine.playback_speed + 0.25))

        # Review labeling: 1=pass 2=fail 3=discard (label current trial, auto-advance); N/B = next/prev trial
        QShortcut(QKeySequence(Qt.Key.Key_1), self).activated.connect(lambda: self._on_trial_label_set("pass"))
        QShortcut(QKeySequence(Qt.Key.Key_2), self).activated.connect(lambda: self._on_trial_label_set("fail"))
        QShortcut(QKeySequence(Qt.Key.Key_3), self).activated.connect(lambda: self._on_trial_label_set("discard"))
        QShortcut(QKeySequence(Qt.Key.Key_N), self).activated.connect(lambda: self._step_trial(1))
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(lambda: self._step_trial(-1))

        # -- Stylesheet (dark theme) ----------------------------------------
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e1e; color: #ddd; }
            QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 8px;
                        padding-top: 14px; font-weight: bold; color: #ccc; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton { background: #333; border: 1px solid #555; border-radius: 4px;
                          padding: 4px 10px; color: #ddd; }
            QPushButton:hover { background: #444; }
            QPushButton:pressed { background: #555; }
            QCheckBox { spacing: 6px; color: #ccc; }
            QListWidget { background: #2a2a2a; border: 1px solid #444; border-radius: 4px;
                          color: #ddd; font-family: Consolas; font-size: 12px; }
            QListWidget::item:selected { background: #0078d4; }
            QLabel { color: #ccc; }
            QSplitter::handle { background: #444; }
            QSplitter::handle:horizontal { width: 3px; }
            QSplitter::handle:vertical { height: 3px; }
            QStatusBar { background: #252525; color: #888; font-size: 11px; }
            QMenuBar { background: #2a2a2a; color: #ccc; }
            QMenuBar::item:selected { background: #0078d4; }
            QMenu { background: #2a2a2a; color: #ccc; border: 1px solid #555; }
            QMenu::item:selected { background: #0078d4; }
        """)

    # -- Session loading ----------------------------------------------------

    def _open_session(self):
        d = QFileDialog.getExistingDirectory(self, "Select Session Output Directory",
                                             os.path.join(os.path.dirname(__file__), "VA_output"))
        if not d:
            return
        if self.engine.load_session(d):
            self.statusBar().showMessage(f"Loaded: {os.path.basename(d)}")
        else:
            self.statusBar().showMessage("Failed to load session — no valid timestamps found")

    def _on_session_loaded(self):
        self.config_panel.set_session_path(self.engine.session_dir)
        self.timeline.set_duration(self.engine.duration)

        # Populate trial markers on timeline and config list
        self._trial_start_norms = None
        self._follow_trial_row = -1
        if self.engine.trial_events_df is not None:
            self.timeline.set_trials(self._build_timeline_trials())
            self.config_panel.populate_trials(self.engine.trial_events_df, self.engine.review)
            try:
                self._trial_start_norms = self.engine.trial_events_df['start_norm'].to_numpy(dtype=float)
            except Exception:
                self._trial_start_norms = None

        # Scale screen-display gaze overlays from original screen res to the (downscaled) video
        self.screen_display.set_gaze_source_size(self.engine.screen_width, self.engine.screen_height)

        # Review labels (human-in-the-loop)
        self.config_panel.set_review_state(self.engine.review, self.engine.review_progress())

        # Validity strips on the timeline + summary numbers in the side panel
        try:
            varrs = self.engine._validity_arrays()
            self.timeline.set_validity(varrs.get('sol'), varrs.get('webcam'))
            self.config_panel.set_validity_summary(self.engine.validity_summary())
        except Exception as e:
            print(f"[Replayer] validity setup error: {e}")

        # Force a frame update at t=0
        self.engine.time_changed.emit(0.0)

    # -- Review labeling ----------------------------------------------------

    def _build_timeline_trials(self):
        """List of (start_norm, end_norm, result, trial_number, cpd, review_label) for the timeline."""
        df = self.engine.trial_events_df
        out = []
        if df is None:
            return out
        rev = (self.engine.review or {}).get("trials", {})
        for _, row in df.iterrows():
            tnum = row.get('trial_number', 0)
            label = rev.get(str(tnum), {}).get('label')
            out.append((row['start_norm'], row['end_norm'], row.get('result', '?'),
                        tnum, row.get('cpd', 0), label))
        return out

    def _row_for_tnum(self, tnum):
        lst = self.config_panel.trial_list
        for i in range(lst.count()):
            if str(lst.item(i).data(Qt.ItemDataRole.UserRole + 1)) == str(tnum):
                return i
        return -1

    def _trial_at_time(self, t):
        df = self.engine.trial_events_df
        if df is None:
            return None
        for _, row in df.iterrows():
            if row['start_norm'] <= t <= row['end_norm']:
                return row.get('trial_number')
        return None

    def _step_trial(self, delta):
        """Select the next/previous trial row and seek to its start."""
        lst = self.config_panel.trial_list
        n = lst.count()
        if n == 0:
            return
        cur = lst.currentRow()
        nxt = 0 if cur < 0 else max(0, min(n - 1, cur + delta))
        lst.setCurrentRow(nxt)
        item = lst.item(nxt)
        if item is not None:
            seek = item.data(Qt.ItemDataRole.UserRole)
            if seek is not None:
                self.engine.seek(float(seek))

    def _on_reviewer_changed(self, name):
        self.config_panel.set_saved(self.engine.set_reviewer(name))

    def _on_record_label_changed(self, label):
        self.config_panel.set_saved(self.engine.set_record_label(label))

    def _on_record_note_changed(self, note):
        self.config_panel.set_saved(self.engine.set_record_note(note))

    def _on_trial_label_set(self, label):
        if not self.engine.review:
            return
        tnum = self.config_panel.current_trial_number()
        if tnum is None:
            tnum = self._trial_at_time(self.engine.master_clock)
        if tnum is None:
            if self.config_panel.trial_list.count() == 0:
                return
            self.config_panel.trial_list.setCurrentRow(0)
            tnum = self.config_panel.current_trial_number()
        # Make the labeled trial the current row (so auto-advance steps from it)
        row = self._row_for_tnum(tnum)
        if row >= 0:
            self.config_panel.trial_list.setCurrentRow(row)
        saved = self.engine.set_trial_label(tnum, label)
        self.config_panel.refresh_trial(tnum, self.engine.review)
        self.config_panel.set_progress(self.engine.review_progress())
        self.config_panel.set_saved(saved)
        self.timeline.set_trials(self._build_timeline_trials())  # recolor marker
        self._step_trial(1)  # auto-advance to next trial

    def _follow_playhead(self, t):
        """Select the trial row the playhead is in (latest trial whose start <= t).
        Uses setCurrentRow (no seek, no focus steal) and only updates on change so it
        does not fight manual selection or text entry."""
        starts = self._trial_start_norms
        if starts is None or len(starts) == 0:
            return
        idx = int(np.searchsorted(starts, t, side='right') - 1)  # -1 before the first trial
        if idx == self._follow_trial_row:
            return
        self._follow_trial_row = idx
        if idx < 0:
            return
        lst = self.config_panel.trial_list
        if 0 <= idx < lst.count():
            lst.setCurrentRow(idx)
            item = lst.item(idx)
            if item is not None:
                lst.scrollToItem(item)

    # -- Per-tick update ----------------------------------------------------

    def _on_time_changed(self, t: float):
        # Update transport time display
        self.transport.update_time(t, self.engine.duration)
        self.timeline.set_time(t)

        # Highlight the trial the playhead is currently in (follow playback)
        self._follow_playhead(t)

        # Fetch gaze data once
        gaze_data = self.engine.get_data_at_time(t)

        # Update each video display
        for disp in self._displays:
            name = disp.stream_name
            ctrl = self.engine.controllers.get(name)
            if ctrl:
                frame = ctrl.get_frame_at_time(t)
                disp.set_frame(frame)
            else:
                disp.set_frame(None)

            # Build gaze points for this display
            pts = []
            if name == "screen":
                # Webcam gaze (blue) on screen
                if self.config_panel.chk_webcam_gaze.isChecked() and 'webcam' in gaze_data:
                    row = gaze_data['webcam']
                    try:
                        wx, wy = row['webcam_gaze_x'], row['webcam_gaze_y']
                        if pd.notna(wx) and pd.notna(wy):
                            pts.append((float(wx), float(wy), QColor(80, 140, 255), "Webcam"))
                    except Exception:
                        pass

                # Sol mapped gaze (green) on screen
                if self.config_panel.chk_sol_gaze.isChecked() and 'sol' in gaze_data:
                    row = gaze_data['sol']
                    try:
                        gx, gy = row['mapped_gaze_x'], row['mapped_gaze_y']
                        valid = row['is_valid']
                        if pd.notna(gx) and pd.notna(gy) and valid == 1:
                            pts.append((float(gx), float(gy), QColor(76, 175, 80), "Sol"))
                    except Exception:
                        pass

            elif name == "sol":
                # Sol raw gaze (yellow) on sol video
                if self.config_panel.chk_sol_raw.isChecked() and 'sol' in gaze_data:
                    row = gaze_data['sol']
                    try:
                        rx, ry = row['raw_gaze_x'], row['raw_gaze_y']
                        valid = row['is_valid']
                        if pd.notna(rx) and pd.notna(ry) and valid == 1:
                            pts.append((float(rx), float(ry), QColor(255, 235, 59), "Gaze"))
                    except Exception:
                        pass

            disp.set_gaze_points(pts)

    # -- D&D rewire ---------------------------------------------------------

    def _on_streams_swapped(self):
        # After a swap, immediately re-render with current time
        self._on_time_changed(self.engine.master_clock)

    # -- Cleanup ------------------------------------------------------------

    def closeEvent(self, event):
        self.engine.pause()
        for c in self.engine.controllers.values():
            c.release()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReplayerApp()
    window.show()
    sys.exit(app.exec())
