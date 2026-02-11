# -*- coding: utf-8 -*-
import logging
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging

import os
import cv2
import pygame
import numpy as np
import time
import sys
import math
import random
import threading
import queue
import asyncio
import json
import traceback
from collections import deque
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from pathlib import Path

from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log as GFLog
from gazefollower.camera import WebCamCamera

# [NEW] Imports for Sol Glasses & Recorder
try:
    from sol_tracker import SolConnector, ScreenProjector3D, create_calibration_assets, SDK_AVAILABLE
except ImportError:
    SDK_AVAILABLE = False
    print("Warning: sol_tracker module not found or dependencies missing (ganzin_sol_sdk). Sol features disabled.")

# [NEW] Sol Offset Calibration (3D angular offset)
try:
    from sol_offset_calibration import (
        apply_angular_offset, load_sol_offset, save_sol_offset, clear_sol_offset,
        SolOffsetCalibrator
    )
    SOL_OFFSET_AVAILABLE = True
except ImportError:
    SOL_OFFSET_AVAILABLE = False
    print("Warning: sol_offset_calibration module not found. Sol offset calibration disabled.")

# [NEW] Sol 2D Offset Calibration (IDW-based with angular support)
try:
    from sol_2d_offset_calibration import (
        Sol2DOffsetCalibrator, Sol2DOffsetModel,
        load_sol_2d_offset, save_sol_2d_offset, clear_sol_2d_offset,
        find_target_in_frame, find_target_multiscale,
        CALIBRATION_POSITIONS_2D, compute_safe_calibration_positions,
        OFFSET_MODE_PIXEL, OFFSET_MODE_ANGULAR  # Offset modes
    )
    SOL_2D_OFFSET_AVAILABLE = True
except ImportError:
    SOL_2D_OFFSET_AVAILABLE = False
    OFFSET_MODE_PIXEL = 'pixel'
    OFFSET_MODE_ANGULAR = 'angular'
    print("Warning: sol_2d_offset_calibration module not found. Sol 2D offset calibration disabled.")

from recorder import Recorder


class DummyRecorder:
    """Dummy recorder for practice mode - does nothing but provides the same interface."""

    def __init__(self, *args, **kwargs):
        self.frame_count = 0
        self.running = True  # Mimic real recorder's running flag

    def process_and_record(self, *args, **kwargs):
        self.frame_count += 1

    def close(self):
        self.running = False
        print("[Practice Mode] DummyRecorder closed (no data recorded)")

# [NEW] Imports for Webcam Preview
try:
    from PIL import Image, ImageTk
    import mediapipe as mp
except ImportError:
    print("Warning: PIL or mediapipe not found. Webcam Preview might fail.")

LAST_SETTINGS_FILE = Path(__file__).resolve().parent / "VA_output" / "last_settings_opt.json"

# [NEW] Global Crash Handler
import ctypes
def global_exception_handler(exctype, value, tb):
    import traceback
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    full_msg = f"[{timestamp}] CRITICAL UNHANDLED EXCEPTION:\n{err_msg}\n"
    print(full_msg, file=sys.stderr)
    try:
        with open("va_crash_log.txt", "a") as f:
            f.write(full_msg + "\n" + "="*40 + "\n")
        messagebox.showerror("Critical Error", f"Application Crashed!\nSee va_crash_log.txt\n\n{value}")
    except: pass

sys.excepthook = global_exception_handler
threading.excepthook = lambda args: global_exception_handler(args.exc_type, args.exc_value, args.exc_traceback)

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

# ---------- Monitor Info Utils ----------
def get_monitor_info_windows():
    """
    Get detailed monitor information on Windows using ctypes.
    Returns a list of dicts with: index, name, x, y, width, height
    """
    monitors = []
    try:
        import ctypes
        from ctypes import wintypes

        # Define necessary structures
        class DISPLAY_DEVICE(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('DeviceName', wintypes.WCHAR * 32),
                ('DeviceString', wintypes.WCHAR * 128),
                ('StateFlags', wintypes.DWORD),
                ('DeviceID', wintypes.WCHAR * 128),
                ('DeviceKey', wintypes.WCHAR * 128),
            ]

        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ('dmDeviceName', wintypes.WCHAR * 32),
                ('dmSpecVersion', wintypes.WORD),
                ('dmDriverVersion', wintypes.WORD),
                ('dmSize', wintypes.WORD),
                ('dmDriverExtra', wintypes.WORD),
                ('dmFields', wintypes.DWORD),
                ('dmPositionX', wintypes.LONG),
                ('dmPositionY', wintypes.LONG),
                ('dmDisplayOrientation', wintypes.DWORD),
                ('dmDisplayFixedOutput', wintypes.DWORD),
                ('dmColor', wintypes.SHORT),
                ('dmDuplex', wintypes.SHORT),
                ('dmYResolution', wintypes.SHORT),
                ('dmTTOption', wintypes.SHORT),
                ('dmCollate', wintypes.SHORT),
                ('dmFormName', wintypes.WCHAR * 32),
                ('dmLogPixels', wintypes.WORD),
                ('dmBitsPerPel', wintypes.DWORD),
                ('dmPelsWidth', wintypes.DWORD),
                ('dmPelsHeight', wintypes.DWORD),
                ('dmDisplayFlags', wintypes.DWORD),
                ('dmDisplayFrequency', wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32

        # Enumerate display devices
        i = 0
        while True:
            device = DISPLAY_DEVICE()
            device.cb = ctypes.sizeof(device)

            if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(device), 0):
                break

            # Check if this is an active display
            DISPLAY_DEVICE_ACTIVE = 0x00000001
            if device.StateFlags & DISPLAY_DEVICE_ACTIVE:
                # Get display settings for position and size
                devmode = DEVMODE()
                devmode.dmSize = ctypes.sizeof(devmode)

                if user32.EnumDisplaySettingsW(device.DeviceName, -1, ctypes.byref(devmode)):  # ENUM_CURRENT_SETTINGS = -1
                    # Get monitor name (try to get the actual monitor device)
                    monitor_device = DISPLAY_DEVICE()
                    monitor_device.cb = ctypes.sizeof(monitor_device)
                    monitor_name = device.DeviceString.strip()

                    # Try to get actual monitor name
                    if user32.EnumDisplayDevicesW(device.DeviceName, 0, ctypes.byref(monitor_device), 0):
                        if monitor_device.DeviceString.strip():
                            monitor_name = monitor_device.DeviceString.strip()

                    monitors.append({
                        'index': len(monitors),
                        'device_name': device.DeviceName.strip(),
                        'name': monitor_name,
                        'x': devmode.dmPositionX,
                        'y': devmode.dmPositionY,
                        'width': devmode.dmPelsWidth,
                        'height': devmode.dmPelsHeight,
                    })
            i += 1

    except Exception as e:
        print(f"[Monitor Info] Error getting monitor info: {e}")

    # Fallback if no monitors found
    if not monitors:
        monitors = [
            {'index': 0, 'name': 'Primary Display', 'x': 0, 'y': 0, 'width': 1920, 'height': 1080},
            {'index': 1, 'name': 'Secondary Display', 'x': 1920, 'y': 0, 'width': 1920, 'height': 1080},
        ]

    return monitors


# ---------- Utils ----------
def restore_event_filter():
    try:
        pygame.event.set_allowed(None)
        pygame.event.clear()
        pygame.event.pump()
    except Exception:
        pass

def ensure_pygame_focus(timeout=2.0):
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout:
            break
        time.sleep(0.02)

def to_rgb_tuple(rgb_like):
    return tuple(int(v) for v in rgb_like)

def screen_width_deg_from_cm(width_cm: float, dist_cm: float) -> float:
    if dist_cm <= 0 or width_cm <= 0:
        return 0.0
    return 2.0 * math.degrees(math.atan((width_cm / 2.0) / dist_cm))

def mean_color_rgb(light, dark):
    l = np.array(light, dtype=np.uint16)
    d = np.array(dark,  dtype=np.uint16)
    return tuple(((l + d) // 2).astype(np.uint8))

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

# ---------- Staircase（cpd） ----------
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

# ---------- GUI ----------
# --- Helper Classes for Webcam Preview ---
class Camera:
    @staticmethod
    def list_cameras(max_cameras=10):
        available_cameras = []
        # Fast probe using DirectShow for speed on Windows
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except: pass
        return available_cameras

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width, self.height, self.fps = width, height, fps
        self.cap = None
        self.running = False
        self.ready = False  # True once first frame is captured
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self):
        if self.running: return
        self.running = True
        self.ready = False
        # Open camera in the capture thread to avoid blocking the UI
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        # Open camera in this thread (DirectShow is faster on Windows)
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not self.cap.isOpened():
                print(f"WARN: Could not open camera {self.camera_id}")
                self.running = False
                return
        except Exception as e:
            print(f"WARN: Camera init error: {e}")
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
                if not self.ready:
                    self.ready = True
                    print(f"[Camera] Camera {self.camera_id} ready")
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None


class ScrollableFrame(ttk.Frame):
    """A scrollable frame container for use in notebook tabs."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bind canvas resize to adjust inner frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel for scrolling
        self.scrollable_frame.bind("<Enter>", self._bind_mousewheel)
        self.scrollable_frame.bind("<Leave>", self._unbind_mousewheel)

    def _on_canvas_configure(self, event):
        # Make the inner frame width match the canvas width
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VA Test Settings (Optimization)")
        self.resizable(True, True)

        # [NEW] Dynamic font size based on screen resolution
        screen_height = self.winfo_screenheight()
        screen_width = self.winfo_screenwidth()

        # Scale font size based on screen height
        # 1080p needs smaller fonts to fit everything
        if screen_height <= 900:
            font_size = 8
            window_height = 580
            window_width = 750
        elif screen_height <= 1080:
            font_size = 8
            window_height = 680
            window_width = 800
        elif screen_height <= 1200:
            font_size = 9
            window_height = 750
            window_width = 850
        elif screen_height <= 1440:
            font_size = 10
            window_height = 820
            window_width = 950
        else:  # 4K and above
            font_size = 11
            window_height = 900
            window_width = 1020

        self.geometry(f"{window_width}x{window_height}")
        print(f"[UI] Screen: {screen_width}x{screen_height}, Font size: {font_size}, Window: {window_width}x{window_height}")

        self.ui_font_size = font_size  # Store for later use
        # Dynamic padding based on screen size
        self.ui_pad = {'padx': 5 if screen_height <= 1080 else 10, 'pady': 2 if screen_height <= 1080 else 5}
        LABEL_FONT = ("Arial", font_size)
        ENTRY_FONT = ("Arial", font_size)

        # 預設校正資料夾
        self.default_calib_dir = Path(__file__).resolve().parent / "calibration_profiles"
        self.default_calib_dir.mkdir(parents=True, exist_ok=True)
        self.calib_dir_var = tk.StringVar(value=str(self.default_calib_dir))

        # --- General Vars ---
        self.user_var = tk.StringVar(value="anonymous")
        self.gaze_color_var  = tk.StringVar(value="0,255,0")
        self.gaze_radius_var = tk.StringVar(value="30")
        self.gaze_width_var  = tk.StringVar(value="4")
        self.stim_var   = tk.StringVar(value="5.0")
        self.pass_dur_var = tk.StringVar(value="2.0")
        self.blank_var  = tk.StringVar(value="1.0")
        self.rad_var    = tk.StringVar(value="400")
        self.rotate_var = tk.BooleanVar(value=False)
        self.rot_speed_var = tk.StringVar(value="60.0")
        self.rot_dir_var   = tk.StringVar(value="CW")
        self.color_light_var = tk.StringVar(value="255,255,255")
        self.color_dark_var  = tk.StringVar(value="0,0,0")
        self.bg_color_var    = tk.StringVar(value="0,0,0")
        self.scr_width_cm_var = tk.StringVar(value="53.0")
        self.view_dist_cm_var = tk.StringVar(value="120.0")
        self.interval_img_path_var = tk.StringVar(value="")
        self.interval_img_dur_var  = tk.StringVar(value="1.5")
        self.bg_after_inter_dur_var= tk.StringVar(value="1.0")
        
        # [NEW] Dual Tracker Vars
        self.enable_webcam_var = tk.BooleanVar(value=True)
        self.enable_sol_var = tk.BooleanVar(value=False)
        self.eval_source_var = tk.StringVar(value="Webcam") # Webcam, Sol, Both (Both logic TBD, usually primary)

        # [NEW] Sol Vars
        self.sol_ip_var = tk.StringVar(value="192.168.1.100")
        self.sol_port_var = tk.StringVar(value="8080")
        self.sol_marker_k_var = tk.StringVar(value="6")
        self.sol_marker_n_var = tk.StringVar(value="4")
        self.sol_marker_size_var = tk.StringVar(value="80")
        self.sol_aruco_dict_var = tk.StringVar(value="DICT_4X4_250")
        # sol_screen_phy_width_var removed, using scr_width_cm_var * 10
        self.sol_pose_smooth_var = tk.StringVar(value="0.1")
        self.sol_gaze_smooth_var = tk.StringVar(value="0.15")
        self.sol_gaze_method_var = tk.StringVar(value="3D")  # "3D" or "2D"
        self.sol_cal_show_gaze_var = tk.BooleanVar(value=True)  # Show gaze during calibration

        # [NEW] Connection State
        self.is_sol_connected = False
        self.active_sol_connector = None
        self.sol_thread = None
        self.sol_gaze_queue = None
        self.sol_scene_queue = None
        self.sol_cam_params = None
        self.sol_cached_homography = None  # Cache homography between preview/calibration sessions

        # [NEW] Recording Vars
        self.rec_resolution_var = tk.StringVar(value="Original") # Original, 1920x1080, 1280x720
        # [NEW] Recording Vars - Optimized
        self.rec_resolution_var = tk.StringVar(value="Original")
        self.rec_webcam_var = tk.BooleanVar(value=True) # Webcam Video + Gaze
        self.rec_sol_data_var = tk.BooleanVar(value=False) # Sol Gaze (+ Screen implicity)
        self.rec_sol_raw_video_var = tk.BooleanVar(value=False) # Only if Sol Data is checked

        # [NEW] Preview Config (Init early for Load Last)
        self.camera_idx_var = tk.StringVar(value="0")
        self.preview_running = False
        self.camera_helper = None
        self.face_aligner = None
        
        # [NEW] Gaze Marker Toggle
        self.show_gaze_marker_var = tk.BooleanVar(value=True)

        # [NEW] Paper Color Mode - gray bg, black/white grating, white border
        self.paper_color_var = tk.BooleanVar(value=False)

        # [NEW] Sol Offset Calibration Vars
        self.sol_offset_target_img_var = tk.StringVar(value="")
        self.sol_offset_target_size_var = tk.StringVar(value="100")
        self.sol_offset_num_points_var = tk.StringVar(value="5")
        self.sol_offset_user_screen_var = tk.StringVar(value="0")
        self.sol_offset_tester_screen_var = tk.StringVar(value="1")

        self.cfg = None

        # Auto-save control
        self._auto_save_timer = None
        self._suppress_auto_save = False  # Suppress during bulk loading
        self._flush_sol_timer = None  # Track flush_sol_queues timer for cleanup

        # Validation Registration
        self.vcmd_int = (self.register(self.validate_int), '%P')
        self.vcmd_float = (self.register(self.validate_float), '%P')
        
        # --- Layout ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)

        # [FIX] Use scrollable frames for tabs to support smaller screens
        self.tab_general_scroll = ScrollableFrame(self.notebook)
        self.tab_sol_scroll = ScrollableFrame(self.notebook)
        self.tab_rec_scroll = ScrollableFrame(self.notebook)

        self.tab_general = self.tab_general_scroll.scrollable_frame
        self.tab_sol = self.tab_sol_scroll.scrollable_frame
        self.tab_rec = self.tab_rec_scroll.scrollable_frame

        self.notebook.add(self.tab_general_scroll, text='General')
        self.notebook.add(self.tab_sol_scroll, text='Sol')
        self.notebook.add(self.tab_rec_scroll, text='Recording')

        self.build_general_tab(self.tab_general, LABEL_FONT, ENTRY_FONT)
        self.build_sol_tab(self.tab_sol, LABEL_FONT, ENTRY_FONT)
        self.build_rec_tab(self.tab_rec, LABEL_FONT, ENTRY_FONT)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="bottom", pady=10)
        self.btn_start = ttk.Button(btn_frame, text="Start Test", command=self.on_start, state="disabled")
        self.btn_start.pack(side="right", padx=10)
        self.btn_practice = ttk.Button(btn_frame, text="Start Practice", command=self.on_start_practice, state="disabled")
        self.btn_practice.pack(side="right", padx=10)

        # Initial Check
        self.check_start_button_state()

        # Monitor changes for button state
        self.enable_webcam_var.trace_add('write', lambda *args: self.check_start_button_state())
        self.enable_sol_var.trace_add('write', lambda *args: self.check_start_button_state())
        
        # [NEW] Webcam Tab
        self.tab_webcam = ttk.Frame(self.notebook)
        # Insert before Rec
        self.notebook.insert(1, self.tab_webcam, text='Webcam Preview')
        self.build_webcam_tab(self.tab_webcam, LABEL_FONT, ENTRY_FONT)

        # [NEW] Sol Offset Calibration Tab
        self.tab_sol_offset = ttk.Frame(self.notebook)
        # Insert after Sol Settings tab (index 3)
        self.notebook.insert(3, self.tab_sol_offset, text='Sol Offset Calibration')
        self.build_sol_offset_tab(self.tab_sol_offset, LABEL_FONT, ENTRY_FONT)

        # Cleanup on close
        self.protocol("WM_DELETE_WINDOW", self.on_close_window)

        # [FIX] Force focus binding and Window Lift
        self.recursive_bind_focus(self)
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)
        self.focus_force()
        self.after(200, lambda: self.entry_user.focus_force() if hasattr(self, 'entry_user') else None)

        style = ttk.Style()
        big_font_size = self.ui_font_size + 4  # Slightly larger for big buttons
        style.configure("Big.TButton", font=("Arial", big_font_size, "bold"), padding=8)

        # Auto-load previous settings on startup (suppresses auto-save during load)
        self._auto_load_settings()

        # Setup auto-save traces on all settings variables (after load to avoid saving during load)
        self._setup_auto_save_traces()

        # Watch for username changes to update user-dependent displays (offset calibration status)
        self.user_var.trace_add("write", self._on_username_changed)

        # Update offset displays after loading settings (delayed to ensure UI is ready)
        self.after(200, self.update_sol_offset_display)
        self.after(200, self.update_sol_2d_offset_display)


    def recursive_bind_focus(self, widget):
        if isinstance(widget, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Spinbox)):
            widget.bind("<Button-1>", lambda e: e.widget.focus_force(), add="+")
        for child in widget.winfo_children():
            self.recursive_bind_focus(child)



    def build_webcam_tab(self, parent, label_font, entry_font):
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill="both", expand=True)

        # Controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0,10))
        
        ttk.Label(ctrl, text="Select Camera:", font=label_font).pack(side="left", padx=5)
        
        # Probe Cameras
        cams = Camera.list_cameras()
        if not cams: cams = [0]
        
        self.combo_cam = ttk.Combobox(ctrl, textvariable=self.camera_idx_var, values=cams, state="readonly", width=5)
        self.combo_cam.pack(side="left", padx=5)
        self.combo_cam.current(0)
        
        ttk.Button(ctrl, text="Start Preview", command=self.start_preview).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Stop Preview", command=self.stop_preview).pack(side="left", padx=10)
        


        # Preview Area
        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill="both", expand=True)
        
        # Main Video
        self.lbl_video = ttk.Label(preview_frame, text="Camera Feed", relief="sunken", anchor="center")
        self.lbl_video.pack(side="top", fill="both", expand=True, pady=5)
        
        # Eye Crops
        eyes_frame = ttk.Frame(preview_frame)
        eyes_frame.pack(side="bottom", fill="x", pady=5)
        
        self.lbl_eye_l = ttk.Label(eyes_frame, text="Left Eye", relief="sunken", width=20, anchor="center")
        self.lbl_eye_l.pack(side="left", padx=10, expand=True)
        
        self.lbl_eye_r = ttk.Label(eyes_frame, text="Right Eye", relief="sunken", width=20, anchor="center")
        self.lbl_eye_r.pack(side="right", padx=10, expand=True)

    def start_preview(self):
        if self.preview_running: self.stop_preview()
        
        try:
            cid = int(self.camera_idx_var.get())
            self.camera_helper = Camera(camera_id=cid)
            self.camera_helper.start()
            
            # Reuse GazeFollower's mpa
            # Check if mpa is the class itself or module
            if hasattr(mpa, 'MediaPipeFaceAlignment'):
                self.face_aligner = mpa.MediaPipeFaceAlignment()
            else:
                self.face_aligner = mpa()
            
            self.preview_running = True
            self.update_preview_loop()
        except Exception as e:
            print(f"Preview Start Error: {e}")
            messagebox.showerror("Error", f"Could not start preview: {e}")

    def stop_preview(self):
        self.preview_running = False
        if self.camera_helper:
            self.camera_helper.stop()
            self.camera_helper = None
        # FaceAligner doesn't need explicit close usually, or doesn't have .close() exposed in base
        # But let's check if it has close. MediaPipeFaceAlignment wrapper might not.
        self.face_aligner = None
        
        # Clear images
        self.lbl_video.configure(image='')
        self.lbl_eye_l.configure(image='')
        self.lbl_eye_r.configure(image='')

    def on_close_window(self):
        self.stop_preview()
        # Cancel pending timers before destroying window
        if self._auto_save_timer is not None:
            self.after_cancel(self._auto_save_timer)
            self._auto_save_timer = None
        if self._flush_sol_timer is not None:
            self.after_cancel(self._flush_sol_timer)
            self._flush_sol_timer = None
        self._auto_save_settings()
        self.cfg = None  # Signal main loop that user wants to exit
        self.destroy()

    # Note: on_start is defined later in the class with full Sol support (line ~2374)

    def update_preview_loop(self):
        if not self.preview_running: return
        
        frame = self.camera_helper.get_frame() if self.camera_helper else None
        
        if frame is not None:
             # Detection
             # mpa.detect(timestamp, image) -> FaceInfo
             # It expects image in BGR (step 871 line 70)
             t = time.time()
             try:
                 fi = self.face_aligner.detect(t, frame)
                 
                 # Draw if status is True
                 if fi.status:
                     # Face Info coordinates are [x, y, w, h] (Step 871 line 203+)
                     # Draw boxes on *copy* of frame
                     disp_frame = frame.copy()
                     
                     def draw_rect(img, rect, color):
                         x,y,w,h = rect
                         x, y, w, h = int(x), int(y), int(w), int(h)
                         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                         return img
                    
                     disp_frame = draw_rect(disp_frame, fi.face_rect, (0, 255, 0))
                     disp_frame = draw_rect(disp_frame, fi.left_rect, (255, 0, 0)) # Left Eye (Blue)
                     disp_frame = draw_rect(disp_frame, fi.right_rect, (0, 0, 255)) # Right Eye (Red)
                     
                     # Crops
                     def get_crop(img, rect):
                         x,y,w,h = rect
                         x, y, w, h = int(x), int(y), int(w), int(h)
                         # Clamp
                         H,W,_ = img.shape
                         x,y = max(0,x), max(0,y)
                         w = min(w, W-x)
                         h = min(h, H-y)
                         if w>0 and h>0: return img[y:y+h, x:x+w]
                         return None

                     l_crop = get_crop(frame, fi.left_rect)
                     r_crop = get_crop(frame, fi.right_rect)
                     
                     # Convert to Tk
                     self_img = self._cv2_tk(disp_frame, (480, 360)) # Resize for fitting
                     self.lbl_video.configure(image=self_img)
                     self.lbl_video.image = self_img
                     
                     if l_crop is not None:
                         l_img = self._cv2_tk(l_crop, (150, 100))
                         self.lbl_eye_l.configure(image=l_img)
                         self.lbl_eye_l.image = l_img
                     
                     if r_crop is not None:
                         r_img = self._cv2_tk(r_crop, (150, 100))
                         self.lbl_eye_r.configure(image=r_img)
                         self.lbl_eye_r.image = r_img
                         
                 else:
                     # Just frame
                     img = self._cv2_tk(frame, (480, 360))
                     self.lbl_video.configure(image=img)
                     self.lbl_video.image = img
                     
             except Exception as e:
                 print(f"Preview Error: {e}")
        
        self.after(30, self.update_preview_loop)

    def _cv2_tk(self, img, size):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_pil = im_pil.resize(size)
        return ImageTk.PhotoImage(image=im_pil)

    def validate_int(self, P):
        if P == "" or P == "-": return True
        try:
            int(P)
            return True
        except ValueError: return False

    def validate_float(self, P):
        if P == "" or P == "-": return True
        try:
            float(P)
            return True
        except ValueError: return False

    def build_general_tab(self, parent, l_font, e_font):
        pad = self.ui_pad  # Use dynamic padding
        r = 0
        
        # Original settings ...
        ttk.Label(parent, text="User name:", font=l_font).grid(row=r, sticky="w", **pad)
        self.entry_user = ttk.Entry(parent, textvariable=self.user_var, font=e_font, width=20)
        self.entry_user.grid(row=r, column=1, **pad); r += 1

        ttk.Label(parent, text="Calibration folder (Webcam):", font=l_font).grid(row=r, sticky="w", **pad)
        cdir_frame = ttk.Frame(parent); cdir_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(cdir_frame, textvariable=self.calib_dir_var, font=e_font, width=40).pack(side="left")
        def _browse_calib_dir():
            p = filedialog.askdirectory(title="Choose calibration folder", initialdir=str(self.default_calib_dir))
            if p: self.calib_dir_var.set(p)
        ttk.Button(parent, text="Browse", command=_browse_calib_dir).grid(row=r, column=2, **pad); r += 1

        # Tracker Selection
        ttk.Label(parent, text="Trackers:", font=l_font).grid(row=r, sticky="w", **pad)
        tracker_frame = ttk.Frame(parent)
        tracker_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Checkbutton(tracker_frame, text="Webcam", variable=self.enable_webcam_var).pack(side="left", padx=5)
        ttk.Checkbutton(tracker_frame, text="Sol Glasses", variable=self.enable_sol_var).pack(side="left", padx=5)
        r += 1

        ttk.Label(parent, text="Evaluation Source:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Combobox(parent, textvariable=self.eval_source_var, values=["Webcam", "Sol"], state="readonly", font=e_font).grid(row=r, column=1, **pad); r += 1

        # Stimulus & Timings
        ttk.Label(parent, text="Stim Duration (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.stim_var, from_=0.5, to=30.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1
        
        ttk.Label(parent, text="Pass Duration (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.pass_dur_var, from_=0.1, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1

        ttk.Label(parent, text="Blank Duration (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.blank_var, from_=0.2, to=10.0, increment=0.1, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1

        ttk.Label(parent, text="Circle Radius (px):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.rad_var, from_=50, to=800, increment=10, font=e_font, width=10).grid(row=r, column=1, **pad); r+=1

        # Colors
        def choose_color(tv):
            c = colorchooser.askcolor()[0]
            if c: tv.set(f"{int(c[0])},{int(c[1])},{int(c[2])}")

        ttk.Label(parent, text="Bright Color:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Entry(parent, textvariable=self.color_light_var, font=e_font, width=15).grid(row=r, column=1, **pad)
        ttk.Button(parent, text="Pick", command=lambda: choose_color(self.color_light_var)).grid(row=r, column=2, **pad); r+=1
        
        # Gaze Marker Toggle and Paper Color Mode
        gaze_frame = ttk.Frame(parent)
        gaze_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Checkbutton(gaze_frame, text="Show Gaze Marker during Test", variable=self.show_gaze_marker_var).pack(side="left")
        ttk.Checkbutton(gaze_frame, text="Paper Color", variable=self.paper_color_var).pack(side="left", padx=(20, 0))
        r += 1

        ttk.Label(parent, text="Dark Color:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Entry(parent, textvariable=self.color_dark_var, font=e_font, width=15).grid(row=r, column=1, **pad)
        ttk.Button(parent, text="Pick", command=lambda: choose_color(self.color_dark_var)).grid(row=r, column=2, **pad); r+=1

        ttk.Label(parent, text="Bg Color:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Entry(parent, textvariable=self.bg_color_var, font=e_font, width=15).grid(row=r, column=1, **pad)
        ttk.Button(parent, text="Pick", command=lambda: choose_color(self.bg_color_var)).grid(row=r, column=2, **pad); r+=1

        # Screen & Rotation
        ttk.Label(parent, text="Screen W (cm):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.scr_width_cm_var, from_=10, to=300, increment=0.5, font=e_font).grid(row=r, column=1, **pad); r+=1
        
        ttk.Label(parent, text="View Dist (cm):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.view_dist_cm_var, from_=10, to=300, increment=1, font=e_font).grid(row=r, column=1, **pad); r+=1

        ttk.Checkbutton(parent, text="Rotate Stimulus", variable=self.rotate_var).grid(row=r, column=0, sticky="w", **pad)
        ttk.Label(parent, text="Speed (d/s):", font=l_font).grid(row=r, column=1, sticky="e")
        ttk.Spinbox(parent, textvariable=self.rot_speed_var, from_=0, to=2000, increment=10, width=8).grid(row=r, column=2, sticky="w"); r+=1
        
        # Inter-trial
        ttk.Label(parent, text="Inter-trial Img:", font=l_font).grid(row=r, sticky="w", **pad)
        img_frame = ttk.Frame(parent); img_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(img_frame, textvariable=self.interval_img_path_var, font=e_font, width=30).pack(side="left")
        def _browse_img():
            p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"),("All","*.*")])
            if p: self.interval_img_path_var.set(p)
        ttk.Button(parent, text="...", command=_browse_img, width=4).grid(row=r, column=2, **pad); r+=1
        
        ttk.Label(parent, text="Inter-trial Dur (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.interval_img_dur_var, from_=0.2, to=10, increment=0.1, font=e_font).grid(row=r, column=1, **pad); r+=1
        
        ttk.Label(parent, text="Bg Hold Dur (s):", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(parent, textvariable=self.bg_after_inter_dur_var, from_=0, to=10, increment=0.1, font=e_font).grid(row=r, column=1, **pad); r+=1

    def build_sol_tab(self, parent, l_font, e_font):
        pad = self.ui_pad  # Use dynamic padding
        r = 0
        
        # Connection
        grp_conn = ttk.LabelFrame(parent, text="Connection"); grp_conn.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_conn, text="IP:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Entry(grp_conn, textvariable=self.sol_ip_var, font=e_font).grid(row=0, column=1, **pad)
        ttk.Label(grp_conn, text="Port:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Entry(grp_conn, textvariable=self.sol_port_var, font=e_font, width=8).grid(row=0, column=3, **pad)
        
        self.btn_connect_sol = ttk.Button(grp_conn, text="Connect", command=self.toggle_sol_connection)
        self.btn_connect_sol.grid(row=0, column=4, **pad)
        self.lbl_sol_status = ttk.Label(grp_conn, text="Not Connected", foreground="red")
        self.lbl_sol_status.grid(row=1, column=0, columnspan=5, **pad)

        # Calibration
        grp_cal = ttk.LabelFrame(parent, text="Marker Calibration"); grp_cal.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_cal, text="Markers (HxV):", font=l_font).grid(row=0, column=0, **pad)
        h_frame = ttk.Frame(grp_cal)
        h_frame.grid(row=0, column=1, columnspan=3, **pad)
        ttk.Label(grp_cal, text="Markers (HxV):", font=l_font).grid(row=0, column=0, **pad)
        h_frame = ttk.Frame(grp_cal)
        h_frame.grid(row=0, column=1, columnspan=3, **pad)
        ttk.Spinbox(h_frame, textvariable=self.sol_marker_k_var, from_=2, to=20, width=5).pack(side="left")
        ttk.Label(h_frame, text="x").pack(side="left")
        ttk.Spinbox(h_frame, textvariable=self.sol_marker_n_var, from_=2, to=20, width=5).pack(side="left")
        
        ttk.Label(grp_cal, text="Pattern Size (px):", font=l_font).grid(row=1, column=0, **pad)
        ttk.Spinbox(grp_cal, textvariable=self.sol_marker_size_var, from_=20, to=400, width=8).grid(row=1, column=1, **pad)
        
        ttk.Label(grp_cal, text="Dict:", font=l_font).grid(row=1, column=2, **pad)
        # Assuming sol_tracker has dict mapping, hardcoding common ones for now
        dicts = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250"]
        ttk.Combobox(grp_cal, textvariable=self.sol_aruco_dict_var, values=dicts, state="readonly", width=15).grid(row=1, column=3, **pad)

        # Screen Width removed from here, shared with General tab
        
        # Gaze Method
        grp_gaze = ttk.LabelFrame(parent, text="Gaze Mapping Method"); grp_gaze.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_gaze, text="Method:", font=l_font).grid(row=0, column=0, **pad)
        gaze_methods = ["3D", "2D"]
        self.cmb_gaze_method = ttk.Combobox(grp_gaze, textvariable=self.sol_gaze_method_var, values=gaze_methods, state="readonly", width=20)
        self.cmb_gaze_method.grid(row=0, column=1, **pad)
        ttk.Label(grp_gaze, text="3D: Ray-plane intersection (uses gaze_3d)", font=("Arial", 9), foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(grp_gaze, text="2D: Homography mapping (uses gaze_2d)", font=("Arial", 9), foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        self.btn_clear_homography = ttk.Button(grp_gaze, text="Clear Homography Cache", command=self._clear_homography_cache)
        self.btn_clear_homography.grid(row=0, column=2, **pad)

        # Smoothing
        grp_sm = ttk.LabelFrame(parent, text="Smoothing"); grp_sm.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_sm, text="Pose Smooth Factor:", font=l_font).grid(row=0, column=0, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_pose_smooth_var, from_=0.01, to=1.0, increment=0.01, width=8).grid(row=0, column=1, **pad)
        ttk.Label(grp_sm, text="Gaze Smooth Factor:", font=l_font).grid(row=0, column=2, **pad)
        ttk.Spinbox(grp_sm, textvariable=self.sol_gaze_smooth_var, from_=0.01, to=1.0, increment=0.01, width=8).grid(row=0, column=3, **pad)

        # Preview Gaze Mapping
        grp_preview = ttk.LabelFrame(parent, text="Preview Gaze Mapping"); grp_preview.pack(fill="x", padx=10, pady=5)
        ttk.Label(grp_preview, text="Test Sol gaze projection on screen with ArUco markers.", font=("Arial", 10)).grid(row=0, column=0, columnspan=4, sticky="w", **pad)

        # Screen selection for preview
        ttk.Label(grp_preview, text="Screen:", font=l_font).grid(row=1, column=0, sticky="w", **pad)
        self.sol_preview_screen_var = tk.StringVar(value="0")
        # Get monitor info for dropdown
        preview_monitors = get_monitor_info_windows()
        preview_screen_options = []
        for mon in preview_monitors:
            label = f"{mon['index']}: {mon['name']} ({mon['width']}x{mon['height']})"
            preview_screen_options.append(label)
        if not preview_screen_options:
            preview_screen_options = ["0: Primary Display"]
        self.cmb_preview_screen = ttk.Combobox(grp_preview, textvariable=self.sol_preview_screen_var, values=preview_screen_options, state="readonly", width=35)
        self.cmb_preview_screen.grid(row=1, column=1, columnspan=2, sticky="w", **pad)
        if preview_screen_options:
            self.cmb_preview_screen.current(0)
        self.preview_monitor_info = preview_monitors  # Store for use in preview

        self.btn_preview_sol_gaze = ttk.Button(grp_preview, text="Preview Gaze", command=self.preview_sol_gaze, state="disabled")
        self.btn_preview_sol_gaze.grid(row=2, column=0, **pad)

        self.lbl_preview_note = ttk.Label(grp_preview, text="(Connect Sol glasses first)", foreground="gray")
        self.lbl_preview_note.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(grp_preview, text="Press Q or ESC to exit preview", font=("Arial", 9), foreground="gray").grid(row=3, column=0, columnspan=4, sticky="w", **pad)

    def build_sol_offset_tab(self, parent, l_font, e_font):
        """Build the Sol Offset Calibration tab (simplified, auto-detects 2D/3D method)."""
        pad = self.ui_pad  # Use dynamic padding

        # Info label showing current gaze method
        method_frame = ttk.Frame(parent)
        method_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(method_frame, text="Current Gaze Method:", font=l_font).pack(side="left")
        self.lbl_current_gaze_method = ttk.Label(method_frame, text="3D", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_current_gaze_method.pack(side="left", padx=10)
        ttk.Label(method_frame, text="(Set in Sol Settings tab)", font=("Arial", 9), foreground="gray").pack(side="left")

        # Target Settings
        grp_target = ttk.LabelFrame(parent, text="Calibration Settings")
        grp_target.pack(fill="x", padx=10, pady=5)

        ttk.Label(grp_target, text="Target Image:", font=l_font).grid(row=0, column=0, sticky="w", **pad)
        img_frame = ttk.Frame(grp_target)
        img_frame.grid(row=0, column=1, columnspan=2, sticky="w", **pad)
        ttk.Entry(img_frame, textvariable=self.sol_offset_target_img_var, font=e_font, width=40).pack(side="left")

        def _browse_target_img():
            p = filedialog.askopenfilename(parent=self, filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*.*")])
            if p:
                self.sol_offset_target_img_var.set(p)
                print(f"[Offset Cal] Selected target image: {p}")
        ttk.Button(img_frame, text="Browse...", command=_browse_target_img).pack(side="left", padx=5)

        ttk.Label(grp_target, text="Target Size (px):", font=l_font).grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_target, textvariable=self.sol_offset_target_size_var, from_=50, to=500, increment=10, width=10).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(grp_target, text="Calibration Points:", font=l_font).grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(grp_target, textvariable=self.sol_offset_num_points_var, from_=1, to=5, increment=1, width=10).grid(row=2, column=1, sticky="w", **pad)

        # Display Settings
        grp_display = ttk.LabelFrame(parent, text="Display Settings")
        grp_display.pack(fill="x", padx=10, pady=5)

        self.monitor_info_list = get_monitor_info_windows()
        screen_options = []
        for mon in self.monitor_info_list:
            label = f"{mon['index']}: {mon['name']} ({mon['width']}x{mon['height']})"
            screen_options.append(label)
        if len(screen_options) < 2:
            screen_options.append("1: Secondary Display (1920x1080)")

        ttk.Label(grp_display, text="User Screen:", font=l_font).grid(row=0, column=0, sticky="w", **pad)
        self.cmb_user_screen = ttk.Combobox(grp_display, textvariable=self.sol_offset_user_screen_var, values=screen_options, state="readonly", width=40)
        self.cmb_user_screen.grid(row=0, column=1, sticky="w", **pad)
        if screen_options:
            self.cmb_user_screen.current(0)

        ttk.Label(grp_display, text="Tester Screen:", font=l_font).grid(row=1, column=0, sticky="w", **pad)
        self.cmb_tester_screen = ttk.Combobox(grp_display, textvariable=self.sol_offset_tester_screen_var, values=screen_options, state="readonly", width=40)
        self.cmb_tester_screen.grid(row=1, column=1, sticky="w", **pad)
        if len(screen_options) > 1:
            self.cmb_tester_screen.current(1)
        elif screen_options:
            self.cmb_tester_screen.current(0)

        # Current Offset Status (shows both 3D and 2D)
        grp_status = ttk.LabelFrame(parent, text="Current Offset Status")
        grp_status.pack(fill="x", padx=10, pady=5)

        # 3D offset status
        ttk.Label(grp_status, text="3D Offset:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", **pad)
        self.lbl_sol_offset_pitch = ttk.Label(grp_status, text="Pitch: --", font=l_font)
        self.lbl_sol_offset_pitch.grid(row=0, column=1, sticky="w", **pad)
        self.lbl_sol_offset_yaw = ttk.Label(grp_status, text="Yaw: --", font=l_font)
        self.lbl_sol_offset_yaw.grid(row=0, column=2, sticky="w", **pad)

        # 2D offset status
        ttk.Label(grp_status, text="2D Offset:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", **pad)
        self.lbl_sol_2d_offset_status = ttk.Label(grp_status, text="Not calibrated", font=l_font)
        self.lbl_sol_2d_offset_status.grid(row=1, column=1, sticky="w", **pad)
        self.lbl_sol_2d_offset_points = ttk.Label(grp_status, text="", font=("Arial", 10))
        self.lbl_sol_2d_offset_points.grid(row=1, column=2, sticky="w", **pad)

        self.lbl_sol_offset_timestamp = ttk.Label(grp_status, text="Last Calibrated: Never", font=("Arial", 10))
        self.lbl_sol_offset_timestamp.grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=10, pady=10)

        self.btn_start_sol_offset_cal = ttk.Button(btn_frame, text="Start Calibration",
                                                    command=self.start_sol_offset_calibration_auto, state="disabled")
        self.btn_start_sol_offset_cal.pack(side="left", padx=5)
        # Keep reference for 2D button (same button, auto-detects method)
        self.btn_start_sol_2d_offset_cal = self.btn_start_sol_offset_cal

        self.btn_clear_sol_offset = ttk.Button(btn_frame, text="Clear Offset", command=self.clear_sol_offset_auto)
        self.btn_clear_sol_offset.pack(side="left", padx=5)

        self.lbl_sol_offset_connect_note = ttk.Label(btn_frame, text="(Connect Sol glasses first)", foreground="gray")
        self.lbl_sol_offset_connect_note.pack(side="left", padx=10)

        # Calibration options
        opt_frame = ttk.Frame(parent)
        opt_frame.pack(fill="x", padx=10, pady=5)
        ttk.Checkbutton(opt_frame, text="Show gaze during calibration (green circle on user screen)",
                       variable=self.sol_cal_show_gaze_var).pack(side="left")

        # Instructions
        grp_instr = ttk.LabelFrame(parent, text="Instructions")
        grp_instr.pack(fill="x", padx=10, pady=5)

        instr_text = """1. Connect Sol glasses in 'Sol Settings' tab and select Gaze Method (2D/3D).
2. Select a target image above.
3. Click 'Start Calibration' - target appears on user screen.
4. Have user look at each target. Press SPACE when gaze is stable.
5. Repeat for all calibration points.

• 3D Method: Corrects angular offset (pitch/yaw) in camera frame.
• 2D Method: Uses SVR model for position-dependent correction.

Controls: SPACE = Record point, Q = Cancel"""

        ttk.Label(grp_instr, text=instr_text, font=("Arial", 10), justify="left").pack(anchor="w", **pad)

        # Bind gaze method variable to update display
        self.sol_gaze_method_var.trace_add("write", lambda *args: self._update_gaze_method_display())

        # Load current offset on tab creation
        self.after(100, self.update_sol_offset_display)
        self.after(100, self.update_sol_2d_offset_display)
        self.after(100, self._update_gaze_method_display)

    def _update_gaze_method_display(self):
        """Update the gaze method display label."""
        method = self.sol_gaze_method_var.get()
        if hasattr(self, 'lbl_current_gaze_method'):
            self.lbl_current_gaze_method.config(text=method, foreground="blue" if method == "3D" else "green")

    def start_sol_offset_calibration_auto(self):
        """Start offset calibration based on current gaze method (2D or 3D)."""
        method = self.sol_gaze_method_var.get()
        print(f"[Offset Cal] Starting calibration for {method} method")

        if method == "2D":
            if SOL_2D_OFFSET_AVAILABLE:
                self.start_sol_2d_offset_calibration()
            else:
                messagebox.showerror("Error", "2D offset calibration module not available.\nPlease install scikit-learn: pip install scikit-learn")
        else:
            self.start_sol_offset_calibration()

    def clear_sol_offset_auto(self):
        """Clear offset based on current gaze method."""
        method = self.sol_gaze_method_var.get()
        if method == "2D":
            self.clear_sol_2d_offset()
        else:
            self.clear_sol_offset()

    def update_sol_offset_display(self):
        """Update the Sol offset status display (3D offset)."""
        if not SOL_OFFSET_AVAILABLE:
            self.lbl_sol_offset_pitch.config(text="Pitch: N/A")
            self.lbl_sol_offset_yaw.config(text="Yaw: N/A")
            return

        username = self.user_var.get().strip() or "anonymous"
        offset_data = load_sol_offset(username, Path(self.calib_dir_var.get()))

        if offset_data:
            pitch_deg = offset_data.get('pitch_offset_deg', 0)
            yaw_deg = offset_data.get('yaw_offset_deg', 0)
            timestamp = offset_data.get('calibration_timestamp', 'Unknown')

            self.lbl_sol_offset_pitch.config(text=f"Pitch: {pitch_deg:.2f}°")
            self.lbl_sol_offset_yaw.config(text=f"Yaw: {yaw_deg:.2f}°")
            self.lbl_sol_offset_timestamp.config(text=f"Last Calibrated: {timestamp}")
        else:
            self.lbl_sol_offset_pitch.config(text="Pitch: --")
            self.lbl_sol_offset_yaw.config(text="Yaw: --")

    def update_sol_2d_offset_display(self):
        """Update the 2D Sol offset status display."""
        if not SOL_2D_OFFSET_AVAILABLE:
            if hasattr(self, 'lbl_sol_2d_offset_status'):
                self.lbl_sol_2d_offset_status.config(text="N/A (install scikit-learn)")
            return

        username = self.user_var.get().strip() or "anonymous"
        offset_data = load_sol_2d_offset(username, Path(self.calib_dir_var.get()))

        if offset_data and offset_data.get('model'):
            num_points = offset_data.get('num_calibration_points', 0)
            timestamp = offset_data.get('calibration_timestamp', 'Unknown')

            self.lbl_sol_2d_offset_status.config(text="Calibrated (SVR)")
            self.lbl_sol_2d_offset_points.config(text=f"({num_points} pts)")
            self.lbl_sol_offset_timestamp.config(text=f"Last Calibrated: {timestamp}")
        else:
            self.lbl_sol_2d_offset_status.config(text="Not calibrated")
            self.lbl_sol_2d_offset_points.config(text="")

        # Update button state
        self._update_offset_button_state()

    def _update_offset_button_state(self):
        """Update offset calibration button state based on connection."""
        if not hasattr(self, 'btn_start_sol_offset_cal'):
            return

        # Button is enabled when Sol is connected
        if self.is_sol_connected:
            self.btn_start_sol_offset_cal.config(state="normal")
            self.lbl_sol_offset_connect_note.config(text="")
        else:
            self.btn_start_sol_offset_cal.config(state="disabled")
            self.lbl_sol_offset_connect_note.config(text="(Connect Sol glasses first)")

    def clear_sol_2d_offset(self):
        """Clear the 2D Sol offset calibration."""
        if not SOL_2D_OFFSET_AVAILABLE:
            return

        username = self.user_var.get().strip() or "anonymous"
        clear_sol_2d_offset(username, Path(self.calib_dir_var.get()))
        self.update_sol_2d_offset_display()
        messagebox.showinfo("2D Offset Cleared", "2D gaze offset calibration has been cleared.")

    def start_sol_2d_offset_calibration(self):
        """Start the 2D Sol offset calibration process."""
        if not SOL_2D_OFFSET_AVAILABLE:
            messagebox.showerror("Error", "Sol 2D offset calibration module not available.")
            return

        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected.")
            return

        target_img_path = self.sol_offset_target_img_var.get().strip()
        print(f"[2D Offset Cal] Target image path: {target_img_path}")
        print(f"[2D Offset Cal] Path exists: {os.path.exists(target_img_path)}")

        if not target_img_path or not os.path.exists(target_img_path):
            messagebox.showerror("Error", f"Please select a valid target image.\nPath: {target_img_path}")
            return

        # Try to load the image to verify it's readable
        test_img = cv2.imread(target_img_path)
        if test_img is None:
            # Try with Unicode path handling
            test_img = cv2.imdecode(np.fromfile(target_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(f"[2D Offset Cal] cv2.imread result: {'OK' if test_img is not None else 'FAILED'}")
        if test_img is None:
            messagebox.showerror("Error", f"Failed to load target image with OpenCV.\nPath: {target_img_path}\nCheck if file is a valid image.")
            return

        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        # Get monitor info for selected screen
        user_screen_str = self.sol_offset_user_screen_var.get()
        try:
            user_screen_idx = int(user_screen_str.split(':')[0].strip())
        except:
            user_screen_idx = 0

        monitors = getattr(self, 'monitor_info_list', None) or get_monitor_info_windows()
        if user_screen_idx >= len(monitors):
            user_screen_idx = 0

        user_screen = monitors[user_screen_idx]
        screen_w = user_screen['width']
        screen_h = user_screen['height']
        screen_x = user_screen.get('x', 0)
        screen_y = user_screen.get('y', 0)

        # Prepare ArUco assets for pose detection
        aruco_dict_key = self.sol_aruco_dict_var.get()
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80)
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30

        # Create calibrator
        target_size = self.safe_get_int(self.sol_offset_target_size_var, 100)
        num_points = self.safe_get_int(self.sol_offset_num_points_var, 5)

        # Use pixel-based offset mode (simpler and proven to work)
        calibrator = Sol2DOffsetCalibrator(
            target_image_path=target_img_path,
            screen_width=screen_w,
            screen_height=screen_h,
            num_points=num_points,
            target_display_size=target_size,
            offset_mode=OFFSET_MODE_PIXEL,
            camera_matrix=None
        )

        # Compute safe calibration positions that avoid ArUco markers (20px gap)
        safe_positions = compute_safe_calibration_positions(
            screen_w, screen_h, aruco_markers_px,
            marker_container_size, target_size, gap=20
        )
        calibrator.set_safe_positions(safe_positions)

        if calibrator.target_image is None:
            messagebox.showerror("Error", "Failed to load target image.")
            return

        # Set flag to prevent queue flushing
        self.in_sol_offset_calibration = True

        # Resume scene stream for calibration (may have been paused when returning to settings)
        if self.active_sol_connector:
            self.active_sol_connector.resume_scene_stream()

        # Hide settings window
        self.withdraw()

        # Run the 2D calibration
        self._run_sol_2d_offset_calibration(
            calibrator, screen_w, screen_h, screen_x, screen_y,
            aruco_markers_px, aruco_imgs, marker_container_size, adict
        )

    def _run_sol_2d_offset_calibration(self, calibrator, screen_w, screen_h, screen_x, screen_y,
                                        aruco_markers_px, aruco_imgs, marker_container_size, adict):
        """Run the 2D offset calibration process with homography-based target positioning."""
        import os as os_module

        # Create a ScreenProjector3D for homography computation
        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict,
                                          smoothing_factor=self.safe_get_float(self.sol_pose_smooth_var, 0.1))

        # Physical screen width in meters
        screen_width_m = self.safe_get_float(self.scr_width_cm_var, 53.0) / 100.0
        marker_physical_size_m = self.safe_get_int(self.sol_marker_size_var, 80) / screen_w * screen_width_m

        # Start background ArUco detection to build the homography
        sol_projector.start_background_detection(
            marker_physical_size_m,
            aruco_markers_px,
            marker_container_size,
            screen_w, screen_h, screen_width_m
        )

        # Get tester screen info for monitoring window
        tester_screen_str = self.sol_offset_tester_screen_var.get()
        try:
            tester_screen_idx = int(tester_screen_str.split(':')[0].strip())
        except:
            tester_screen_idx = 1  # Default to second screen

        monitors = getattr(self, 'monitor_info_list', None) or get_monitor_info_windows()
        if tester_screen_idx >= len(monitors):
            tester_screen_idx = 0

        tester_screen = monitors[tester_screen_idx]
        tester_x = tester_screen.get('x', 0)
        tester_y = tester_screen.get('y', 0)

        # Setup user screen window - use NOFRAME instead of FULLSCREEN for multi-monitor support
        os_module.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
        pygame.init()
        win = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
        pygame.display.set_caption("Sol 2D Offset Calibration")

        # Force window to front
        import ctypes
        try:
            hwnd = pygame.display.get_wm_info()['window']
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
        except Exception as e:
            print(f"[2D Cal] Could not bring window to front: {e}")

        # Setup tester monitoring window (OpenCV)
        tester_win_name = "Tester View - Sol 2D Calibration"
        cv2.namedWindow(tester_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tester_win_name, 800, 600)
        cv2.moveWindow(tester_win_name, tester_x + 50, tester_y + 50)

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        running = True
        latest_gaze = None
        latest_frame = None
        homography_ready = False
        homography_good = False  # True only when homography passes quality check
        H_screen_to_image = None
        target_pos_camera = None

        # Multi-sample collection (like GazeFollower's 45 frames per point)
        SAMPLES_TO_COLLECT = 30  # Collect 30 samples (~1 second at 30fps)
        collecting_samples = False
        collected_gaze_samples = []
        collection_start_time = 0

        # Minimum frames to wait for stable homography
        MIN_FRAMES_FOR_HOMOGRAPHY = 100
        frame_count = 0

        # Load target image for pygame display
        target_surface = None
        try:
            if calibrator.target_image_display is not None:
                target_cv = calibrator.target_image_display.copy()
                target_cv = cv2.cvtColor(target_cv, cv2.COLOR_BGR2RGB)
                target_surface = pygame.image.frombuffer(target_cv.tobytes(), target_cv.shape[1::-1], "RGB")
                print(f"[2D Cal] Target surface created: {calibrator.target_display_size}x{calibrator.target_display_size}")
        except Exception as e:
            print(f"[2D Cal] Failed to create target surface: {e}")

        try:
            while running and not calibrator.calibration_complete:
                # Handle events
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_q:
                            running = False
                        elif ev.key == pygame.K_SPACE:
                            # Start collecting samples when SPACE is pressed
                            # ONLY if homography passes quality check
                            if homography_good and not collecting_samples and H_screen_to_image is not None:
                                collecting_samples = True
                                collected_gaze_samples = []
                                collection_start_time = time.time()
                                print(f"[2D Cal] Starting sample collection for point {calibrator.current_point_index + 1}...")
                            elif not homography_good:
                                print(f"[2D Cal] Cannot record - homography not stable yet. Wait for quality check to pass.")

                # Get latest gaze data (drain queue but only use the last one)
                # Limit to 20 items per frame to prevent long delays when queue builds up
                for _ in range(20):
                    try:
                        gaze = self.sol_gaze_queue.get_nowait()
                        latest_gaze = gaze
                    except queue.Empty:
                        break

                # If collecting samples, add ONE sample per render frame (30fps)
                # This ensures collection takes ~1 second regardless of gaze queue rate
                if collecting_samples and latest_gaze:
                    g2d = latest_gaze.combined.gaze_2d
                    collected_gaze_samples.append((g2d.x, g2d.y))

                # Check if we've collected enough samples
                if collecting_samples and len(collected_gaze_samples) >= SAMPLES_TO_COLLECT:
                    try:
                        # Compute average gaze_2d from all samples
                        avg_x = np.mean([s[0] for s in collected_gaze_samples])
                        avg_y = np.mean([s[1] for s in collected_gaze_samples])
                        std_x = np.std([s[0] for s in collected_gaze_samples])
                        std_y = np.std([s[1] for s in collected_gaze_samples])
                        avg_gaze_2d = (avg_x, avg_y)

                        print(f"[2D Cal] Collected {len(collected_gaze_samples)} samples: "
                              f"avg=({avg_x:.1f}, {avg_y:.1f}), std=({std_x:.1f}, {std_y:.1f})")

                        # Record the calibration point with averaged gaze
                        calibrator.record_calibration_point_with_homography(avg_gaze_2d, H_screen_to_image)
                    except Exception as e:
                        print(f"[2D Cal] Error recording calibration point: {e}")
                        traceback.print_exc()

                    collecting_samples = False
                    collected_gaze_samples = []

                # Get latest scene frame and submit for ArUco detection
                # Limit to 10 items per frame to prevent long delays when queue builds up
                for _ in range(10):
                    try:
                        frame = self.sol_scene_queue.get_nowait()
                        if hasattr(frame, 'img') and frame.img is not None:
                            latest_frame = frame.img
                        elif hasattr(frame, 'get_buffer'):
                            try:
                                w, h = 1328, 1200
                                buf = frame.get_buffer()
                                arr = np.frombuffer(buf, dtype=np.uint8)
                                latest_frame = arr.reshape((h, w, 3))
                            except:
                                pass
                        elif isinstance(frame, np.ndarray):
                            latest_frame = frame
                    except queue.Empty:
                        break

                # Submit frame for ArUco detection (builds homography)
                if latest_frame is not None:
                    sol_projector.submit_frame_for_pose(latest_frame.copy())

                # Count frames
                frame_count += 1

                # Check homography status and quality (strict mode for calibration)
                homography_ready = sol_projector.is_homography_valid(strict=True)
                homography_good = False
                if homography_ready:
                    H_screen_to_image = sol_projector.get_screen_to_image_homography()

                    # Validate homography quality by checking if screen center maps reasonably
                    if H_screen_to_image is not None and frame_count >= MIN_FRAMES_FOR_HOMOGRAPHY:
                        # Test: screen center should map to somewhere within camera image
                        test_screen = (screen_w // 2, screen_h // 2)
                        test_cam = sol_projector.project_screen_to_image(test_screen)
                        if test_cam is not None:
                            cam_x, cam_y = test_cam
                            # Camera is 1328x1200 - check if result is reasonable
                            if 100 < cam_x < 1200 and 100 < cam_y < 1100:
                                homography_good = True
                            else:
                                # Bad homography - screen center maps outside camera
                                if frame_count % 100 == 0:
                                    print(f"[2D Cal] Homography quality check FAILED: screen({test_screen[0]}, {test_screen[1]}) -> cam({cam_x:.1f}, {cam_y:.1f})")

                    # Compute target position in camera image using homography
                    target_screen_pos = calibrator.get_current_target_screen_position()
                    if target_screen_pos and H_screen_to_image is not None and homography_good:
                        target_pos_camera = calibrator.compute_target_camera_position(H_screen_to_image)
                    else:
                        target_pos_camera = None

                # === TESTER MONITORING WINDOW ===
                if latest_frame is not None:
                    tester_view = latest_frame.copy()

                    # Draw current gaze point (blue circle)
                    if latest_gaze:
                        g2d = latest_gaze.combined.gaze_2d
                        gaze_x, gaze_y = int(g2d.x), int(g2d.y)
                        cv2.circle(tester_view, (gaze_x, gaze_y), 15, (255, 0, 0), 3)
                        cv2.putText(tester_view, "Gaze", (gaze_x + 20, gaze_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Draw projected target position (green circle)
                    if target_pos_camera:
                        tx, ty = int(target_pos_camera[0]), int(target_pos_camera[1])
                        cv2.circle(tester_view, (tx, ty), 20, (0, 255, 0), 3)
                        cv2.drawMarker(tester_view, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                        cv2.putText(tester_view, "Target", (tx + 25, ty - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw offset line between gaze and target
                    if latest_gaze and target_pos_camera:
                        gaze_x, gaze_y = int(latest_gaze.combined.gaze_2d.x), int(latest_gaze.combined.gaze_2d.y)
                        tx, ty = int(target_pos_camera[0]), int(target_pos_camera[1])
                        cv2.line(tester_view, (gaze_x, gaze_y), (tx, ty), (0, 255, 255), 2)
                        offset_dist = np.sqrt((gaze_x - tx)**2 + (gaze_y - ty)**2)
                        mid_x, mid_y = (gaze_x + tx) // 2, (gaze_y + ty) // 2
                        cv2.putText(tester_view, f"Offset: {offset_dist:.1f}px", (mid_x, mid_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Status bar at top
                    pos_name = calibrator.get_current_position_name() or "Complete"
                    point_num = calibrator.current_point_index + 1
                    total_points = len(calibrator.positions)

                    status_text = f"Position: {pos_name} ({point_num}/{total_points})"

                    if collecting_samples:
                        homog_status = f"COLLECTING... {len(collected_gaze_samples)}/{SAMPLES_TO_COLLECT}"
                        homog_color = (0, 255, 255)  # Yellow
                    elif homography_good:
                        homog_status = "READY - Press SPACE"
                        homog_color = (0, 255, 0)
                    elif homography_ready:
                        homog_status = f"Stabilizing... ({frame_count}/{MIN_FRAMES_FOR_HOMOGRAPHY})"
                        homog_color = (0, 200, 255)  # Orange
                    else:
                        homog_status = "Waiting for ArUco..."
                        homog_color = (0, 100, 255)

                    cv2.rectangle(tester_view, (0, 0), (tester_view.shape[1], 60), (40, 40, 40), -1)
                    cv2.putText(tester_view, status_text, (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(tester_view, homog_status, (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, homog_color, 2)
                    cv2.putText(tester_view, "SPACE=Record  Q=Quit", (tester_view.shape[1] - 250, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                    cv2.imshow(tester_win_name, tester_view)
                    cv2.waitKey(1)

                # === USER SCREEN (Pygame) ===
                win.fill((50, 50, 50))

                # Draw ArUco markers
                for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape) == 2:
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                        elif cv_img.shape[2] == 4:
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                        else:
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(py_img, (pos[0], pos[1]))

                # Draw target at current calibration position
                target_screen_pos = calibrator.get_current_target_screen_position()
                if target_screen_pos and target_surface:
                    tx = target_screen_pos[0] - calibrator.target_display_size // 2
                    ty = target_screen_pos[1] - calibrator.target_display_size // 2
                    win.blit(target_surface, (tx, ty))

                # Draw gaze visualization if enabled
                if self.sol_cal_show_gaze_var.get() and latest_gaze and sol_projector.is_homography_valid():
                    try:
                        g2d = latest_gaze.combined.gaze_2d
                        gaze_2d_pt = (g2d.x, g2d.y)
                        # Project gaze to screen using homography (without offset correction during calibration)
                        screen_pt = sol_projector.project_gaze_2d_to_screen(gaze_2d_pt, apply_smoothing=False, apply_offset=False)
                        if screen_pt:
                            gx, gy = int(screen_pt[0]), int(screen_pt[1])
                            # Only draw if within screen bounds
                            if 0 <= gx < screen_w and 0 <= gy < screen_h:
                                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 20, 4)
                                pygame.draw.circle(win, (0, 200, 0), (gx, gy), 8)
                    except Exception as e:
                        pass  # Don't crash if gaze projection fails

                # Status bar
                pos_name = calibrator.get_current_position_name() or "Complete"
                point_num = calibrator.current_point_index + 1
                total_points = len(calibrator.positions)

                if collecting_samples:
                    status_color = (255, 255, 100)  # Yellow
                    detect_text = f"COLLECTING... {len(collected_gaze_samples)}/{SAMPLES_TO_COLLECT} - Keep looking at target!"
                elif homography_good:
                    status_color = (100, 255, 100)
                    detect_text = "READY - Press SPACE to start recording"
                elif homography_ready:
                    status_color = (255, 200, 100)
                    detect_text = f"Stabilizing homography... ({frame_count}/{MIN_FRAMES_FOR_HOMOGRAPHY})"
                else:
                    status_color = (255, 100, 100)
                    detect_text = "Waiting for ArUco markers..."

                pygame.draw.rect(win, (30, 30, 30), (0, screen_h - 80, screen_w, 80))

                text1 = font.render(f"Position: {pos_name} ({point_num}/{total_points})", True, (255, 255, 255))
                text2 = font.render(detect_text, True, status_color)
                text3 = small_font.render("Press Q to cancel", True, (150, 150, 150))

                win.blit(text1, (20, screen_h - 75))
                win.blit(text2, (20, screen_h - 45))
                win.blit(text3, (screen_w - 150, screen_h - 30))

                pygame.display.flip()
                clock.tick(30)

            # Calibration complete or cancelled
            if calibrator.calibration_complete:
                # Train the model
                if calibrator.finish_calibration():
                    # Save the model
                    username = self.user_var.get().strip() or "anonymous"
                    save_sol_2d_offset(
                        username,
                        Path(self.calib_dir_var.get()),
                        calibrator.get_model(),
                        screen_w, screen_h,
                        calibrator.target_image_path
                    )
                    self.after(0, lambda: messagebox.showinfo("Calibration Complete",
                        f"2D offset calibration complete!\n{len(calibrator.model.calibration_points)} points recorded."))
                else:
                    self.after(0, lambda: messagebox.showerror("Calibration Failed",
                        "Failed to train the SVR model. Please try again."))
            else:
                self.after(0, lambda: messagebox.showinfo("Calibration Cancelled", "2D offset calibration was cancelled."))

        except Exception as cal_error:
            print(f"[2D Cal] FATAL ERROR in calibration: {cal_error}")
            traceback.print_exc()
            self.after(0, lambda e=str(cal_error): messagebox.showerror("Calibration Error", f"Calibration crashed: {e}"))

        finally:
            # Save homography for next session
            try:
                H = sol_projector.get_homography()
                if H is not None:
                    self.sol_cached_homography = H
                    print(f"[2D Cal] Cached homography for next session")
            except Exception as e:
                print(f"[2D Cal] Error caching homography: {e}")
            # Cleanup (with error handling to prevent crashes)
            try:
                sol_projector.stop_background_detection()
            except Exception as e:
                print(f"[2D Cal] Error stopping ArUco: {e}")
            try:
                cv2.destroyWindow(tester_win_name)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[2D Cal] Error destroying windows: {e}")
            try:
                pygame.quit()
            except Exception as e:
                print(f"[2D Cal] Error quitting pygame: {e}")
            self.in_sol_offset_calibration = False
            # Pause scene stream when returning to settings to avoid native crash
            if self.active_sol_connector:
                self.active_sol_connector.pause_scene_stream()
            self.deiconify()
            self.after(100, self.update_sol_2d_offset_display)

    def start_sol_offset_calibration(self):
        """Start the Sol offset calibration process."""
        if not SOL_OFFSET_AVAILABLE:
            messagebox.showerror("Error", "Sol offset calibration module not available.")
            return

        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected. Please connect in 'Sol Settings' tab first.")
            return

        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        # Get current screen dimensions (will be overridden by pygame)
        info = pygame.display.Info() if pygame.get_init() else None

        # Prepare ArUco assets
        aruco_dict_key = self.sol_aruco_dict_var.get()
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        # Get screen dimensions (use temp pygame init)
        pygame.init()
        screen_info = pygame.display.Info()
        screen_w, screen_h = screen_info.current_w, screen_info.current_h
        pygame.quit()

        # Create ArUco markers
        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80)
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30

        # Physical screen width in meters
        screen_width_m = self.safe_get_float(self.scr_width_cm_var, 53.0) / 100.0

        # Create projector
        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict,
                                         smoothing_factor=self.safe_get_float(self.sol_pose_smooth_var, 0.1))

        # Restore cached homography from previous preview session
        if self.sol_cached_homography is not None:
            sol_projector.set_homography(self.sol_cached_homography)
            print(f"[Offset Cal] Restored cached homography from preview")

        # Start background detection
        sol_projector.start_background_detection(
            sol_cfg_for_assets['marker_pattern_size'] / screen_w * screen_width_m,
            aruco_markers_px,
            marker_container_size,
            screen_w, screen_h, screen_width_m
        )

        # Create calibrator
        calibrator = SolOffsetCalibrator(
            sol_projector=sol_projector,
            sol_gaze_queue=self.sol_gaze_queue,
            sol_scene_queue=self.sol_scene_queue,
            screen_width_m=screen_width_m,
            aruco_markers_px=aruco_markers_px,
            aruco_imgs=aruco_imgs,
            marker_container_size=marker_container_size
        )

        # Set flag to prevent flush_sol_queues from stealing frames
        self.in_sol_offset_calibration = True

        # Resume scene stream for calibration (may have been paused when returning to settings)
        if self.active_sol_connector:
            self.active_sol_connector.resume_scene_stream()

        # Hide settings window during calibration
        self.withdraw()

        # Parse screen indices from dropdown selections (format: "0: Model Name (1920x1080)")
        def parse_screen_idx(combo_value):
            try:
                return int(combo_value.split(':')[0].strip())
            except:
                return 0

        user_screen_idx = parse_screen_idx(self.sol_offset_user_screen_var.get())
        tester_screen_idx = parse_screen_idx(self.sol_offset_tester_screen_var.get())

        try:
            # Run calibration
            result = calibrator.run_calibration(
                num_points=self.safe_get_int(self.sol_offset_num_points_var, 5),
                target_image_path=self.sol_offset_target_img_var.get().strip() or None,
                target_size_px=self.safe_get_int(self.sol_offset_target_size_var, 100),
                user_screen_idx=user_screen_idx,
                tester_screen_idx=tester_screen_idx,
                monitor_info=self.monitor_info_list if hasattr(self, 'monitor_info_list') else None
            )

            if result:
                # Save result
                username = self.user_var.get().strip() or "anonymous"
                save_sol_offset(username, Path(self.calib_dir_var.get()), result)
                messagebox.showinfo("Success",
                                   f"Calibration complete!\n\n"
                                   f"Pitch offset: {result['pitch_offset_deg']:.2f} deg\n"
                                   f"Yaw offset: {result['yaw_offset_deg']:.2f} deg\n"
                                   f"Points used: {result['num_calibration_points']}")
            else:
                messagebox.showwarning("Cancelled", "Calibration was cancelled.")

        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed: {e}")
        finally:
            # Reset calibration flag
            self.in_sol_offset_calibration = False
            # Stop background detection
            sol_projector.stop_background_detection()
            # Pause scene stream when returning to settings to avoid native crash
            if self.active_sol_connector:
                self.active_sol_connector.pause_scene_stream()
            # Show settings window again
            self.deiconify()
            self.update_sol_offset_display()

    def clear_sol_offset(self):
        """Clear the Sol offset calibration file."""
        if not SOL_OFFSET_AVAILABLE:
            return

        username = self.user_var.get().strip() or "anonymous"
        if messagebox.askyesno("Confirm", f"Clear Sol offset calibration for user '{username}'?"):
            clear_sol_offset(username, Path(self.calib_dir_var.get()))
            self.update_sol_offset_display()
            messagebox.showinfo("Cleared", "Sol offset calibration cleared.")

    def preview_sol_gaze(self):
        """Preview Sol gaze mapping with ArUco markers on screen."""
        if not self.is_sol_connected:
            messagebox.showerror("Error", "Sol glasses not connected.")
            return

        if self.sol_cam_params is None:
            messagebox.showerror("Error", "Sol camera parameters not available.")
            return

        # Prepare ArUco assets
        aruco_dict_key = self.sol_aruco_dict_var.get()
        aruco_dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        }
        selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
        adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)

        # Get screen info from selection
        monitors = getattr(self, 'preview_monitor_info', None) or get_monitor_info_windows()
        if not monitors:
            monitors = [{'index': 0, 'width': 1920, 'height': 1080, 'x': 0, 'y': 0}]

        # Parse selected screen index from dropdown (format: "0: Model Name (1920x1080)")
        try:
            screen_idx = int(self.sol_preview_screen_var.get().split(':')[0].strip())
        except:
            screen_idx = 0
        screen_idx = min(screen_idx, len(monitors) - 1)
        screen = monitors[screen_idx]
        screen_w, screen_h = screen['width'], screen['height']
        screen_x, screen_y = screen.get('x', 0), screen.get('y', 0)
        print(f"[Sol Preview] Using screen {screen_idx}: {screen_w}x{screen_h} at ({screen_x}, {screen_y})")

        # Create ArUco markers
        sol_cfg_for_assets = {
            'marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'marker_pattern_size': self.safe_get_int(self.sol_marker_size_var, 80)
        }
        aruco_markers_px, aruco_imgs = create_calibration_assets(screen_w, screen_h, adict, sol_cfg_for_assets)
        marker_container_size = sol_cfg_for_assets['marker_pattern_size'] + 30

        # Physical screen width in meters
        screen_width_m = self.safe_get_float(self.scr_width_cm_var, 53.0) / 100.0

        # Load 2D offset model if available
        preview_2d_offset_model = None
        if SOL_2D_OFFSET_AVAILABLE:
            username = self.user_var.get().strip() or 'anonymous'
            calib_dir = Path(self.calib_dir_var.get())
            preview_2d_offset_data = load_sol_2d_offset(username, calib_dir)
            if preview_2d_offset_data and preview_2d_offset_data.get('model') and preview_2d_offset_data['model'].is_trained:
                preview_2d_offset_model = preview_2d_offset_data['model']
                print(f"[Sol Preview] Loaded 2D offset model for user '{username}'")

        # Create projector
        cam_matrix = self.sol_cam_params.get('cam_matrix')
        dist_coeffs = self.sol_cam_params.get('dist_coeffs')
        if cam_matrix is None:
            cam_matrix = np.array([[screen_w, 0, screen_w / 2], [0, screen_w, screen_h / 2], [0, 0, 1]], dtype=float)
            dist_coeffs = np.zeros(5)

        sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict,
                                         smoothing_factor=self.safe_get_float(self.sol_pose_smooth_var, 0.1))

        # Restore cached homography if available for faster startup
        if self.sol_cached_homography is not None:
            sol_projector.set_homography(self.sol_cached_homography)
            print(f"[Sol Preview] Restored cached homography")

        # Set 2D gaze smoothing factor and reset smoothing state
        sol_projector.set_gaze_2d_smoothing_factor(self.safe_get_float(self.sol_gaze_smooth_var, 0.15))
        sol_projector.reset_gaze_2d_smoothing()

        # Set 2D offset model if available
        if preview_2d_offset_model and preview_2d_offset_model.is_trained:
            sol_projector.set_gaze_2d_offset_model(preview_2d_offset_model)
            print(f"[Sol Preview] Applied 2D offset model")

        # Start background detection
        sol_projector.start_background_detection(
            sol_cfg_for_assets['marker_pattern_size'] / screen_w * screen_width_m,
            aruco_markers_px,
            marker_container_size,
            screen_w, screen_h, screen_width_m
        )

        # Set flag to prevent queue flushing
        self.in_sol_offset_calibration = True

        # Resume scene stream for preview (may have been paused when returning to settings)
        if self.active_sol_connector:
            self.active_sol_connector.resume_scene_stream()

        # Hide settings window
        self.withdraw()

        # Set window position and init pygame
        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
        pygame.init()
        win = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
        pygame.display.set_caption("Sol Gaze Preview")

        # Make window transparent using Windows API
        # Color key: this specific color will be transparent
        TRANSPARENT_COLOR = (1, 1, 1)  # Nearly black, used as transparent key
        try:
            import ctypes
            from ctypes import wintypes

            # Get the window handle
            hwnd = pygame.display.get_wm_info()['window']

            # Windows constants
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            LWA_COLORKEY = 0x00000001

            user32 = ctypes.windll.user32

            # Set extended window style to layered
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)

            # Set color key for transparency (RGB as 0x00BBGGRR)
            color_key = TRANSPARENT_COLOR[0] | (TRANSPARENT_COLOR[1] << 8) | (TRANSPARENT_COLOR[2] << 16)
            user32.SetLayeredWindowAttributes(hwnd, color_key, 0, LWA_COLORKEY)

            # Bring window to top
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_TOPMOST, SWP_NOMOVE | SWP_NOSIZE

            print("[Sol Preview] Transparent window enabled")
        except Exception as e:
            print(f"[Sol Preview] Could not enable transparency: {e}")
            TRANSPARENT_COLOR = (128, 128, 128)  # Fallback to grey

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        running = True
        latest_gaze = None
        current_gaze_pt = None
        frame_count = 0

        print("[Sol Preview] Entering main loop...")
        try:
            while running:
                # Check if pygame display is still valid
                try:
                    if not pygame.display.get_init():
                        print("[Sol Preview] pygame display lost, exiting")
                        break
                except:
                    print("[Sol Preview] pygame check failed, exiting")
                    break
                frame_count += 1
                if frame_count == 1:
                    print("[Sol Preview] First frame processing...")
                try:
                    # Handle events
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            running = False
                        elif ev.type == pygame.KEYDOWN:
                            if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                                running = False

                    # Get gaze data (limit to 20 items to prevent delays)
                    for _ in range(20):
                        try:
                            gaze = self.sol_gaze_queue.get_nowait()
                            latest_gaze = gaze
                        except queue.Empty:
                            break

                    # Get scene frames and submit for ArUco detection (limit to 10 items)
                    latest_frame_numpy = None
                    for _ in range(10):
                        try:
                            frame = self.sol_scene_queue.get_nowait()
                            if hasattr(frame, 'img') and frame.img is not None:
                                latest_frame_numpy = frame.img
                            elif hasattr(frame, 'get_buffer'):
                                try:
                                    w, h = 1328, 1200
                                    buf = frame.get_buffer()
                                    arr = np.frombuffer(buf, dtype=np.uint8)
                                    latest_frame_numpy = arr.reshape((h, w, 3))
                                except:
                                    pass
                            elif isinstance(frame, np.ndarray):
                                latest_frame_numpy = frame
                        except queue.Empty:
                            break

                    if latest_frame_numpy is not None:
                        sol_projector.submit_frame_for_pose(latest_frame_numpy)

                    # Process gaze based on selected method
                    gaze_method = self.sol_gaze_method_var.get()

                    if latest_gaze is not None:
                        try:
                            if gaze_method == "2D" and sol_projector.is_homography_valid():
                                # 2D Gaze Mapping: Use gaze_2d and homography
                                if hasattr(latest_gaze.combined, 'gaze_2d'):
                                    g2d = latest_gaze.combined.gaze_2d
                                    gaze_2d_pt = (g2d.x, g2d.y)
                                    screen_pt = sol_projector.project_gaze_2d_to_screen(gaze_2d_pt)
                                    if screen_pt:
                                        current_gaze_pt = screen_pt
                            elif gaze_method == "3D" and sol_projector.is_calibrated():
                                # 3D Gaze Mapping: Use gaze_3d and ray-plane intersection
                                left_o = latest_gaze.left_eye.gaze.origin
                                right_o = latest_gaze.right_eye.gaze.origin
                                gaze_origin_mm = np.array([
                                    (left_o.x + right_o.x) / 2.0,
                                    (left_o.y + right_o.y) / 2.0,
                                    (left_o.z + right_o.z) / 2.0
                                ])
                                g3d = latest_gaze.combined.gaze_3d
                                gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])
                                gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                                norm = np.linalg.norm(gaze_direction_vec)
                                if norm > 0:
                                    gaze_direction_unit = gaze_direction_vec / norm
                                    gaze_origin_m = gaze_origin_mm / 1000.0
                                    screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                    if screen_pt_m is not None:
                                        pix = sol_projector.physical_to_pixels(screen_pt_m, screen_w, screen_width_m)
                                        if pix:
                                            current_gaze_pt = (int(pix[0]), int(pix[1]))
                        except Exception as e:
                            print(f"[Sol Preview] Gaze processing error: {e}")
                            traceback.print_exc()

                    # Log progress every 100 frames
                    if frame_count % 100 == 0:
                        print(f"[Sol Preview] Frame {frame_count}, gaze_pt={current_gaze_pt}")

                    # Render - fill with transparent color key
                    win.fill(TRANSPARENT_COLOR)

                    # Draw ArUco markers
                    for mid, pos in aruco_markers_px.items():
                        if mid in aruco_imgs:
                            cv_img = aruco_imgs[mid]
                            if len(cv_img.shape) == 2:
                                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                            elif cv_img.shape[2] == 4:
                                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                            else:
                                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                            py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                            win.blit(py_img, (pos[0], pos[1]))

                    # Draw green dots on detected markers (visualization of detection status)
                    detected_ids = sol_projector.get_detected_marker_ids()
                    for mid in detected_ids:
                        if mid in aruco_markers_px:
                            pos = aruco_markers_px[mid]
                            # Center of the marker container
                            center_x = pos[0] + marker_container_size // 2
                            center_y = pos[1] + marker_container_size // 2
                            pygame.draw.circle(win, (0, 255, 0), (center_x, center_y), 10)

                    # Draw gaze point (only if within screen bounds)
                    if current_gaze_pt:
                        gx, gy = current_gaze_pt
                        # Skip rendering if gaze is outside screen (user looking away)
                        if 0 <= gx < screen_w and 0 <= gy < screen_h:
                            pygame.draw.circle(win, (0, 255, 0), current_gaze_pt, 20, 4)
                            pygame.draw.circle(win, (0, 200, 0), current_gaze_pt, 8)

                    # Status bar - show whether using live or cached homography
                    markers_active = sol_projector.is_homography_valid(strict=True) if gaze_method == "2D" else sol_projector.is_calibrated()
                    has_homography = sol_projector.is_homography_valid() if gaze_method == "2D" else sol_projector.is_calibrated()
                    if markers_active:
                        aruco_status = "LIVE"
                        aruco_color = (100, 255, 100)
                    elif has_homography:
                        aruco_status = "CACHED"
                        aruco_color = (255, 255, 100)  # Yellow for cached
                    else:
                        aruco_status = "WAITING"
                        aruco_color = (255, 100, 100)
                    status_text = f"Method: {gaze_method} | Homography: {aruco_status} | Gaze: {'OK' if current_gaze_pt else 'N/A'} | Press Q/ESC to exit"

                    pygame.draw.rect(win, (0, 0, 0), (0, screen_h - 50, screen_w, 50))
                    text_surf = font.render(status_text, True, (255, 255, 255))
                    win.blit(text_surf, (10, screen_h - 40))

                    pygame.display.flip()
                    clock.tick(60)

                except Exception as loop_error:
                    print(f"[Sol Preview] Error in frame {frame_count}: {loop_error}")
                    traceback.print_exc()
                    # Continue running to allow graceful exit
                    time.sleep(0.1)

        except Exception as outer_error:
            print(f"[Sol Preview] FATAL ERROR: {outer_error}")
            traceback.print_exc()

        finally:
            # Save homography for use in calibration
            try:
                self.sol_cached_homography = sol_projector.get_homography()
                if self.sol_cached_homography is not None:
                    print(f"[Sol Preview] Cached homography for calibration")
            except Exception as e:
                print(f"[Sol Preview] Error caching homography: {e}")
            try:
                sol_projector.stop_background_detection()
            except Exception as e:
                print(f"[Sol Preview] Error stopping ArUco detection: {e}")
            try:
                pygame.quit()
            except Exception as e:
                print(f"[Sol Preview] Error quitting pygame: {e}")
            self.in_sol_offset_calibration = False
            # Pause scene stream when returning to settings to avoid native crash
            if self.active_sol_connector:
                self.active_sol_connector.pause_scene_stream()
            # Small delay to ensure cleanup is complete before showing main window
            time.sleep(0.1)
            self.deiconify()

    def build_rec_tab(self, parent, l_font, e_font):
        pad = self.ui_pad  # Use dynamic padding
        r = 0
        
        ttk.Label(parent, text="Recording Resolution:", font=l_font).grid(row=r, sticky="w", **pad)
        ttk.Combobox(parent, textvariable=self.rec_resolution_var, values=["Original", "1920x1080", "1280x720"], state="readonly", font=e_font).grid(row=r, column=1, **pad); r+=1
        
        ttk.Combobox(parent, textvariable=self.rec_resolution_var, values=["Original", "1920x1080", "1280x720"], state="readonly", font=e_font).grid(row=r, column=1, **pad); r+=1
        
        # 1. Webcam Recording
        ttk.Checkbutton(parent, text="Record Webcam Data (Video & Gaze)", variable=self.rec_webcam_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r+=1
        
        # 2. Sol Recording
        def _on_sol_rec_change(*args):
             if not self.rec_sol_data_var.get():
                 self.rec_sol_raw_video_var.set(False)
                 self.chk_sol_raw.configure(state="disabled")
             else:
                 self.chk_sol_raw.configure(state="normal")

        self.rec_sol_data_var.trace_add("write", _on_sol_rec_change)

        ttk.Checkbutton(parent, text="Record Sol Glasses Data (Gaze)", variable=self.rec_sol_data_var).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r+=1
        
        # Indented Option
        f_indent = ttk.Frame(parent); f_indent.grid(row=r, column=0, columnspan=2, sticky="w", padx=(30, 10))
        self.chk_sol_raw = ttk.Checkbutton(f_indent, text="Export Raw Sol Video", variable=self.rec_sol_raw_video_var)
        self.chk_sol_raw.pack(side="left")
        _on_sol_rec_change() # Init state
        r+=1

        ttk.Label(parent, text="* Screen Recording is enabled if Webcam or Sol is recorded.", font=("Arial", 9, "italic")).grid(row=r, column=0, columnspan=2, **pad)


    def parse_rgb(self, s, default=(127,127,127)):
        try:
            parts = [int(x.strip()) for x in s.split(",")]
            if len(parts) == 3: return tuple(np.clip(parts, 0, 255))
        except: pass
        return default

    def safe_get_int(self, var, default):
        try: return int(var.get())
        except: return default
    
    def safe_get_float(self, var, default):
        try: return float(var.get())
        except: return default

    def check_start_button_state(self):
        # Enable Start if (Sol is Disabled OR (Sol is Enabled AND Connected))
        # And ensure at least one tracker enabled? (Optional, logic TBD)
        sol_ok = (not self.enable_sol_var.get()) or self.is_sol_connected
        if sol_ok:
            self.btn_start.configure(state="normal")
            self.btn_practice.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")
            self.btn_practice.configure(state="disabled")

    def toggle_sol_connection(self):
        if not self.is_sol_connected: # Connect
            if not SDK_AVAILABLE:
                messagebox.showerror("Error", "Sol SDK not available.")
                return
            
            self.btn_connect_sol.configure(state="disabled", text="Connecting...")
            self.lbl_sol_status.configure(text="Connecting...", foreground="orange")
            
            self.sol_gaze_queue = queue.Queue(maxsize=100)  # Limit to prevent memory issues
            self.sol_scene_queue = queue.Queue(maxsize=1)
            
            self.active_sol_connector = SolConnector(
                self.sol_ip_var.get(),
                self.safe_get_int(self.sol_port_var, 8080),
                self.sol_gaze_queue,
                self.sol_scene_queue
            )
            
            self.sol_thread = threading.Thread(
                target=run_sol_worker,
                args=(self.active_sol_connector, self.sol_connected_callback, self.sol_failed_callback),
                daemon=True
            )
            self.sol_thread.start()
            self.active_sol_connector._worker_thread = self.sol_thread

        else: # Disconnect (Not fully implemented cleanup in logic, but UI wise)
            # For simplicity, we just mark disconnected and stop flush.
            # Real cleanup happens when stopping connector.
            if self.active_sol_connector:
                self.active_sol_connector.stop()
            self.is_sol_connected = False
            self.btn_connect_sol.configure(text="Connect")
            self.lbl_sol_status.configure(text="Disconnected", foreground="red")
            self.check_start_button_state()
            # Disable Sol offset calibration buttons (3D and 2D)
            if hasattr(self, 'btn_start_sol_offset_cal'):
                self.btn_start_sol_offset_cal.configure(state="disabled")
            if hasattr(self, 'btn_start_sol_2d_offset_cal'):
                self.btn_start_sol_2d_offset_cal.configure(state="disabled")
            if hasattr(self, 'lbl_sol_offset_connect_note'):
                self.lbl_sol_offset_connect_note.configure(text="(Connect Sol glasses first)")
            # Disable preview button
            if hasattr(self, 'btn_preview_sol_gaze'):
                self.btn_preview_sol_gaze.configure(state="disabled")
            if hasattr(self, 'lbl_preview_note'):
                self.lbl_preview_note.configure(text="(Connect Sol glasses first)")

    def sol_connected_callback(self, msg, params, offset):
        self.after(0, lambda: self._on_sol_connected_main(msg, params))

    def _on_sol_connected_main(self, msg, params):
        self.is_sol_connected = True
        self.sol_cam_params = params
        self.btn_connect_sol.configure(state="normal", text="Disconnect")
        self.lbl_sol_status.configure(text=msg, foreground="green")
        self.check_start_button_state()
        self.flush_sol_queues()
        # Enable Sol offset calibration buttons (3D and 2D)
        if hasattr(self, 'btn_start_sol_offset_cal'):
            self.btn_start_sol_offset_cal.configure(state="normal")
        if hasattr(self, 'btn_start_sol_2d_offset_cal'):
            self.btn_start_sol_2d_offset_cal.configure(state="normal")
        if hasattr(self, 'lbl_sol_offset_connect_note'):
            self.lbl_sol_offset_connect_note.configure(text="")
        # Enable preview button
        if hasattr(self, 'btn_preview_sol_gaze'):
            self.btn_preview_sol_gaze.configure(state="normal")
        if hasattr(self, 'lbl_preview_note'):
            self.lbl_preview_note.configure(text="")

    def sol_failed_callback(self, err_msg):
        self.after(0, lambda: self._on_sol_failed_main(err_msg))

    def _on_sol_failed_main(self, err_msg):
        self.is_sol_connected = False
        self.btn_connect_sol.configure(state="normal", text="Connect")
        self.lbl_sol_status.configure(text=f"Error: {err_msg}", foreground="red")
        self.active_sol_connector = None
        self.check_start_button_state()
        # Disable Sol offset calibration buttons (3D and 2D)
        if hasattr(self, 'btn_start_sol_offset_cal'):
            self.btn_start_sol_offset_cal.configure(state="disabled")
        if hasattr(self, 'btn_start_sol_2d_offset_cal'):
            self.btn_start_sol_2d_offset_cal.configure(state="disabled")
        if hasattr(self, 'lbl_sol_offset_connect_note'):
            self.lbl_sol_offset_connect_note.configure(text="(Connect Sol glasses first)")
        # Disable preview button
        if hasattr(self, 'btn_preview_sol_gaze'):
            self.btn_preview_sol_gaze.configure(state="disabled")
        if hasattr(self, 'lbl_preview_note'):
            self.lbl_sol_offset_connect_note.configure(text="(Connect Sol glasses first)")

    def flush_sol_queues(self):
        try:
            if not self.winfo_exists(): return
        except: return
        if not self.is_sol_connected: return
        # Skip flushing during calibration - calibrator needs the frames!
        if getattr(self, 'in_sol_offset_calibration', False):
            self._flush_sol_timer = self.after(100, self.flush_sol_queues)
            return
        # [OPT] Drain queues to prevent backpressure if we are just sitting in menu
        try:
            if self.sol_gaze_queue is not None:
                while True:
                    try: self.sol_gaze_queue.get_nowait()
                    except queue.Empty: break
            if self.sol_scene_queue is not None:
                while True:
                    try: self.sol_scene_queue.get_nowait()
                    except queue.Empty: break
        except Exception as e:
            print(f"[Flush] Error: {e}")
        try:
            self._flush_sol_timer = self.after(100, self.flush_sol_queues)
        except Exception:
            pass  # Window might be destroyed

    def on_start(self, practice_mode=False):
        # Stop webcam preview to release camera before test
        self.stop_preview()

        # Validation
        if self.enable_webcam_var.get():
             if not self.calib_dir_var.get().strip() or not Path(self.calib_dir_var.get().strip()).exists():
                messagebox.showerror("Error", "Webcam enabled but Calibration folder invalid.")
                return

        sw_cm = float(self.scr_width_cm_var.get())
        dist_cm = float(self.view_dist_cm_var.get())
        sw_deg = screen_width_deg_from_cm(sw_cm, dist_cm)

        self.cfg = {
            # General
            'user_name': self.user_var.get(),
            'calib_dir': self.calib_dir_var.get().strip(),
            'stim_dur': self.safe_get_float(self.stim_var, 5.0),
            'pass_dur': self.safe_get_float(self.pass_dur_var, 2.0),
            'blank_dur': self.safe_get_float(self.blank_var, 1.0),
            'radius': self.safe_get_int(self.rad_var, 400),
            'rotate': self.rotate_var.get(),
            'rot_speed': self.safe_get_float(self.rot_speed_var, 60.0),
            'rot_dir': (1 if self.rot_dir_var.get() == "CW" else -1),
            'color_light': self.parse_rgb(self.color_light_var.get(), (255,255,255)),
            'color_dark': self.parse_rgb(self.color_dark_var.get(), (0,0,0)),
            'bg_color': self.parse_rgb(self.bg_color_var.get(), (0,0,0)),
            'screen_width_cm': sw_cm,
            'view_distance_cm': dist_cm,
            'screen_width_deg': sw_deg,
            'gaze_marker_color': self.parse_rgb(self.gaze_color_var.get(), (0,255,0)),
            'gaze_marker_radius': self.safe_get_int(self.gaze_radius_var, 30),
            'gaze_marker_width': self.safe_get_int(self.gaze_width_var, 4),
            'inter_interval_img_path': self.interval_img_path_var.get().strip(),
            'inter_interval_img_dur': self.safe_get_float(self.interval_img_dur_var, 1.5),
            'bg_after_inter_dur': self.safe_get_float(self.bg_after_inter_dur_var, 1.0),

            # Dual Tracker
            'enable_webcam': self.enable_webcam_var.get(),
            'enable_sol': self.enable_sol_var.get(),
            'eval_source': self.eval_source_var.get(),

            # Sol
            'sol_ip': self.sol_ip_var.get(),
            'sol_port': self.safe_get_int(self.sol_port_var, 8080),
            'sol_marker_k': self.safe_get_int(self.sol_marker_k_var, 6),
            'sol_marker_n': self.safe_get_int(self.sol_marker_n_var, 4),
            'sol_marker_size': self.safe_get_int(self.sol_marker_size_var, 80),
            'sol_aruco_dict': self.sol_aruco_dict_var.get(),
            'sol_screen_phy_width_mm': self.safe_get_float(self.scr_width_cm_var, 53.0) * 10.0, # Convert cm to mm
            'sol_pose_smooth': self.safe_get_float(self.sol_pose_smooth_var, 0.1),
            'sol_gaze_smooth': self.safe_get_float(self.sol_gaze_smooth_var, 0.15),
            'sol_gaze_method': self.sol_gaze_method_var.get(),  # "3D" or "2D"

            # Recording
            'rec_resolution': self.rec_resolution_var.get(),
            'rec_webcam': self.rec_webcam_var.get(),
            'rec_sol_data': self.rec_sol_data_var.get(),
            'rec_sol_raw_video': self.rec_sol_raw_video_var.get(),
            'camera_id': self.safe_get_int(self.camera_idx_var, 0),
            'show_gaze_marker': self.show_gaze_marker_var.get(),

            # [NEW] Practice mode and Paper color
            'practice_mode': practice_mode,
            'paper_color': self.paper_color_var.get(),
        }

        # Cancel pending timers while test runs
        if self._auto_save_timer is not None:
            self.after_cancel(self._auto_save_timer)
            self._auto_save_timer = None
        if self._flush_sol_timer is not None:
            self.after_cancel(self._flush_sol_timer)
            self._flush_sol_timer = None
        self._auto_save_settings()

        # Hide window and exit mainloop (like preview_sol_gaze pattern)
        # Window stays alive so Sol connection is preserved
        self.withdraw()
        self.quit()

    def on_start_practice(self):
        """Start practice mode - runs test without recording, returns to config after."""
        self.on_start(practice_mode=True)

    def _collect_gui_values(self):
        """Collect all GUI values for saving."""
        return {
            # General settings
            'user_name': self.user_var.get(),
            'calib_dir': self.calib_dir_var.get(),

            # Tracker settings
            'enable_webcam': self.enable_webcam_var.get(),
            'enable_sol': self.enable_sol_var.get(),
            'eval_source': self.eval_source_var.get(),

            # Sol settings
            'sol_ip': self.sol_ip_var.get(),
            'sol_port': self.sol_port_var.get(),
            'sol_marker_size': self.sol_marker_size_var.get(),
            'sol_marker_k': self.sol_marker_k_var.get(),
            'sol_marker_n': self.sol_marker_n_var.get(),
            'sol_aruco_dict': self.sol_aruco_dict_var.get(),
            'sol_pose_smooth': self.sol_pose_smooth_var.get(),
            'sol_gaze_smooth': self.sol_gaze_smooth_var.get(),
            'sol_gaze_method': self.sol_gaze_method_var.get(),

            # Sol offset calibration settings
            'sol_offset_target_img': self.sol_offset_target_img_var.get(),
            'sol_offset_target_size': self.sol_offset_target_size_var.get(),
            'sol_offset_num_points': self.sol_offset_num_points_var.get(),
            'sol_offset_user_screen': self.sol_offset_user_screen_var.get(),
            'sol_offset_tester_screen': self.sol_offset_tester_screen_var.get(),
            'sol_preview_screen': self.sol_preview_screen_var.get() if hasattr(self, 'sol_preview_screen_var') else "0",

            # Stimulus settings
            'gaze_color': self.gaze_color_var.get(),
            'gaze_radius': self.gaze_radius_var.get(),
            'gaze_width': self.gaze_width_var.get(),
            'stim_duration': self.stim_var.get(),
            'pass_duration': self.pass_dur_var.get(),
            'blank_duration': self.blank_var.get(),
            'radius': self.rad_var.get(),
            'rotate': self.rotate_var.get(),
            'rot_speed': self.rot_speed_var.get(),
            'rot_dir': self.rot_dir_var.get(),
            'color_light': self.color_light_var.get(),
            'color_dark': self.color_dark_var.get(),
            'bg_color': self.bg_color_var.get(),
            'scr_width_cm': self.scr_width_cm_var.get(),
            'view_dist_cm': self.view_dist_cm_var.get(),
            'interval_img_path': self.interval_img_path_var.get(),
            'interval_img_dur': self.interval_img_dur_var.get(),
            'bg_after_inter_dur': self.bg_after_inter_dur_var.get(),

            # Recording settings
            'camera_id': self.camera_idx_var.get(),
            'rec_webcam': self.rec_webcam_var.get(),
            'rec_sol_data': self.rec_sol_data_var.get(),
            'rec_sol_raw_video': self.rec_sol_raw_video_var.get(),
            'show_gaze_marker': self.show_gaze_marker_var.get(),
            'paper_color': self.paper_color_var.get(),
        }

    def _auto_save_settings(self):
        """Silently save all settings to file."""
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._collect_gui_values(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Settings] Failed to auto-save: {e}")

    def _schedule_auto_save(self, *args):
        """Debounced auto-save: waits 1 second after the last change before saving."""
        if self._suppress_auto_save:
            return
        if self._auto_save_timer is not None:
            self.after_cancel(self._auto_save_timer)
        self._auto_save_timer = self.after(1000, self._auto_save_settings)

    def _setup_auto_save_traces(self):
        """Attach trace callbacks to all settings variables for auto-save."""
        tracked_vars = [
            self.user_var, self.calib_dir_var,
            self.gaze_color_var, self.gaze_radius_var, self.gaze_width_var,
            self.stim_var, self.pass_dur_var, self.blank_var, self.rad_var,
            self.rotate_var, self.rot_speed_var, self.rot_dir_var,
            self.color_light_var, self.color_dark_var, self.bg_color_var,
            self.scr_width_cm_var, self.view_dist_cm_var,
            self.interval_img_path_var, self.interval_img_dur_var, self.bg_after_inter_dur_var,
            self.enable_webcam_var, self.enable_sol_var, self.eval_source_var,
            self.sol_ip_var, self.sol_port_var, self.sol_marker_k_var, self.sol_marker_n_var,
            self.sol_marker_size_var, self.sol_aruco_dict_var,
            self.sol_pose_smooth_var, self.sol_gaze_smooth_var, self.sol_gaze_method_var,
            self.sol_cal_show_gaze_var,
            self.sol_offset_target_img_var, self.sol_offset_target_size_var,
            self.sol_offset_num_points_var, self.sol_offset_user_screen_var,
            self.sol_offset_tester_screen_var,
            self.rec_resolution_var, self.rec_webcam_var, self.rec_sol_data_var,
            self.rec_sol_raw_video_var,
            self.camera_idx_var, self.show_gaze_marker_var, self.paper_color_var,
        ]
        if hasattr(self, 'sol_preview_screen_var'):
            tracked_vars.append(self.sol_preview_screen_var)
        for var in tracked_vars:
            var.trace_add('write', self._schedule_auto_save)

    def _auto_load_settings(self):
        """Silently load settings from file on startup."""
        self._suppress_auto_save = True
        try:
            if not LAST_SETTINGS_FILE.exists():
                print("[Settings] No previous settings file found.")
                return

            with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply all settings
            if 'user_name' in data: self.user_var.set(data['user_name'])
            if 'calib_dir' in data: self.calib_dir_var.set(data['calib_dir'])

            if 'enable_webcam' in data: self.enable_webcam_var.set(data['enable_webcam'])
            if 'enable_sol' in data: self.enable_sol_var.set(data['enable_sol'])
            if 'eval_source' in data: self.eval_source_var.set(data['eval_source'])

            if 'sol_ip' in data: self.sol_ip_var.set(data['sol_ip'])
            if 'sol_port' in data: self.sol_port_var.set(str(data['sol_port']))
            if 'sol_marker_k' in data: self.sol_marker_k_var.set(str(data['sol_marker_k']))
            if 'sol_marker_n' in data: self.sol_marker_n_var.set(str(data['sol_marker_n']))
            if 'sol_marker_size' in data: self.sol_marker_size_var.set(str(data['sol_marker_size']))
            if 'sol_aruco_dict' in data: self.sol_aruco_dict_var.set(data['sol_aruco_dict'])
            if 'sol_pose_smooth' in data: self.sol_pose_smooth_var.set(str(data['sol_pose_smooth']))
            if 'sol_gaze_smooth' in data: self.sol_gaze_smooth_var.set(str(data['sol_gaze_smooth']))
            if 'sol_gaze_method' in data: self.sol_gaze_method_var.set(data['sol_gaze_method'])

            if 'sol_offset_target_img' in data: self.sol_offset_target_img_var.set(data['sol_offset_target_img'])
            if 'sol_offset_target_size' in data: self.sol_offset_target_size_var.set(str(data['sol_offset_target_size']))
            if 'sol_offset_num_points' in data: self.sol_offset_num_points_var.set(str(data['sol_offset_num_points']))
            if 'sol_offset_user_screen' in data: self.sol_offset_user_screen_var.set(data['sol_offset_user_screen'])
            if 'sol_offset_tester_screen' in data: self.sol_offset_tester_screen_var.set(data['sol_offset_tester_screen'])
            if 'sol_preview_screen' in data and hasattr(self, 'sol_preview_screen_var'):
                self.sol_preview_screen_var.set(data['sol_preview_screen'])

            if 'gaze_color' in data: self.gaze_color_var.set(data['gaze_color'])
            if 'gaze_radius' in data: self.gaze_radius_var.set(str(data['gaze_radius']))
            if 'gaze_width' in data: self.gaze_width_var.set(str(data['gaze_width']))
            if 'stim_duration' in data: self.stim_var.set(str(data['stim_duration']))
            if 'pass_duration' in data: self.pass_dur_var.set(str(data['pass_duration']))
            if 'blank_duration' in data: self.blank_var.set(str(data['blank_duration']))
            if 'radius' in data: self.rad_var.set(str(data['radius']))
            if 'rotate' in data: self.rotate_var.set(data['rotate'])
            if 'rot_speed' in data: self.rot_speed_var.set(str(data['rot_speed']))
            if 'rot_dir' in data: self.rot_dir_var.set(data['rot_dir'])
            if 'color_light' in data: self.color_light_var.set(data['color_light'])
            if 'color_dark' in data: self.color_dark_var.set(data['color_dark'])
            if 'bg_color' in data: self.bg_color_var.set(data['bg_color'])
            if 'scr_width_cm' in data: self.scr_width_cm_var.set(str(data['scr_width_cm']))
            if 'view_dist_cm' in data: self.view_dist_cm_var.set(str(data['view_dist_cm']))
            if 'interval_img_path' in data: self.interval_img_path_var.set(data['interval_img_path'])
            if 'interval_img_dur' in data: self.interval_img_dur_var.set(str(data['interval_img_dur']))
            if 'bg_after_inter_dur' in data: self.bg_after_inter_dur_var.set(str(data['bg_after_inter_dur']))

            if 'camera_id' in data: self.camera_idx_var.set(str(data['camera_id']))
            if 'rec_webcam' in data: self.rec_webcam_var.set(data['rec_webcam'])
            if 'rec_sol_data' in data: self.rec_sol_data_var.set(data['rec_sol_data'])
            if 'rec_sol_raw_video' in data: self.rec_sol_raw_video_var.set(data['rec_sol_raw_video'])
            if 'show_gaze_marker' in data: self.show_gaze_marker_var.set(data['show_gaze_marker'])
            if 'paper_color' in data: self.paper_color_var.set(data['paper_color'])

            print(f"[Settings] Auto-loaded from {LAST_SETTINGS_FILE}")
        except Exception as e:
            print(f"[Settings] Failed to auto-load: {e}")
        finally:
            self._suppress_auto_save = False

    def _on_username_changed(self, *args):
        """Called when username changes - update user-dependent displays."""
        # Update offset calibration status displays
        if hasattr(self, 'lbl_sol_offset_pitch'):
            self.update_sol_offset_display()
        if hasattr(self, 'lbl_sol_2d_offset_status'):
            self.update_sol_2d_offset_display()

    def _clear_homography_cache(self):
        """Clear the cached homography for debugging."""
        self.sol_cached_homography = None
        print("[Debug] Homography cache cleared")
        messagebox.showinfo("Cleared", "Homography cache cleared. Next preview will compute a fresh homography.")

# ---------- Sol Thread Helper ----------
def run_sol_worker(connector, on_connect, on_fail):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connector.run_session(on_connect, on_fail))
    except Exception as e:
        print(f"[Sol Worker] Unexpected error: {e}")
        traceback.print_exc()
        if on_fail:
            on_fail(f"Unexpected error: {e}")
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending: t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as cleanup_error:
            print(f"[Sol Worker] Cleanup error: {cleanup_error}")
        finally:
            try:
                loop.close()
            except Exception:
                pass

# ---------- Main Experiment ----------
def run_test(cfg, sol_context=None):
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    win = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    # 1. Initialize Webcam Tracker
    gf = None
    webcam = None
    if cfg['enable_webcam']:
        profile_dir = Path(cfg['calib_dir'])
        if not profile_dir.exists():
            messagebox.showerror("Error", "Calibration folder missing")
            return
        # [FIX] Sync screen size with Pygame
        dcfg = DefaultConfig()
        dcfg.screen_size = np.array([W, H])
        
        # [FIX] Pass Camera ID
        cid = cfg.get('camera_id', 0)
        webcam = WebCamCamera(webcam_id=cid)

        calib = SVRCalibration(model_save_path=str(profile_dir))
        gf = GazeFollower(config=dcfg, calibration=calib, camera=webcam)
        if not gf.calibration.has_calibrated:
            messagebox.showerror("Error", "Calibration not found. Run calibration.py first.")
            return
        ensure_pygame_focus()
        gf.start_sampling()
        time.sleep(0.1)

    # 2. Initialize Sol Tracker
    sol_connector = None
    sol_projector = None
    sol_gaze_queue = None
    sol_scene_queue = None

    # [DEBUG] Sol gaze processing counters
    sol_debug_counters = {
        'total_frames': 0,
        'gaze_queue_empty': 0,
        'not_calibrated': 0,
        'attribute_error': 0,
        'projection_failed': 0,
        'valid_gaze': 0,
        'smoothed_gaze': 0,
        'frames_with_gaze_data': 0,  # Frames where we got gaze from queue
        'used_cached_gaze': 0,  # Frames where we used last known gaze due to queue empty
        'zero_norm_vector': 0,  # Gaze direction vector has zero length
        'gaze_data_structure_error': 0  # Missing left_eye, right_eye, or combined fields
    }

    # [NOTE] Gaze caching is now handled by persistent sol_gaze_pt variable (matches preview behavior)

    # Load Sol offset calibration if available
    sol_offset = None
    sol_2d_offset_model = None
    if cfg['enable_sol'] and SOL_OFFSET_AVAILABLE:
        username = cfg.get('user_name', 'anonymous')
        calib_dir = Path(cfg.get('calib_dir', 'calibration_profiles'))

        # Load 3D angular offset (for 3D gaze method)
        sol_offset = load_sol_offset(username, calib_dir)
        if sol_offset:
            print(f"[Sol Offset] Loaded 3D offset for user '{username}':")
            print(f"             Pitch: {sol_offset['pitch_offset_deg']:.2f} deg ({sol_offset['pitch_offset_rad']:.4f} rad)")
            print(f"             Yaw: {sol_offset['yaw_offset_deg']:.2f} deg ({sol_offset['yaw_offset_rad']:.4f} rad)")
        else:
            print(f"[Sol Offset] No 3D offset calibration found for user '{username}'.")

        # Load 2D SVR offset model (for 2D gaze method)
        if SOL_2D_OFFSET_AVAILABLE:
            sol_2d_offset_data = load_sol_2d_offset(username, calib_dir)
            if sol_2d_offset_data and sol_2d_offset_data.get('model') and sol_2d_offset_data['model'].is_trained:
                sol_2d_offset_model = sol_2d_offset_data['model']
                print(f"[Sol 2D Offset] Loaded 2D offset model for user '{username}':")
                print(f"             Points: {sol_2d_offset_data.get('num_calibration_points', '?')}, Trained: {sol_2d_offset_data.get('calibration_timestamp', '?')}")
            else:
                print(f"[Sol 2D Offset] No 2D offset calibration found for user '{username}'. Using raw gaze_2d.")

    # Aruco Assets
    aruco_markers_px = {}
    aruco_imgs = {}
    marker_container_size = 0
    physical_width_m = cfg['sol_screen_phy_width_mm'] / 1000.0

    if cfg['enable_sol'] and SDK_AVAILABLE:
        # Use existing context if passed
        if sol_context and sol_context.get('connector'):
            print("Using existing Sol Connection...")
    

            sol_connector = sol_context['connector']
            sol_gaze_queue = sol_context['gaze_queue']
            sol_scene_queue = sol_context['scene_queue']
            cam_params = sol_context.get('cam_params', {})

            # Resume scene stream for ArUco detection during test
            sol_connector.resume_scene_stream()
            
            # Setup Projector with actual params
            # Setup Projector with actual params
            aruco_dict_key = cfg['sol_aruco_dict']
            aruco_dict_map = {
                "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
                "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
                "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
                "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
                "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
                "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            }
            
            # Default to 4x4_250 if not found
            selected_dict_id = aruco_dict_map.get(aruco_dict_key, cv2.aruco.DICT_4X4_250)
            adict = cv2.aruco.getPredefinedDictionary(selected_dict_id)
            
            sol_cfg_for_assets = {
                'marker_k': cfg['sol_marker_k'],
                'marker_n': cfg['sol_marker_n'],
                'marker_pattern_size': cfg['sol_marker_size']
            }
            aruco_markers_px, aruco_imgs = create_calibration_assets(W, H, adict, sol_cfg_for_assets)
            marker_container_size = cfg['sol_marker_size'] + 30 

            cam_matrix = cam_params.get('cam_matrix')
            dist_coeffs = cam_params.get('dist_coeffs')
            
            # Fallback if params missing (shouldn't happen if connected)
            if cam_matrix is None: 
                cam_matrix = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=float)
                dist_coeffs = np.zeros(5)

            sol_projector = ScreenProjector3D(cam_matrix, dist_coeffs, adict, smoothing_factor=cfg['sol_pose_smooth'])

            # Set 2D gaze smoothing factor and reset smoothing state
            sol_projector.set_gaze_2d_smoothing_factor(cfg.get('sol_gaze_smooth', 0.15))
            sol_projector.reset_gaze_2d_smoothing()

            # Set 2D offset model if available
            if sol_2d_offset_model and sol_2d_offset_model.is_trained:
                sol_projector.set_gaze_2d_offset_model(sol_2d_offset_model)
                print(f"[Sol 2D Offset] Applied to projector")

            # [OPT] Start background ArUco detection thread
            sol_projector.start_background_detection(
                cfg['sol_marker_size']/W*physical_width_m,
                aruco_markers_px,
                marker_container_size,
                W, H, physical_width_m
            )

        else:
            # Fallback: validation should have caught this, but if we are here without context,
            # we try to connect or just fail.
            print("No existing Sol connection found. Attempting legacy init...")
            try:
                sol_connector = SolConnector(cfg['sol_ip'], cfg['sol_port'], sol_gaze_queue, sol_scene_queue)
                th = threading.Thread(target=run_sol_worker, args=(sol_connector, None, None), daemon=True)
                th.start()
                sol_connector._worker_thread = th

                # Default Projector
                adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                cam_matrix = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=float)
                sol_projector = ScreenProjector3D(cam_matrix, np.zeros(5), adict, smoothing_factor=cfg['sol_pose_smooth'])

                # Set 2D offset model if available (legacy init)
                if sol_2d_offset_model and sol_2d_offset_model.is_trained:
                    sol_projector.set_gaze_2d_offset_model(sol_2d_offset_model)
                    print(f"[Sol 2D Offset] Applied to projector (legacy)")

                # Assets
                sol_cfg_for_assets = {
                    'marker_k': cfg['sol_marker_k'],
                    'marker_n': cfg['sol_marker_n'],
                    'marker_pattern_size': cfg['sol_marker_size']
                }
                aruco_markers_px, aruco_imgs = create_calibration_assets(W, H, adict, sol_cfg_for_assets)

                marker_container_size = cfg['sol_marker_size'] + 30

            except Exception as e:
                print(f"Sol init error: {e}")
                messagebox.showwarning("Sol Error", f"Failed to init Sol: {e}")
                cfg['enable_sol'] = False

    # [NEW] Frame Helpers
    def get_sol_frame():
        try:
            frame_obj = None
            if sol_scene_queue:
                # [OPT] Drain queue to get latest frame (limit to 10 to prevent delays)
                for _ in range(10):
                    try:
                        frame_obj = sol_scene_queue.get_nowait()
                    except queue.Empty:
                        break

            if frame_obj:
                result = None
                if hasattr(frame_obj, 'img'):
                     result = frame_obj.img # Sol SDK (v2) Frame has .img (numpy)

                # Legacy / Buffer Fallback
                elif hasattr(frame_obj, 'get_buffer'):
                    try:
                        # Determine Resolution
                        w, h = 1328, 1200 # Default Sol Resolution
                        if sol_context and 'cam_params' in sol_context:
                             try:
                                 res = sol_context['cam_params'].resolution
                                 if res: w, h = res.width, res.height
                             except: pass

                        buf = frame_obj.get_buffer()
                        arr = np.frombuffer(buf, dtype=np.uint8)
                        arr = arr.reshape((h, w, 3))
                        result = arr
                    except Exception as e:
                        print(f"Sol Frame Convert Err: {e}")
                        return None

                # Assume it's already Numpy
                else:
                    result = frame_obj

                # [FIX] Validate the frame before returning
                if result is not None:
                    if hasattr(result, 'shape') and len(result.shape) >= 2:
                        if result.shape[0] > 0 and result.shape[1] > 0:
                            return result
                return None

            return None
        except Exception as e:
            # [FIX] Catch any unexpected errors in frame retrieval
            print(f"[get_sol_frame] Unexpected error: {e}")
            return None

    def get_webcam_frame():
        # [FIX] Unified webcam frame retrieval
        if gf and hasattr(gf, 'camera') and gf.camera:
            return getattr(gf.camera, 'last_frame', None)
        elif webcam:
            # Check both last_frame (WebCamCamera) and latest_frame (WebcamFrameGrabber)
            frame = getattr(webcam, 'last_frame', None) or getattr(webcam, 'latest_frame', None)
            if frame is not None:
                return frame
        return None

    def pump_recorder():
         # [OPT] Helper to keep recorder buffer fed during blocking setup
         try:
             if recorder and recorder.running:
                  sol_f = get_sol_frame() if cfg.get('rec_sol_raw_video') else None
                  wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
                  rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')
                  recorder.process_and_record(
                      wb_f,
                      win if rec_screen else None,
                      sol_frame=sol_f
                  )
         except Exception as e:
             # [FIX] Don't let pump_recorder crash the main loop
             print(f"[pump_recorder] Error: {e}")

    # 3. Initialize Recorder (or DummyRecorder for practice mode)
    if cfg.get('practice_mode', False):
        print("[Practice Mode] Using DummyRecorder - no data will be saved")
        recorder = DummyRecorder()
    else:
        recorder = Recorder(output_dir="VA_output", subject_id=cfg['user_name'], session_num="1", is_va=True)

    # [NEW] Paper Color Mode - override colors for paper-like appearance
    PAPER_COLOR_ENABLED = cfg.get('paper_color', False)
    if PAPER_COLOR_ENABLED:
        PAPER_GRAY = (128, 128, 128)  # Gray for background and non-grating circle
        PAPER_WHITE = (255, 255, 255)  # White for grating bright bars and borders
        PAPER_BLACK = (0, 0, 0)  # Black for grating dark bars
        PAPER_BORDER_WIDTH = 5  # 5px white border
        print("[Paper Color Mode] Enabled - gray bg, black/white grating, 5px white border")

    # [FIX] Screen recording throttle - only capture every Nth frame to prevent queue overflow
    SCREEN_RECORD_EVERY_N_FRAMES = 2  # Capture at 30fps instead of 60fps
    screen_record_frame_counter = 0

    # Staircase
    stair = Staircase(start=2.0, step=2.0, minv=2.0, maxv=20.0)

    # [FIX] Calculate safe stimulus positions to avoid ArUco markers
    # Uses actual marker rectangles to compute minimum distance from stimulus circle
    MARKER_GAP_PX = 20  # Required gap between stimulus edge and nearest marker
    if cfg['enable_sol'] and aruco_markers_px:
        # Build list of marker rectangles: (x1, y1, x2, y2)
        marker_rects = []
        for mid, pos in aruco_markers_px.items():
            marker_rects.append((pos[0], pos[1], pos[0] + marker_container_size, pos[1] + marker_container_size))

        def min_dist_to_markers(cx, cy):
            """Minimum distance from point (cx, cy) to any marker rectangle."""
            min_d = float('inf')
            for (x1, y1, x2, y2) in marker_rects:
                dx = max(x1 - cx, 0, cx - x2)
                dy = max(y1 - cy, 0, cy - y2)
                d = math.sqrt(dx * dx + dy * dy)
                min_d = min(min_d, d)
            return min_d

        # Account for paper color border that extends beyond the circle radius
        border_extra = PAPER_BORDER_WIDTH if PAPER_COLOR_ENABLED else 0

        def max_radius_for_center(cx, cy):
            """Max radius for a circle at (cx, cy) that keeps MARKER_GAP_PX from all markers and screen edges."""
            r_markers = min_dist_to_markers(cx, cy) - MARKER_GAP_PX - border_extra
            r_edges = min(cx, cy, W - cx, H - cy) - border_extra
            return min(r_markers, r_edges)

        # Place stimulus centers at 1/4 and 3/4 of screen width, vertically centered
        left_center_x = W // 4
        right_center_x = 3 * W // 4
        center_y = H // 2

        # Compute max radius for each center, then take the smaller one (both must be same size)
        max_r_left = max_radius_for_center(left_center_x, center_y)
        max_r_right = max_radius_for_center(right_center_x, center_y)
        # Also ensure gap between the two stimuli
        max_r_gap = (right_center_x - left_center_x) // 2 - MARKER_GAP_PX - border_extra
        max_safe_radius = int(min(max_r_left, max_r_right, max_r_gap))

        original_radius = cfg['radius']
        if original_radius > max_safe_radius:
            print(f"[Stimulus] Radius {original_radius} exceeds safe area, limiting to {max_safe_radius}")
            cfg['radius'] = max_safe_radius

        centers = {'left': (left_center_x, center_y), 'right': (right_center_x, center_y)}
        print(f"[Stimulus] Safe centers: left={centers['left']}, right={centers['right']}, radius={cfg['radius']}, max_safe={max_safe_radius}")
    else:
        # No Sol markers, use original positions
        centers = {'left': (W // 4, H // 2), 'right': (3 * W // 4, H // 2)}
    clock   = pygame.time.Clock()
    results = []

    # Background Surface
    def build_bg_surface(rad):
        # [PAPER COLOR MODE] Use gray colors if enabled
        if PAPER_COLOR_ENABLED:
            bg_color = PAPER_GRAY
            other_color = PAPER_GRAY
            border_color = PAPER_WHITE
            border_width = PAPER_BORDER_WIDTH
        else:
            bg_color = to_rgb_tuple(cfg['bg_color'])
            other_color = to_rgb_tuple(mean_color_rgb(cfg['color_light'], cfg['color_dark']))
            border_color = None
            border_width = 0

        surf = pygame.Surface((W, H))
        surf.fill(bg_color)

        # [NEW] Draw Aruco Markers
        if cfg['enable_sol']:
            for mid, pos in aruco_markers_px.items():
                if mid in aruco_imgs:
                    # Convert numpy image to pygame surface
                    cv_img = aruco_imgs[mid]
                    if len(cv_img.shape) == 2:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                    elif len(cv_img.shape) == 3:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                    # Create Pygame Surface
                    py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                    surf.blit(py_img, (pos[0], pos[1]))

        for pos in (centers['left'], centers['right']):
            # Draw white border first (if paper color mode)
            if border_color and border_width > 0:
                pygame.draw.circle(surf, border_color, pos, rad + border_width)
            # Draw the inner circle
            pygame.draw.circle(surf, other_color, pos, rad)
        return surf

    # Interval Img
    interval_img_surf = None
    if cfg.get('inter_interval_img_path'):
        try:
            interval_img_surf = pygame.image.load(cfg['inter_interval_img_path']).convert_alpha()
        except: pass

    # [FIX] Persistent gaze point for display across all phases (interval, feedback, etc.)
    # Uses list for mutability in nested function closures.
    _display_gaze = [None]

    def process_and_draw_gaze(surface):
        """Process Sol gaze data and draw gaze marker on the surface if enabled.
        Called from interval screens, feedback screens, etc. to show gaze continuously."""
        if not cfg['enable_sol'] or not sol_connector or not sol_projector:
            return
        # Submit scene frame for pose detection
        sol_frame_numpy = get_sol_frame()
        if sol_frame_numpy is not None:
            try:
                sol_projector.submit_frame_for_pose(sol_frame_numpy)
            except Exception:
                pass
        # Drain gaze queue to get latest
        latest_gaze = None
        if sol_gaze_queue:
            for _ in range(20):
                try:
                    latest_gaze = sol_gaze_queue.get_nowait()
                except queue.Empty:
                    break
        if latest_gaze:
            sol_gaze_method = cfg.get('sol_gaze_method', '3D')
            can_process = False
            if sol_gaze_method == '2D':
                can_process = sol_projector.is_homography_valid()
            else:
                can_process = sol_projector.is_calibrated()
            if can_process:
                try:
                    if not hasattr(latest_gaze, 'combined'):
                        pass
                    else:
                        raw_gaze_2d = None
                        if hasattr(latest_gaze.combined, 'gaze_2d'):
                            g2d = latest_gaze.combined.gaze_2d
                            raw_gaze_2d = (g2d.x, g2d.y)
                        if sol_gaze_method == '2D':
                            if raw_gaze_2d:
                                screen_pt = sol_projector.project_gaze_2d_to_screen(raw_gaze_2d)
                                if screen_pt:
                                    _display_gaze[0] = screen_pt
                        else:
                            if hasattr(latest_gaze, 'left_eye') and hasattr(latest_gaze, 'right_eye'):
                                left_o = latest_gaze.left_eye.gaze.origin
                                right_o = latest_gaze.right_eye.gaze.origin
                                gaze_origin_mm = np.array([
                                    (left_o.x + right_o.x) / 2.0,
                                    (left_o.y + right_o.y) / 2.0,
                                    (left_o.z + right_o.z) / 2.0
                                ])
                                g3d = latest_gaze.combined.gaze_3d
                                gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])
                                gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                                norm = np.linalg.norm(gaze_direction_vec)
                                if norm > 0:
                                    gaze_direction_unit = gaze_direction_vec / norm
                                    if sol_offset is not None:
                                        gaze_direction_unit = apply_angular_offset(
                                            gaze_direction_unit,
                                            sol_offset['pitch_offset_rad'],
                                            sol_offset['yaw_offset_rad']
                                        )
                                    gaze_origin_m = gaze_origin_mm / 1000.0
                                    screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                    if screen_pt_m is not None:
                                        pix = sol_projector.physical_to_pixels(screen_pt_m, W, physical_width_m)
                                        if pix:
                                            _display_gaze[0] = (int(pix[0]), int(pix[1]))
                except (AttributeError, Exception):
                    pass
        # Draw gaze marker
        if cfg.get('show_gaze_marker', True) and _display_gaze[0] is not None:
            gx, gy = _display_gaze[0]
            if 0 <= gx < W and 0 <= gy < H:
                pygame.draw.circle(surface, to_rgb_tuple(cfg['gaze_marker_color']),
                                   (gx, gy), cfg['gaze_marker_radius'], cfg['gaze_marker_width'])

    def show_interval_center(duration_s):
        t0 = time.time()
        # [TODO: This logic is same as original, just ensuring we record if needed]
        # For simplicity, not recording during interval, or should we?
        # User requested "Screen Recording", implying whole session.
        # Minimal implementation for interval recording:
        # [PAPER COLOR MODE] Use gray background
        interval_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        while time.time() - t0 < duration_s:
          try:
            # [FIX] Pump events to prevent hanging/hard crash
            pygame.event.pump()
            win.fill(interval_bg)
            if interval_img_surf:
                win.blit(interval_img_surf, ((W-interval_img_surf.get_width())//2, (H-interval_img_surf.get_height())//2))
            else:
                # Draw Cross
                cx, cy = W // 2, H // 2
                pygame.draw.line(win, (255,255,255), (cx-40, cy), (cx+40, cy), 4)
                pygame.draw.line(win, (255,255,255), (cx, cy-40), (cx, cy+40), 4)

            # Draw Aruco here too? Ideally yes for continuous tracking.
            if cfg['enable_sol']:
                 for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape)==2: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB)
                        else: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))

            # Process and draw gaze marker (continuous tracking)
            process_and_draw_gaze(win)

            pygame.display.flip()

            # Record
            sol_f = get_sol_frame() if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame()
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

            recorder.process_and_record(
                wb_f,
                win if rec_screen else None,
                sol_frame=sol_f
            )
            # [OPT] Restored to 30 FPS for high-rate data collection
            clock.tick(30)
          except Exception as e:
            print(f"[show_interval_center] Error: {e}")

    def show_background_blank(duration_s):
        t0 = time.time()
        # [PAPER COLOR MODE] Use gray background
        blank_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        while time.time() - t0 < duration_s:
          try:
            pygame.event.pump()
            win.fill(blank_bg)
            if cfg['enable_sol']:
                 for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape)==2: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_GRAY2RGB)
                        else: cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))

            # Process and draw gaze marker (continuous tracking)
            process_and_draw_gaze(win)

            pygame.display.flip()

            sol_f = get_sol_frame() if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

            recorder.process_and_record(
                wb_f,
                win if rec_screen else None,
                sol_frame=sol_f
            )
            clock.tick(60)
          except Exception as e:
            print(f"[show_background_blank] Error: {e}")

    # Initial delay longer than inter-trial interval to allow homography setup
    show_interval_center(5.0)

    # --- Experiment Loop ---
    # [FIX] Pre-create fonts to avoid GDI resource leak (was creating new font every frame!)
    status_font = pygame.font.SysFont(None, 30)
    feedback_font = pygame.font.SysFont(None, 100)
    trial_number = 0
    while not stair.done():
      try:  # [FIX] Wrap entire trial in try-except for crash debugging
        trial_number += 1
        print(f"[Trial] Starting trial {trial_number}")

        side  = random.choice(['left', 'right'])
        cpd   = float(stair.freq)
        cs    = cpd * cfg['screen_width_deg']
        rad   = int(cfg['radius'])
        diam  = rad * 2

        # [OPT] Pump data to prevent gap
        pump_recorder()
        bg_surface = build_bg_surface(rad)

        pump_recorder()
        xx_patch, yy_patch, circle_mask_patch = prepare_patch_grid(rad)

        pump_recorder()
        circle_alpha = (circle_mask_patch.astype(np.uint8) * 255)

        # [OPT] Pre-generate base pattern (0 degrees) once per trial
        # [PAPER COLOR MODE] Use black/white for grating
        if PAPER_COLOR_ENABLED:
            grating_dark = PAPER_BLACK
            grating_light = PAPER_WHITE
        else:
            grating_dark = cfg['color_dark']
            grating_light = cfg['color_light']

        patch_rgb_0deg = generate_grating_oriented_patch(cs, xx_patch, yy_patch, 0.0, W, grating_dark, grating_light, DO_BLUR)
        base_surf = pygame.Surface((diam, diam), pygame.SRCALPHA)
        pygame.surfarray.blit_array(base_surf.subsurface((0, 0, diam, diam)), patch_rgb_0deg.swapaxes(0, 1)[:,:,:3])
        pa = pygame.surfarray.pixels_alpha(base_surf)
        pa[:] = circle_alpha.T
        del pa

        start  = time.time()
        passed = False
        hold_start = None

        # Initialize Sol frame variable to prevent UnboundLocalError when Sol is disabled
        sol_frame_numpy = None

        # [FIX] Match preview behavior: sol_gaze_pt persists between frames (not reset to None)
        # This provides smoother gaze tracking like the preview window
        sol_gaze_pt = None  # Initialize once per trial, persists between frames

        # Determine center pos
        x0 = centers[side][0] - rad
        y0 = centers[side][1] - rad

        while True:
          try:  # [FIX] Inner try-except for frame-level errors
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    # Quit test early - cleanup and return to main loop
                    if gf: gf.stop_sampling(); gf.release()
                    if sol_projector: sol_projector.stop_background_detection()
                    recorder.close()
                    pygame.quit()
                    return

            t = time.time() - start
            win.blit(bg_surface, (0, 0))

            # [PAPER COLOR MODE] Draw white border around grating stimulus
            if PAPER_COLOR_ENABLED:
                stim_center = (x0 + rad, y0 + rad)
                pygame.draw.circle(win, PAPER_WHITE, stim_center, rad + PAPER_BORDER_WIDTH)

            # Draw Stimulus
            if cfg['rotate']:
                angle = (t * cfg['rot_speed'] * cfg['rot_dir']) % 360.0
                rotated_surf = pygame.transform.rotate(base_surf, -angle)
                rot_rect = rotated_surf.get_rect(center=(x0 + rad, y0 + rad))
                win.blit(rotated_surf, rot_rect)
            else:
                win.blit(base_surf, (x0, y0))

            # --- Data Collection ---
            webcam_gaze_pt = None
            webcam_face_info = None
            # Note: sol_gaze_pt is NOT reset here - it persists like preview mode
            # CSV recording variables are reset each frame (only record new data)
            sol_mapped_gaze_pt_for_csv = None
            sol_raw_gaze_pt_for_csv = None
            sol_raw_gaze_data_for_csv = None
            sol_info = {}

            # 1. Webcam Gaze
            if gf:
                gi = gf.get_gaze_info()
                if gi and getattr(gi, 'status', False):
                    coords = getattr(gi, 'filtered_gaze_coordinates', None) or getattr(gi, 'gaze_coordinates', None)
                    if coords: webcam_gaze_pt = (int(coords[0]), int(coords[1]))

                # Get FaceInfo for landmarks and boxes (from MediaPipe)
                # We need to call face_alignment.detect() to get the FaceInfo
                if hasattr(gf, 'face_alignment') and gf.face_alignment:
                    try:
                        # Get latest frame from webcam (note: it's last_frame, not latest_frame)
                        if webcam and hasattr(webcam, 'last_frame') and webcam.last_frame is not None:
                            # Detect face info from current frame
                            # Note: frame is already in RGB format from WebCamCamera
                            webcam_face_info = gf.face_alignment.detect(int(time.time() * 1000), webcam.last_frame)
                    except Exception as e:
                        print(f"Failed to get face info: {e}")

            # 2. Sol Gaze
            if sol_connector:
                sol_debug_counters['total_frames'] += 1

                # [DEBUG] Print status every 100 frames
                if sol_debug_counters['total_frames'] % 100 == 0:
                    total = sol_debug_counters['total_frames']
                    valid = sol_debug_counters['valid_gaze']
                    with_data = sol_debug_counters['frames_with_gaze_data']
                    cached = sol_debug_counters['used_cached_gaze']
                    effective = valid + cached  # Total usable gaze data
                    print(f"[Sol Debug] Frame {total}: NewData={valid} ({valid/total*100:.1f}%), "
                          f"Cached={cached} ({cached/total*100:.1f}%), "
                          f"Effective={effective} ({effective/total*100:.1f}%)")

                # Unified Frame Retrieval (Pose + Record)
                sol_frame_numpy = get_sol_frame()

                if sol_frame_numpy is not None:
                     try:
                        # [OPT] Submit frame for background ArUco detection (non-blocking!)
                        sol_projector.submit_frame_for_pose(sol_frame_numpy)
                     except Exception as e:
                        print(f"Sol Pose Err: {e}")

                # Get Gaze
                try:
                    # [OPT] Drain queue to get latest (limit to 20 to prevent delays)
                    latest_gaze = None
                    got_new_gaze_data = False
                    if sol_gaze_queue:
                        for _ in range(20):
                            try:
                                latest_gaze = sol_gaze_queue.get_nowait()
                                got_new_gaze_data = True
                            except queue.Empty:
                                break

                    if not got_new_gaze_data:
                        sol_debug_counters['gaze_queue_empty'] += 1
                    else:
                        sol_debug_counters['frames_with_gaze_data'] += 1

                    if latest_gaze:
                        sol_gaze_method = cfg.get('sol_gaze_method', '3D')
                        is_ready = sol_projector.is_homography_valid() if sol_gaze_method == '2D' else sol_projector.is_calibrated()
                        if not is_ready:
                            sol_debug_counters['not_calibrated'] += 1

                    # Get gaze method from config
                    sol_gaze_method = cfg.get('sol_gaze_method', '3D')

                    # Check if we can process gaze (depends on method)
                    can_process_gaze = False
                    if sol_gaze_method == '2D':
                        can_process_gaze = latest_gaze and sol_projector.is_homography_valid()
                    else:  # 3D method
                        can_process_gaze = latest_gaze and sol_projector.is_calibrated()

                    if can_process_gaze:
                         try:
                             # Check if gaze data has required structure
                             if not hasattr(latest_gaze, 'left_eye') or not hasattr(latest_gaze, 'right_eye') or not hasattr(latest_gaze, 'combined'):
                                 sol_debug_counters['gaze_data_structure_error'] += 1
                                 raise AttributeError("Missing left_eye, right_eye, or combined")

                             # Extract raw SDK gaze_2d for both methods
                             raw_gaze_2d = None
                             if hasattr(latest_gaze.combined, 'gaze_2d'):
                                 g2d = latest_gaze.combined.gaze_2d
                                 raw_gaze_2d = (g2d.x, g2d.y)

                             if sol_gaze_method == '2D':
                                 # 2D Gaze Mapping: Use gaze_2d and homography
                                 if raw_gaze_2d:
                                     screen_pt = sol_projector.project_gaze_2d_to_screen(raw_gaze_2d)
                                     if screen_pt:
                                         sol_gaze_pt = screen_pt
                                         sol_debug_counters['valid_gaze'] += 1

                                         # Save for CSV recording
                                         sol_mapped_gaze_pt_for_csv = sol_gaze_pt
                                         sol_raw_gaze_pt_for_csv = raw_gaze_2d
                                         sol_raw_gaze_data_for_csv = latest_gaze
                                     else:
                                         sol_debug_counters['projection_failed'] += 1
                             else:
                                 # 3D Gaze Mapping: Use gaze_3d and ray-plane intersection
                                 left_o = latest_gaze.left_eye.gaze.origin
                                 right_o = latest_gaze.right_eye.gaze.origin
                                 gaze_origin_mm = np.array([
                                     (left_o.x + right_o.x)/2.0,
                                     (left_o.y + right_o.y)/2.0,
                                     (left_o.z + right_o.z)/2.0
                                 ])

                                 # Get 3D Gaze Point
                                 g3d = latest_gaze.combined.gaze_3d
                                 gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])

                                 # Compute Direction Vector
                                 gaze_direction_vec = gaze_point_mm - gaze_origin_mm
                                 norm = np.linalg.norm(gaze_direction_vec)

                                 if norm > 0:
                                     gaze_direction_unit = gaze_direction_vec / norm

                                     # Apply Sol offset correction if available
                                     if sol_offset is not None:
                                         gaze_direction_unit = apply_angular_offset(
                                             gaze_direction_unit,
                                             sol_offset['pitch_offset_rad'],
                                             sol_offset['yaw_offset_rad']
                                         )

                                     gaze_origin_m = gaze_origin_mm / 1000.0  # Convert to meters

                                     # Projection logic (ArUco marker-based)
                                     screen_pt_m = sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
                                     if screen_pt_m is not None:
                                          pix = sol_projector.physical_to_pixels(screen_pt_m, W, physical_width_m)
                                          if pix:
                                              sol_gaze_pt = (int(pix[0]), int(pix[1]))
                                              sol_debug_counters['valid_gaze'] += 1

                                              # Save for CSV recording
                                              sol_mapped_gaze_pt_for_csv = sol_gaze_pt
                                              sol_raw_gaze_pt_for_csv = raw_gaze_2d
                                              sol_raw_gaze_data_for_csv = latest_gaze
                                          else:
                                              sol_debug_counters['projection_failed'] += 1
                                     else:
                                         sol_debug_counters['projection_failed'] += 1
                         except AttributeError as e:
                             sol_debug_counters['attribute_error'] += 1
                             # print(f"[DEBUG] AttributeError in gaze processing: {e}")
                except Exception as e:
                    print(f"[DEBUG] Exception in Sol gaze processing: {e}")

                # [FIX] Match preview behavior: sol_gaze_pt persists between frames
                # No explicit caching needed - the variable itself maintains the last valid gaze
                # Just count frames where we're using the persisted value
                if not got_new_gaze_data and sol_gaze_pt is not None:
                    sol_debug_counters['used_cached_gaze'] += 1

            # --- Evaluation Logic ---
            eval_pt = None
            if cfg['eval_source'] == "Webcam": eval_pt = webcam_gaze_pt
            else: eval_pt = sol_gaze_pt

            # Validation Area Logic
            valid_sample = False
            in_correct_half = False
            
            if eval_pt:
                gx, gy = eval_pt
                # Check Bounds
                if 0 <= gx < W and 0 <= gy < H:
                    valid_sample = True
                    # Check Correct Half
                    if side == 'right': in_correct_half = (gx >= W // 2)
                    else: in_correct_half = (gx < W // 2)
                    
                    # Draw Marker (only if within screen bounds)
                    if cfg.get('show_gaze_marker', True) and 0 <= gx < W and 0 <= gy < H:
                         pygame.draw.circle(win, to_rgb_tuple(cfg['gaze_marker_color']), (gx, gy), cfg['gaze_marker_radius'], cfg['gaze_marker_width'])

            # Pass/Fail Logic
            if valid_sample and in_correct_half:
                if hold_start is None: hold_start = time.time()
                if time.time() - hold_start >= cfg['pass_dur']:
                    passed = True
                    break
            else:
                hold_start = None

            # Render & Record
            # Status Bar
            msg_txt = f"{cpd:.2f} cpd  ({cs:.1f} cyc/screen)  t={t:.1f}s"
            if hold_start:
                msg_txt += f"  hold={time.time() - hold_start:.1f}/{cfg['pass_dur']:.1f}s"
            txt = status_font.render(msg_txt, True, (255, 255, 255))
            win.blit(txt, (10, 10))

            pygame.display.flip()
            
            # Recording
            # Extract Face Mesh info from MediaPipe FaceInfo
            lms_str = ""
            face_box = ""
            left_eye_box = ""
            right_eye_box = ""

            if webcam_face_info and getattr(webcam_face_info, 'status', False):
                try:
                    # Face landmarks - format as semicolon-separated x,y,z tuples
                    if hasattr(webcam_face_info, 'face_landmarks') and webcam_face_info.face_landmarks is not None:
                        # face_landmarks is a numpy array of shape (478, 3)
                        landmarks = webcam_face_info.face_landmarks
                        lms_parts = []
                        for i in range(len(landmarks)):
                            lms_parts.append(f"{landmarks[i][0]:.4f},{landmarks[i][1]:.4f},{landmarks[i][2]:.4f}")
                        lms_str = ";".join(lms_parts) if lms_parts else ""

                    # Face box [x, y, w, h]
                    if hasattr(webcam_face_info, 'face_rect') and webcam_face_info.face_rect is not None:
                        face_box = str([int(v) for v in webcam_face_info.face_rect])

                    # Left eye box [x, y, w, h]
                    if hasattr(webcam_face_info, 'left_rect') and webcam_face_info.left_rect is not None:
                        left_eye_box = str([int(v) for v in webcam_face_info.left_rect])

                    # Right eye box [x, y, w, h]
                    if hasattr(webcam_face_info, 'right_rect') and webcam_face_info.right_rect is not None:
                        right_eye_box = str([int(v) for v in webcam_face_info.right_rect])
                except Exception as e:
                    print(f"Error extracting face info: {e}")

            # Recording Logic
            rec_screen = cfg.get('rec_webcam') or cfg.get('rec_sol_data')

            # [FIX] Throttle screen recording to prevent queue overflow
            screen_record_frame_counter += 1
            should_record_screen = (screen_record_frame_counter % SCREEN_RECORD_EVERY_N_FRAMES == 0)

            # Fetch Frames
            sol_f = sol_frame_numpy if cfg.get('rec_sol_raw_video') else None
            wb_f = get_webcam_frame() if cfg.get('rec_webcam') else None

            recorder.process_and_record(
                wb_f,
                win if (rec_screen and should_record_screen) else None,
                stim_pos=(x0, y0),
                webcam_gaze=webcam_gaze_pt,
                sol_mapped_gaze=sol_mapped_gaze_pt_for_csv if cfg.get('rec_sol_data') else None,
                sol_raw_gaze=sol_raw_gaze_pt_for_csv if cfg.get('rec_sol_data') else None,
                sol_raw_gaze_data=sol_raw_gaze_data_for_csv if cfg.get('rec_sol_data') else None,
                sol_frame=sol_f,
                target_letter=f"{cpd:.1f}",
                is_correct=passed,
                landmarks_str=lms_str,
                face_box=face_box,
                left_eye_box=left_eye_box,
                right_eye_box=right_eye_box
            )
            
            clock.tick(60)
            if t > cfg['stim_dur'] and hold_start is None: break
          except Exception as frame_err:
            # [FIX] Log frame-level errors but continue
            import traceback
            print(f"[Trial {trial_number}] Frame error: {frame_err}")
            traceback.print_exc()
            # Continue to next frame rather than crashing

        # Pass/Fail Feedback
        # Seed display gaze with trial's last position to prevent flash of incorrect gaze
        _display_gaze[0] = sol_gaze_pt
        fb_text = "PASS" if passed else "FAIL"
        fb_color = (0, 255, 0) if passed else (255, 0, 0)
        fb_surf = feedback_font.render(fb_text, True, fb_color)
        fb_bg = PAPER_GRAY if PAPER_COLOR_ENABLED else to_rgb_tuple(cfg['bg_color'])
        fb_start = time.time()
        while time.time() - fb_start < 1.0:
            pygame.event.pump()
            win.fill(fb_bg)
            # Draw ArUco markers
            if cfg['enable_sol']:
                for mid, pos in aruco_markers_px.items():
                    if mid in aruco_imgs:
                        cv_img = aruco_imgs[mid]
                        if len(cv_img.shape) == 2: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                        else: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        pimg = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                        win.blit(pimg, (pos[0], pos[1]))
            # Draw feedback text
            win.blit(fb_surf, ((W - fb_surf.get_width()) // 2, (H - fb_surf.get_height()) // 2))
            # Process and draw gaze marker
            process_and_draw_gaze(win)
            pygame.display.flip()
            pump_recorder()
            clock.tick(30)

        # Feedback
        stair.update(passed)
        results.append({'cpd': cpd, 'res': "PASS" if passed else "FAIL"})
        
        # Interval
        # Interval
        show_interval_center(cfg['inter_interval_img_dur'])
        show_background_blank(cfg.get('bg_after_inter_dur', 1.0))

        print(f"[Trial] Completed trial {trial_number}")

      except Exception as trial_err:
        # [FIX] Log trial-level errors with full traceback
        import traceback
        err_msg = traceback.format_exc()
        print(f"\n[TRIAL ERROR] Trial {trial_number} crashed!")
        print(f"Error: {trial_err}")
        print(f"Full traceback:\n{err_msg}")

        # Write to crash log file
        try:
            with open("va_trial_crash_log.txt", "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Trial {trial_number} crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(err_msg)
                f.write("\n")
        except:
            pass

        # Try to continue with next trial
        stair.update(False)  # Mark as failed
        results.append({'cpd': float(stair.freq), 'res': "ERROR"})
        print(f"[TRIAL ERROR] Attempting to continue to next trial...")

    # Stop background ArUco detection (per-test resource), but keep Sol connection alive
    # so it can be reused when returning to settings.
    if sol_projector:
        sol_projector.stop_background_detection()

    # [DEBUG] Print Sol gaze statistics
    if cfg['enable_sol'] and sol_debug_counters['total_frames'] > 0:
        print("\n" + "="*60)
        print("SOL GAZE PROCESSING STATISTICS")
        print("="*60)
        for key, value in sol_debug_counters.items():
            pct = (value / sol_debug_counters['total_frames'] * 100) if key != 'total_frames' else 100
            print(f"{key:20s}: {value:6d} ({pct:5.1f}%)")
        print("="*60 + "\n")

    # Final Result & CSV
    final_cpd = float(stair.freq)
    va_score  = get_va_result_cpd(final_cpd)
    
    # Save Summary CSV
    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(f"VA_output/VA_{cfg['user_name']}_opt.csv", index=False, encoding='utf-8-sig')
        print(f"Summary saved to VA_output/VA_{cfg['user_name']}_opt.csv")
    except Exception as e:
        print(f"Failed to save summary CSV: {e}")

    # Result Screen
    result_font = pygame.font.SysFont(None, 80)
    info_font   = pygame.font.SysFont(None, 40)

    win.fill(to_rgb_tuple(cfg['bg_color']))
    text1 = result_font.render(f"Final Spatial Freq: {final_cpd:.2f} cpd", True, (255, 255, 255))
    text2 = result_font.render(f"Estimated VA Score: {va_score}", True, (0, 255, 255))
    text3 = info_font.render("Press Q to Exit", True, (200, 200, 200))

    win.blit(text1, ((W - text1.get_width()) // 2, H // 3 - 50))
    win.blit(text2, ((W - text2.get_width()) // 2, H // 3 + 50))
    win.blit(text3, ((W - text3.get_width()) // 2, H // 3 + 150))
    pygame.display.flip()
    
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: break
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q: break
        else:
            time.sleep(0.05)
            continue
        break

    # End - cleanup and return to main loop (keep Sol connection alive for reuse)
    if gf: gf.stop_sampling(); gf.release()
    if sol_projector: sol_projector.stop_background_detection()
    recorder.close()
    pygame.quit()

if __name__ == '__main__':
    # [FIX] Enable faulthandler to get stack traces on segfaults/native crashes
    import faulthandler
    faulthandler.enable()

    # [FIX] DPI Awareness
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except: pass

    # [FIX] Init GazeFollower Logger
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        log_file = log_dir / f"gazefollower_{_time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))
    except Exception as e:
        print(f"Logger init failed: {e}")

    # Main loop - SettingsWindow stays alive across test runs (like preview pattern)
    # Sol connection is preserved because the window is only hidden, not destroyed.
    # Only exits when user closes the settings window without starting.
    s = SettingsWindow()
    while True:
        s.mainloop()
        if s.cfg:
            # Build Sol context from the (still alive) Settings window
            sol_ctx = None
            if s.active_sol_connector:
                sol_ctx = {
                    'connector': s.active_sol_connector,
                    'gaze_queue': s.sol_gaze_queue,
                    'scene_queue': s.sol_scene_queue,
                    'cam_params': s.sol_cam_params,
                }

            try:
                run_test(s.cfg, sol_ctx)
            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                print(f"CRITICAL ERROR IN RUN_TEST:\n{err_msg}")
                with open("va_crash_log.txt", "w") as f:
                    f.write(err_msg)
                try:
                    messagebox.showerror("Crash", f"An error occurred:\n{e}\nSee va_crash_log.txt")
                except Exception:
                    pass  # messagebox may fail if tkinter is not available
                try:
                    pygame.quit()
                except Exception:
                    pass

            # Show settings window again and restart its timers
            mode = "Practice" if s.cfg.get('practice_mode', False) else "Test"
            print(f"[{mode}] Returning to settings...")
            s.cfg = None  # Reset for next iteration
            s.deiconify()
            # Pause scene stream to avoid Sol SDK native crash during idle
            if s.active_sol_connector:
                s.active_sol_connector.pause_scene_stream()
            # Restart flush timer if Sol is still connected
            if s.is_sol_connected:
                s.flush_sol_queues()
            # Restart auto-save traces (they survive since window is alive)
            continue
        else:
            # User closed settings without starting
            if s.active_sol_connector:
                try:
                    s.active_sol_connector.stop()
                except Exception:
                    pass
            break
