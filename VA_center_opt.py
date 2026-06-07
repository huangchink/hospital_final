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
import csv
import traceback
import ctypes
from collections import deque
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from pathlib import Path

# Base directory for writable user data (settings, calibration profiles, output, logs).
# When frozen by PyInstaller, __file__ points inside the bundle, so anchor to the .exe
# directory instead; otherwise use the script's own directory (unchanged dev behaviour).
from ntuh.common.app_env import APP_DIR, LAST_SETTINGS_FILE


# [FIX] Windows keyboard layout management - switch to English to ensure
# keystroke-based controls (q, SPACE, etc.) work regardless of IME state.
# Persists original layout to file for crash recovery.
class KeyboardLayoutManager:
    """Cache current keyboard layout, switch to English, restore on exit.
    Saves original layout to a file so it can be recovered after a native crash."""
    EN_US_LAYOUT = 0x0409
    _RECOVERY_FILE = APP_DIR / ".kb_layout_backup"

    def __init__(self):
        self._original_layout = None
        self._switched = False
        # On startup, check if a previous crash left the keyboard in English
        self._recover_from_crash()

    def _recover_from_crash(self):
        """If a previous run crashed without restoring, restore now."""
        try:
            if self._RECOVERY_FILE.exists():
                layout_hex = self._RECOVERY_FILE.read_text().strip()
                if layout_hex:
                    layout = int(layout_hex, 16)
                    user32 = ctypes.windll.user32
                    current_lang = user32.GetKeyboardLayout(0) & 0xFFFF
                    if current_lang == self.EN_US_LAYOUT and (layout & 0xFFFF) != self.EN_US_LAYOUT:
                        # Previous crash left us in English, restore
                        user32.ActivateKeyboardLayout(layout, 0)
                        user32.PostMessageW(0xFFFF, 0x0050, 0, layout)
                        print(f"[Keyboard] Recovered from previous crash - restored layout 0x{layout:08X}")
                self._RECOVERY_FILE.unlink(missing_ok=True)
        except Exception as e:
            print(f"[Keyboard] Crash recovery check failed: {e}")

    def switch_to_english(self):
        """Cache current layout and switch to English (US)."""
        try:
            user32 = ctypes.windll.user32
            self._original_layout = user32.GetKeyboardLayout(0)
            current_lang = self._original_layout & 0xFFFF
            if current_lang != self.EN_US_LAYOUT:
                # Save original layout to file for crash recovery
                try:
                    self._RECOVERY_FILE.write_text(f"0x{self._original_layout:08X}")
                except Exception:
                    pass
                # Load and activate English (US) layout
                hkl = user32.LoadKeyboardLayoutW(f"{self.EN_US_LAYOUT:08X}", 0x01)  # KLF_ACTIVATE
                if hkl:
                    user32.PostMessageW(0xFFFF, 0x0050, 0, hkl)
                    self._switched = True
                    print(f"[Keyboard] Switched to English (US) from layout 0x{self._original_layout:08X}")
                else:
                    print("[Keyboard] Failed to load English layout")
                    self._RECOVERY_FILE.unlink(missing_ok=True)
            else:
                print("[Keyboard] Already using English layout")
        except Exception as e:
            print(f"[Keyboard] Error switching layout: {e}")

    def restore(self):
        """Restore the original keyboard layout."""
        if not self._switched or self._original_layout is None:
            return
        try:
            user32 = ctypes.windll.user32
            user32.ActivateKeyboardLayout(self._original_layout, 0)
            user32.PostMessageW(0xFFFF, 0x0050, 0, self._original_layout)
            print(f"[Keyboard] Restored original layout 0x{self._original_layout:08X}")
            self._switched = False
        except Exception as e:
            print(f"[Keyboard] Error restoring layout: {e}")
        finally:
            # Remove recovery file - we restored successfully
            try:
                self._RECOVERY_FILE.unlink(missing_ok=True)
            except Exception:
                pass

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


# DummyRecorder moved to recorder.py (alongside Recorder)
# [NEW] Imports for Webcam Preview
try:
    from PIL import Image, ImageTk
    import mediapipe as mp
except ImportError:
    print("Warning: PIL or mediapipe not found. Webcam Preview might fail.")

# LAST_SETTINGS_FILE now imported from ntuh.common.app_env (top of file).

# [NEW] Global Crash Handler
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

from ntuh.vavf.stimuli import (
    get_va_result_cpd, DO_BLUR, BLUR_KSIZE, BLUR_SIGMA,
    VF_ANGULAR_DIAMETERS, VF_PASS_DWELL_SEC, VF_TIMEOUT_SEC,
    vf_angular_to_pixel_diameter, vf_get_quadrant, vf_generate_points,
    vf_convert_positions_to_pixels, prepare_patch_grid,
    generate_grating_oriented_patch, Staircase,
)

# Monitor enumeration + tester-rect moved to ntuh.common.win_monitors.
from ntuh.common.win_monitors import get_monitor_info_windows, resolve_tester_rect


# Pygame input/focus helpers moved to ntuh.common.pygame_utils.
from ntuh.common.pygame_utils import restore_event_filter, ensure_pygame_focus

from ntuh.common.optics import (
    to_rgb_tuple, screen_width_deg_from_cm, px_to_cm, cm_to_px, mean_color_rgb,
)

# VF geometry, VA gratings and the Staircase moved to ntuh.vavf.stimuli (imported above).

# Sol/webcam gaze-quality accounting + DashboardState moved to ntuh.tracking.sol_quality.
from ntuh.tracking.sol_quality import (
    sol_sample_is_valid, _sol_eye_valid, _ValidityCounter, SolQualityTracker,
    build_quality_summary, build_summary_lines, DashboardState,
)


# Webcam face-quality overlay moved to ntuh.ui.face_overlay (shared by preview + dashboard).
from ntuh.ui.face_overlay import (
    draw_face_quality_overlay, guide_kwargs_from_cfg, _quality_color, _put_text,
    _Q_GREEN, _Q_RED, _Q_YELLOW, _Q_GRAY, _Q_WHITE,
    _GUIDE_OVAL_SIZE_FRAC, _GUIDE_OVAL_BOTTOM_X_FRAC, _GUIDE_OVAL_BOTTOM_Y_FRAC,
)


# Tester dashboard moved to ntuh.ui.tester_dashboard.
from ntuh.ui.tester_dashboard import TesterDashboard


# resolve_tester_rect moved to ntuh.common.win_monitors (imported above).


# Webcam-preview Camera + ScrollableFrame moved to ntuh.ui.widgets.
from ntuh.ui.widgets import Camera, ScrollableFrame


# SettingsWindow moved to ntuh.ui.settings_window.
from ntuh.ui.settings_window import SettingsWindow

# ---------- Sol Thread Helper ----------
# run_sol_worker moved to ntuh.tracking.sol_session
from ntuh.tracking.sol_session import run_sol_worker
# ---------- VF Experiment ----------
# run_vf_test moved to ntuh.flows.vf_test
from ntuh.flows.vf_test import run_vf_test
# ---------- VA Experiment ----------
# run_test moved to ntuh.flows.va_test
from ntuh.flows.va_test import run_test
if __name__ == '__main__':
    # [FIX] Enable faulthandler to get stack traces on segfaults/native crashes
    import faulthandler
    faulthandler.enable()

    # [FIX] DPI Awareness
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except: pass

    # [FIX] Switch keyboard to English so keystroke controls (q, SPACE, etc.) work
    kb_manager = KeyboardLayoutManager()
    kb_manager.switch_to_english()
    # Register atexit to guarantee restore even on crashes or sys.exit()
    import atexit
    atexit.register(kb_manager.restore)

    # [FIX] Init GazeFollower Logger
    try:
        log_dir = APP_DIR / "logs"
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
                    'cached_homography': s.sol_cached_homography,
                }

            try:
                if s.cfg.get('experiment_type') == 'VF':
                    run_vf_test(s.cfg, sol_ctx)
                else:
                    run_test(s.cfg, sol_ctx)
            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                exp_type = s.cfg.get('experiment_type', 'VA')
                print(f"CRITICAL ERROR IN {exp_type} TEST:\n{err_msg}")
                with open(f"{exp_type.lower()}_crash_log.txt", "w") as f:
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

    # [FIX] Restore original keyboard layout on exit
    kb_manager.restore()
