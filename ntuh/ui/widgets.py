"""Reusable tk / OpenCV helpers: a threaded webcam capture and a scrollable tk frame.
Extracted verbatim from VA_center_opt.py (no behaviour change).
"""
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2


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

        # Bind mousewheel on canvas itself
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        # Make the inner frame width match the canvas width
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def bind_mousewheel_recursive(self):
        """Call after all child widgets are added to enable mousewheel on all descendants."""
        self._bind_wheel(self.scrollable_frame)

    def _bind_wheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)
