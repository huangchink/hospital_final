#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import sys
import threading
import time

import cv2

from .Camera import Camera  # Adjust import according to your package structure
from ..logger import Log


class WebCamCamera(Camera):
    """
    A class to manage webcam operations, inheriting from the base Camera class.
    """

    # Reconnection settings
    _MAX_CONSECUTIVE_FAILURES = 30   # Failures before attempting reconnect
    _RECONNECT_COOLDOWN_S = 2.0     # Seconds to wait before reconnecting
    _MAX_RECONNECT_ATTEMPTS = 5     # Max reconnect attempts before giving up

    def __init__(self, webcam_id=0, img_height=480, img_width=640, cam_fps=30):
        """
        Initializes the WebCamCamera object, sets up the camera properties,
        creates the capture thread, and ensures the save directory exists.
        """
        super().__init__()
        self._camera_thread_running = None
        self._camera_thread = None
        self.webcam_id = webcam_id
        self.img_height = img_height
        self.img_width = img_width
        self.cam_fps = cam_fps
        self.last_frame = None
        self._cap = cv2.VideoCapture()
        # Set the camera resolution and frame rate.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        # Reconnection state
        self._consecutive_failures = 0
        self._reconnect_attempts = 0

    def _create_capture_thread(self):
        """
        Creates and starts a daemon thread for continuously capturing frames from the camera.
        """
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self.capture)
        self._camera_thread.daemon = True
        self._camera_thread.start()

    def _try_reopen(self):
        """Attempt to re-open the camera after disconnect. Returns True on success."""
        if self._reconnect_attempts >= self._MAX_RECONNECT_ATTEMPTS:
            return False
        self._reconnect_attempts += 1
        Log.w(f"Attempting webcam reconnect ({self._reconnect_attempts}/{self._MAX_RECONNECT_ATTEMPTS})...")
        try:
            if self._cap.isOpened():
                self._cap.release()
            time.sleep(self._RECONNECT_COOLDOWN_S)
            if sys.platform == 'win32':
                opened = self._cap.open(self.webcam_id, cv2.CAP_DSHOW)
            else:
                opened = self._cap.open(self.webcam_id)
            if opened:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
                self._cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
                self._consecutive_failures = 0
                self._reconnect_attempts = 0
                Log.i("WebCam reconnected successfully")
                return True
            else:
                Log.w("WebCam reconnect failed - camera not available")
                return False
        except Exception as e:
            Log.w(f"WebCam reconnect error: {e}")
            return False

    def capture(self):
        """
        Continuously captures frames from the webcam while the camera is in specific running states.
        If a callback is set, it executes the callback function with the current frame.
        Automatically attempts reconnection if camera disconnects.
        """
        while self._camera_thread_running:
            # Capture a frame from the webcam.
            try:
                ret, frame = self._cap.read()
            except Exception as e:
                Log.w(f"Camera read exception: {e}")
                ret = False
                frame = None

            # Capture the current timestamp.
            timestamp = time.time_ns()
            if not ret:
                self._consecutive_failures += 1
                if self._consecutive_failures == 1:
                    Log.w("Failed to grab frame")
                # Attempt reconnect after sustained failures
                if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    Log.w(f"Camera disconnected ({self._consecutive_failures} consecutive failures)")
                    if not self._try_reopen():
                        # Wait before next retry cycle
                        time.sleep(self._RECONNECT_COOLDOWN_S)
                    self._consecutive_failures = 0
                else:
                    time.sleep(0.01)  # Brief sleep to avoid busy-spin
                continue

            # Reset failure counter on success
            self._consecutive_failures = 0
            self._reconnect_attempts = 0

            # Check if the frame is in BGR format (default for OpenCV) and convert to RGB if necessary
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to target dimensions if necessary
            frame = cv2.resize(frame, (self.img_width, self.img_height))
            self.last_frame = frame

            # Lock and execute callback function if set.
            try:
                with self.callback_and_param_lock:
                    if self.callback_and_params:
                        func, args, kwargs = self.callback_and_params
                        func(self.camera_running_state, timestamp, frame, *args, **kwargs)
            except Exception as e:
                # Log callback errors but do NOT exit - let the capture loop continue
                Log.e(f"Camera callback error: {e}")

    def open(self):
        """
        Opens the webcam if it is not already opened.
        Uses DirectShow backend on Windows for faster initialization.
        """
        if sys.platform == 'win32':
            # DirectShow is significantly faster than MSMF on Windows
            if not self._cap.open(self.webcam_id, cv2.CAP_DSHOW):
                Log.e("Failed to open webcam camera (DirectShow)")
                raise Exception("Failed to open webcam camera")
        else:
            if not self._cap.open(self.webcam_id):
                Log.e("Failed to open webcam camera")
                raise Exception("Failed to open webcam camera")
        self._consecutive_failures = 0
        self._reconnect_attempts = 0
        Log.i("WebCam opened")
        self._create_capture_thread()

    def close(self):
        """
        Releases the webcam resources if the camera is currently opened.
        """
        Log.i("WebCam closed")
        if self._camera_thread is not None:
            self._camera_thread_running = False
            self._camera_thread.join(timeout=3.0)
        if self._cap.isOpened():
            self._cap.release()

    def set_on_image_callback(self, func, args=(), kwargs=None):
        """
        Sets a callback function to be called with each captured frame.
        """
        super().set_on_image_callback(func, args, kwargs)

    def release(self):
        self._camera_thread_running = False
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
        self.close()
