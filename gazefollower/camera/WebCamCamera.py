#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import threading
import time

import cv2

from .Camera import Camera  # Adjust import according to your package structure
from ..misc import CameraRunningState


class WebCamCamera(Camera):
    """
    A class to manage webcam operations, inheriting from the base Camera class.
    """

    def __init__(self, webcam_id=0):
        """
        Initializes the WebCamCamera object, sets up the camera properties,
        creates the capture thread, and ensures the save directory exists.

        Attributes:
        ----------
        webcam_id : int
            Which webcam camera is connected.
        cap: cv2.VideoCapture
            The instance of cv2.VideoCapture and it can be None.
        """
        super().__init__()
        self._camera_thread_running = None
        self._camera_thread = None
        self.webcam_id = webcam_id
        self._cap = None

    def _create_capture_thread(self):
        """
        Creates and starts a daemon thread for continuously capturing frames from the camera.
        """
        self._cap = cv2.VideoCapture(self.webcam_id)
        # Set the camera resolution and frame rate.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        # self.cap = cv2.VideoCapture(1)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self.capture)
        self._camera_thread.daemon = True
        self._camera_thread.start()

    def capture(self):
        """
        Continuously captures frames from the webcam while the camera is in specific running states.
        If a callback is set, it executes the callback function with the current frame.
        """

        while self._camera_thread_running:

            # Capture a frame from the webcam.
            ret, frame = self._cap.read()

            # Capture the current timestamp.
            timestamp = time.time_ns()
            if not ret:
                print("Failed to grab frame")
                continue

            # Check if the frame is in BGR format (default for OpenCV) and convert to RGB if necessary
            # Preprocessing image data
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to 640x480 if necessary
            # if frame.shape[0] != 480 or frame.shape[1] != 640:
            #     frame = cv2.resize(frame, (640, 480))

            # Lock and execute callback function if set.
            with self.callback_and_param_lock:
                if self.callback_and_params:
                    func, args, kwargs = self.callback_and_params
                    func(self.camera_running_state, timestamp, frame, *args, **kwargs)
        else:
            self.close()

    def open(self):
        """
        Opens the webcam if it is not already opened.
        """
        if self._camera_thread is None:
            self._camera_thread_running = True
            self._create_capture_thread()
        else:
            print("WebCam already opened")

    def close(self):
        """
        Releases the webcam resources if the camera is currently opened.
        """
        if self._cap.isOpened():
            self._cap.release()

    def set_on_image_callback(self, func, args=(), kwargs=None):
        """
        Sets a callback function to be called with each captured frame.
        The callback function must have the following args,
            timestamp and frame, which are the timestamp when the image was
            captured and the captured image frame (np.ndarray).

        Parameters:
        - func: The callback function to handle the image frame.
        - args: Tuple of arguments to pass to the callback function.
        - kwargs: Dictionary of keyword arguments to pass to the callback function.
        """
        super().set_on_image_callback(func, args, kwargs)

    def release(self):
        self._camera_thread_running = False
        self._camera_thread.join()
        self.close()
