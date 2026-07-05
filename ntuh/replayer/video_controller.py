"""Timestamp-based per-stream video frame seeking (extracted from replayer.py)."""
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
