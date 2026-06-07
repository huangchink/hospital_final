"""Play/pause + speed transport controls (from replayer.py)."""
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
from ntuh.replayer.playback_engine import PlaybackEngine

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
