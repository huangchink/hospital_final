# -*- coding: utf-8 -*-
"""
Integrated VA test: PyQt5 settings + gazefollower calibration/sampling
+ dynamic staircase-controlled number of trials + VA result mapping
"""
import sys, os, time, random, logging
from collections import deque

import cv2
import numpy as np
import pygame
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QVBoxLayout, QWidget, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt

# ensure submodules use logging
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig

def get_va_result(frequency):
    if 800 <= frequency <= 900:
        return 0.7
    elif 700 <= frequency < 800:
        return 0.6
    elif 600 <= frequency < 700:
        return 0.5
    elif 500 <= frequency < 600:
        return 0.3
    elif 400 <= frequency < 500:
        return 0.2
    elif 300 <= frequency < 400:
        return 0.1
    elif 200 <= frequency < 300:
        return 0.05
    elif 100 <= frequency < 200:
        return 0.01
    return "Unknown"

def generate_grating(freq, w, h):
    x = np.arange(w)
    xx, _ = np.meshgrid(x, np.arange(h))
    g = 0.5 + 0.5 * np.sin(2 * np.pi * freq * xx / w)
    img = (g * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (5,5), sigmaX=1.0)

def create_circle_mask(radius, center, W, H):
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m, center, radius, 255, -1)
    return m

class Staircase:
    def __init__(self, start, step, minv, maxv):
        self.freq = start
        self.step = step
        self.minv, self.maxv = minv, maxv
        self.reversals = []
        self.last_correct = None
        self.correct_streak = 0
        self.max_correct_streak = 0

    def update(self, correct):
        if self.last_correct is not None and correct != self.last_correct:
            self.reversals.append(self.freq)
        self.last_correct = correct

        if correct:
            self.correct_streak += 1
            self.max_correct_streak = max(self.max_correct_streak, self.correct_streak)
        else:
            self.correct_streak = 0

        delta = self.step if correct else -self.step
        self.freq = min(self.maxv, max(self.minv, self.freq + delta))

    def done(self):
        return (len(self.reversals) >= 8) or (self.freq >= self.maxv and self.max_correct_streak >= 3)

class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VA Test Settings")
        self.setFixedSize(400,400)
        layout = QVBoxLayout()
        def add_label(t): layout.addWidget(QLabel(t))
        add_label("Calibration Points:")
        self.calib = QComboBox(); self.calib.addItems(["5","9","13"]); layout.addWidget(self.calib)
        add_label("Stimulus Duration (s):")
        self.stim_dur = QDoubleSpinBox(); self.stim_dur.setValue(2.0); layout.addWidget(self.stim_dur)
        add_label("Blank Duration (s):")
        self.blank_dur = QDoubleSpinBox(); self.blank_dur.setValue(1.0); layout.addWidget(self.blank_dur)
        add_label("Circle Radius (px):")
        self.rad = QSpinBox(); self.rad.setValue(250); layout.addWidget(self.rad)
        add_label("Threshold (px):")
        self.thresh = QSpinBox(); self.thresh.setValue(100); layout.addWidget(self.thresh)
        self.sound = QCheckBox("Enable Sound"); self.sound.setChecked(True); layout.addWidget(self.sound)
        btn = QPushButton("Start"); btn.clicked.connect(self.on_start); layout.addWidget(btn)
        w=QWidget(); w.setLayout(layout); self.setCentralWidget(w)

    def on_start(self):
        self.cfg = {
            'calib_pts': int(self.calib.currentText()),
            'stim_dur': self.stim_dur.value(),
            'blank_dur': self.blank_dur.value(),
            'rad': self.rad.value(),
            'thresh': self.thresh.value(),
            'sound': self.sound.isChecked()
        }
        self.close()

def run_test(cfg):
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    win = pygame.display.set_mode((W,H), pygame.FULLSCREEN)

    # initialize gaze follower
    dcfg = DefaultConfig(); dcfg.cali_mode = cfg['calib_pts']
    gf = GazeFollower(config=dcfg)
    gf.preview(win=win)
    gf.calibrate(win=win)
    gf.start_sampling()
    time.sleep(0.1)

    stair = Staircase(start=100, step=50, minv=10, maxv=900)
    centers = {'left':(W//4,H//2), 'right':(3*W//4,H//2)}
    clock = pygame.time.Clock()
    results = []

    while not stair.done():
        side = random.choice(['left','right'])
        freq = stair.freq
        gr = generate_grating(freq, W, H)
        mask = create_circle_mask(cfg['rad'], centers[side], W, H)
        gaze_q = deque(maxlen=60)

        start = time.time()
        passed = False

        while True:
            # quit on Q
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    gf.stop_sampling()
                    gf.save_data('VA_raw.csv')
                    gf.release()
                    pygame.quit()
                    sys.exit()

            t = time.time() - start
            gi = gf.get_gaze_info()
            dist = float('inf')
            if gi and getattr(gi,'status',False) and gi.filtered_gaze_coordinates:
                gx, gy = map(int, gi.filtered_gaze_coordinates)
                gaze_q.append((gx,gy))
                avgx = sum(p[0] for p in gaze_q)/len(gaze_q)
                avgy = sum(p[1] for p in gaze_q)/len(gaze_q)
                dist = np.hypot(avgx - centers[side][0], avgy - centers[side][1])

            win.fill((127,127,127))
            stim = cv2.bitwise_and(gr, gr, mask=mask)
            arr = np.stack([stim]*3, axis=-1).swapaxes(0,1)
            surf = pygame.surfarray.make_surface(arr)
            win.blit(surf, (0,0))

            if gaze_q:
                pygame.draw.circle(win, (0,255,0), (int(avgx), int(avgy)), 30, 4)

            font = pygame.font.SysFont(None, 30)
            txt = font.render(f"{freq}Hz  t={t:.1f}s d={dist:.1f}", True, (255,255,255))
            win.blit(txt, (10,10))

            pygame.display.flip()
            clock.tick(60)

            if t <= cfg['stim_dur'] and gaze_q and dist <= cfg['thresh']:
                passed = True
                break
            if t > cfg['stim_dur']:
                break

        stair.update(passed)
        results.append({'freq':freq, 'side':side, 'res':'PASS' if passed else 'FAIL'})
        time.sleep(cfg['blank_dur'])

    # compute final VA and save
    final_freq = stair.freq
    va_score = get_va_result(final_freq)
    for r in results:
        r['final_freq'] = final_freq
        r['va_score'] = va_score

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('VA_summary.csv', index=False)

    gf.stop_sampling()
    # gf.save_data('VA_raw.csv')
    gf.release()
    pygame.quit()

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = SettingsWindow(); w.show()
    app.exec_()
    run_test(w.cfg)
