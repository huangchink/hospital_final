# ----------------------------------------------------------------
# Patch: ensure submodules use our logging
# ----------------------------------------------------------------
import logging
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging

# ----------------------------------------------------------------
# Official imports
# ----------------------------------------------------------------
import cv2
import pygame
import numpy as np
import pandas as pd
import math
import time
import datetime
import os
import sys
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig

# ----------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("svop_debug.log"),
        logging.StreamHandler()
    ]
)

# ----------------------------------------------------------------
# Default constants
# ----------------------------------------------------------------
ANGULAR_DIAMETERS = {"Goldmann IV": 0.86}
BACKGROUND_COLOR   = (0,   0,   0)
PASS_COLOR         = (0, 255,   0)
ERROR_COLOR        = (255, 0,   0)

# ----------------------------------------------------------------
# 1. Configuration GUI
# ----------------------------------------------------------------
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SVOP Test Configuration")
        self.root.geometry("400x350")

        # Calibration point count
        tk.Label(self.root, text="Calibration Points:").pack(pady=5)
        self.calib_points = tk.IntVar(value=5)
        ttk.Combobox(
            self.root,
            textvariable=self.calib_points,
            values=[5, 9, 13],
            state="readonly"
        ).pack()

        # Stimulus point count
        tk.Label(self.root, text="Stimulus Points:").pack(pady=5)
        self.stim_points = tk.IntVar(value=9)
        ttk.Combobox(
            self.root,
            textvariable=self.stim_points,
            values=[5, 9, 13],
            state="readonly"
        ).pack()

        # Screen width
        tk.Label(self.root, text="Screen Width (cm):").pack(pady=5)
        self.screen_width_cm = tk.DoubleVar(value=52.704)
        tk.Entry(self.root, textvariable=self.screen_width_cm).pack()

        # Viewing distance
        tk.Label(self.root, text="Viewing Distance (cm):").pack(pady=5)
        self.viewing_distance_cm = tk.DoubleVar(value=45.0)
        tk.Entry(self.root, textvariable=self.viewing_distance_cm).pack()

        # Pass threshold
        tk.Label(self.root, text="Pass Threshold (px):").pack(pady=5)
        self.threshold_dist = tk.IntVar(value=100)
        tk.Entry(self.root, textvariable=self.threshold_dist).pack()

        # Start button
        tk.Button(
            self.root,
            text="Start Test",
            command=self.on_start
        ).pack(pady=15)

        self.root.mainloop()

    def on_start(self):
        try:
            if self.screen_width_cm.get() <= 0 or self.viewing_distance_cm.get() <= 0:
                raise ValueError("Screen width and viewing distance must be positive")
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return
        self.root.destroy()

    def get_config(self):
        return {
            'calib_points':        self.calib_points.get(),
            'stim_points':         self.stim_points.get(),
            'screen_width_cm':     self.screen_width_cm.get(),
            'viewing_distance_cm': self.viewing_distance_cm.get(),
            'threshold_dist':      self.threshold_dist.get()
        }

# ----------------------------------------------------------------
# 2. Helper functions
# ----------------------------------------------------------------
def angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg / 2))
    return int(size_cm * px_per_cm)

def generate_points(num_points):
    if num_points == 5:
        return [(0,0),(10,10),(-10,10),(10,-10),(-10,-10)]
    if num_points == 9:
        return [(0,10),(10,10),(-10,10),(10,-10),(-10,-10),
                (0,20),(20,0),(-20,0),(0,-20)]
    if num_points == 13:
        pts = [(0,0),(10,10),(-10,10),(10,-10),(-10,-10),
               (0,20),(20,0),(-20,0),(0,-20)]
        pts += [(15,15),(-15,15),(15,-15),(-15,-15)]
        return pts
    return []

def convert_positions_to_pixels(positions, width, height,
                                px_per_cm, dist_cm, diameter_px):
    DEG_TO_PX = lambda deg: int(px_per_cm * math.tan(math.radians(deg)) * dist_cm)
    raw = [(width//2 + DEG_TO_PX(x), height//2 - DEG_TO_PX(y)) for x,y in positions]
    margin = diameter_px//2 + 10
    clamped = []
    for x,y in raw:
        x = min(max(x, margin), width - margin - 140)
        y = min(max(y, margin), height - margin)
        clamped.append((x,y))
    return clamped
def calculate_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# ----------------------------------------------------------------
# 3. SVOP test with Quit button
# ----------------------------------------------------------------
def svop_test(screen, stimulus_positions,
              diameter_px, gf, stim_img, font, config):
    WIDTH, HEIGHT = screen.get_size()
    sidebar_w, btn_h, spacing = 140, 40, 10
    sidebar_x = WIDTH - sidebar_w

    btns = {
        'Pause': pygame.Rect(sidebar_x+10, 150,                sidebar_w-20, btn_h),
        'Skip' : pygame.Rect(sidebar_x+10, 150+btn_h+spacing,  sidebar_w-20, btn_h),
        'Retry': pygame.Rect(sidebar_x+10, 150+2*(btn_h+spacing),sidebar_w-20, btn_h),
        'Quit' : pygame.Rect(sidebar_x+10, 150+3*(btn_h+spacing),sidebar_w-20, btn_h)
    }

    pending = list(enumerate(stimulus_positions, start=1))
    results = []

    while pending:
        orig_idx, stim = pending.pop(0)
        gaze_q = deque(maxlen=10)
        paused = False
        start_t = time.time()
        total_pause = 0.0
        pause_start = None
        skip_flag = False
        retry_flag = False
        quit_flag = False

        while True:
            now = time.time()
            raw_elapsed = now - start_t
            if paused and pause_start is not None:
                elapsed = pause_start - start_t - total_pause
            else:
                elapsed = raw_elapsed - total_pause

            dist    = float('inf')

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = ev.pos
                    if btns['Pause'].collidepoint(mx,my):
                        paused = not paused
                        if paused:
                            pause_start = time.time()
                        else:
                            total_pause += time.time() - pause_start
                            pause_start = None
                    elif btns['Skip'].collidepoint(mx,my):
                        skip_flag = True
                        logging.info(f"Skip stimulus {orig_idx}")
                        break
                    elif btns['Retry'].collidepoint(mx,my):
                        retry_flag = True
                        logging.info(f"Retry stimulus {orig_idx}")
                        break
                    elif btns['Quit'].collidepoint(mx,my):
                        quit_flag = True
                        logging.info("User quit test")
                        break
            if skip_flag or retry_flag or quit_flag:
                break

            if paused:
                screen.fill(BACKGROUND_COLOR)
                pygame.draw.rect(screen, (50,50,50), (sidebar_x,0,sidebar_w,HEIGHT))
                for label, rect in btns.items():
                    color = {
                        'Pause': (100,100,100),
                        'Skip' : ERROR_COLOR,
                        'Retry': PASS_COLOR,
                        'Quit' : (150,50,50)
                    }[label]
                    pygame.draw.rect(screen, color, rect)
                    screen.blit(font.render(label, True, (0,0,0)), (rect.x+10, rect.y+8))
                screen.blit(font.render("PAUSED", True, (255,255,0)), (sidebar_x+10,10))
                pygame.display.flip()
                time.sleep(0.1)
                continue

            screen.fill(BACKGROUND_COLOR)
            pygame.draw.rect(screen, (50,50,50), (sidebar_x,0,sidebar_w,HEIGHT))
            rect = stim_img.get_rect(center=stim)
            screen.blit(stim_img, rect)

            gi = gf.get_gaze_info()
            if gi and getattr(gi,'status',False) and gi.filtered_gaze_coordinates:
                gx,gy = map(int, gi.filtered_gaze_coordinates)
                gaze_q.append((gx,gy))
                avgx = sum(x for x,_ in gaze_q)/len(gaze_q)
                avgy = sum(y for _,y in gaze_q)/len(gaze_q)
                dist = calculate_distance(stim,(avgx,avgy))
                pygame.draw.circle(screen, PASS_COLOR, (int(avgx),int(avgy)), 30,4)

            for label, rect in btns.items():
                color = {
                    'Pause': (100,100,100),
                    'Skip' : ERROR_COLOR,
                    'Retry': PASS_COLOR,
                    'Quit' : (150,50,50)
                }[label]
                pygame.draw.rect(screen, color, rect)
                screen.blit(font.render(label, True, (0,0,0)), (rect.x+10, rect.y+8))

            xo, yo = sidebar_x+10, 10
            screen.blit(font.render(f"Stim {orig_idx}/{len(stimulus_positions)}", True, (255,255,255)), (xo, yo)); yo+=30
            screen.blit(font.render(f"Time:{elapsed:.1f}s", True, (255,255,255)), (xo, yo)); yo+=30
            screen.blit(font.render(
                f"Dist:{'--' if dist==float('inf') else f'{dist:.1f}'}px",
                True, (255,255,255)
            ), (xo, yo))

            pygame.display.flip()
            time.sleep(0.01)

            if gaze_q and elapsed <= 5 and dist <= config['threshold_dist']:
                screen.fill(BACKGROUND_COLOR)
                pass_msg = font.render("PASS", True, PASS_COLOR)
                screen.blit(pass_msg, (WIDTH//2 - pass_msg.get_width()//2, HEIGHT//2 - pass_msg.get_height()//2))
                pygame.display.flip()
                time.sleep(1)
                passed = True
                break
            if elapsed > 5:
                passed = False
                logging.warning(f"Timeout stimulus {orig_idx}")
                screen.fill(BACKGROUND_COLOR)
                fail_msg = font.render("FAIL", True, ERROR_COLOR)
                screen.blit(fail_msg, (WIDTH//2 - fail_msg.get_width()//2, HEIGHT//2 - fail_msg.get_height()//2))
                pygame.display.flip()
                time.sleep(1)
                break

        if quit_flag:
            pygame.quit(); sys.exit()
        if skip_flag:
            continue
        if retry_flag:
            pending.insert(0, (orig_idx, stim))
            continue

        results.append({
            'stim_index': orig_idx,
            'stim_x': stim[0], 'stim_y': stim[1],
            'distance': dist, 'result': "PASS" if passed else "FAIL"
        })
        time.sleep(0.5)

    df = pd.DataFrame(results)
    fname = f"svop_results_{datetime.datetime.now():%Y%m%d%H%M%S}.csv"
    df.to_csv(fname, index=False)
    logging.info(f"Results saved to {fname}")

# ----------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------
def main():
    gui = ConfigGUI()
    cfg = gui.get_config()

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 28)
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    PX_PER_CM = WIDTH / cfg['screen_width_cm']
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    stim_img = pygame.image.load("./VF-test/ball.jpg").convert_alpha()
    dpx = angular_to_pixel_diameter(ANGULAR_DIAMETERS['Goldmann IV'], cfg['viewing_distance_cm'], PX_PER_CM)
    stim_img = pygame.transform.scale(stim_img, (dpx*2, dpx*2))

    dcfg = DefaultConfig(); dcfg.cali_mode = cfg['calib_points']
    gf = GazeFollower(config=dcfg)
    gf.preview(win=screen); gf.calibrate(win=screen); gf.start_sampling(); time.sleep(0.1)

    pts_deg = generate_points(cfg['stim_points'])
    stim_px = convert_positions_to_pixels(pts_deg, WIDTH, HEIGHT, PX_PER_CM, cfg['viewing_distance_cm'], dpx)
    svop_test(screen, stim_px, dpx, gf, stim_img, font, cfg)

    gf.stop_sampling(); os.makedirs('data', exist_ok=True)
    gf.save_data(os.path.join('data','svop_demo.csv'))
    gf.release(); pygame.quit()

if __name__ == '__main__':
    main()
