# -*- coding: utf-8 -*-
# ================================================================
# SVOP Demo ‧ 隱藏側邊按鈕版（可從 GUI 選擇Goldmann II~V、刺激圖片、並持續旋轉）
# 可載入/儲存 calibration checkpoint；可調整 gaze marker；加入校正圖與尺寸
# 預覽/校正前後使用事件過濾器，避免空白鍵卡住
# 結束時顯示 PASS/FAIL 總數
# ================================================================
import os, sys, math, time, datetime, logging
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import pygame, pandas as pd
from pathlib import Path

import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from gazefollower.calibration import SVRCalibration
import json
from tkinter import messagebox
from gazefollower.logger import Log as GFLog

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("svop_debug.log"), logging.StreamHandler()]
)
LAST_SETTINGS_FILE = Path(__file__).resolve().parent / "VF_output" / "last_settings.json"

# ----------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------
ANGULAR_DIAMETERS = {
    "Goldmann II":   0.43,
    "Goldmann III":  0.64,
    "Goldmann IV":   0.86,
    "Goldmann V":    1.72
}
SHOW_BUTTONS      = False
PASS_DWELL_SEC    = 2.0
TIMEOUT_SEC       = 5.0
BACKGROUND_COLOR  = (0, 0, 0)
PASS_COLOR        = (0, 255, 0)
ERROR_COLOR       = (255, 0, 0)

# ----------------------------------------------------------------
# Event-filter helpers（防止空白鍵卡住）
# ----------------------------------------------------------------
def restore_event_filter():
    # 允許所有事件（None = 關閉過濾）
    try:
        pygame.event.set_allowed(None)
        pygame.event.clear()
        pygame.event.pump()
        # 視情況釋放 grab
        try: pygame.event.set_grab(False)
        except Exception: pass
    except Exception:
        pass

def prep_input_for_calibration():
    # 1) 關掉文字輸入/IME，避免空白被當成選字
    try:
        pygame.key.stop_text_input()
    except Exception:
        pass
    # 2) 清掉卡住的修飾鍵狀態（Shift/Ctrl/Alt）
    try:
        pygame.key.set_mods(0)
    except Exception:
        pass
    # 3) 事件佇列清空，只保留關鍵事件
    pygame.event.set_allowed(None)
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT,
                              pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                              pygame.ACTIVEEVENT])
    pygame.event.clear()
    pygame.event.pump()
    # 4) 把鍵盤/滑鼠焦點鎖到這個視窗（可選）
    try:
        pygame.event.set_grab(True)
    except Exception:
        pass

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def to_rgb_tuple_str(s, default=(0,255,0)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) == 3:
            return tuple(max(0, min(255, v)) for v in parts)
    except Exception:
        pass
    return default

def angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg/2))
    return int(size_cm * px_per_cm)

def get_quadrant(x, y, cx, cy):
    if x > cx and y < cy: return 1
    if x < cx and y < cy: return 2
    if x < cx and y > cy: return 3
    if x > cx and y > cy: return 4
    return None

def generate_points(n, max_deg_horizon, max_deg_vertical_deg):
    if n == 5:
        return [(0,0),( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),
                ( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg)]
    if n == 9:
        return [(0, max_deg_vertical_deg),( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),
                ( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg),( max_deg_horizon,0),
                (-max_deg_horizon,0),(0,-max_deg_vertical_deg),(0,0)]
    if n == 13:
        return [
            (0,0),( max_deg_horizon,0),(-max_deg_horizon,0),(0, max_deg_vertical_deg),(0,-max_deg_vertical_deg),
            ( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),
            ( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg),
            ( max_deg_horizon, max_deg_vertical_deg/2),(-max_deg_horizon, max_deg_vertical_deg/2),
            ( max_deg_horizon,-max_deg_vertical_deg/2),(-max_deg_horizon,-max_deg_vertical_deg/2)
        ]
    raise ValueError("num_points must be 5/9/13")

def convert_positions_to_pixels(deg_pts, w, h, px_per_cm, dist_cm, diameter_px):
    d2p = lambda d: int(px_per_cm * math.tan(math.radians(d)) * dist_cm)
    raw = [(w//2 + d2p(x), h//2 - d2p(y)) for x,y in deg_pts]
    margin = diameter_px//2 + 10
    right  = w - margin
    return [(max(margin,min(x,right)),max(margin,min(y,h-margin))) for x,y in raw]

def _default_profile_dir(cfg, W, H):
    """預設的校正檔案資料夾：<專案>/calibration_profiles/{user}_{pts}pt_{WxH}"""
    project_root = Path(__file__).resolve().parent
    base = project_root / "calibration_profiles"
    base.mkdir(parents=True, exist_ok=True)
    name = (cfg['user_name'] or "default").strip().replace(" ", "_")
    return base / f"{name}_{cfg['calib_points']}pt_{W}x{H}"

# ----------------------------------------------------------------
# Config GUI
# ----------------------------------------------------------------
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SVOP Test Configuration")
        self.root.geometry("560x1080")
        self.cancelled = True  # 預設視為取消，只有按 Start 時才改為 False
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
        # User name
        tk.Label(self.root, text="User Name:").pack(pady=4)
        self.user_name = tk.StringVar(value="test_subject")
        tk.Entry(self.root, textvariable=self.user_name).pack()

        # Goldmann Size
        tk.Label(self.root, text="Goldmann Size:").pack(pady=4)
        self.size = tk.StringVar(value="Goldmann IV")
        ttk.Combobox(
            self.root,
            textvariable=self.size,
            values=list(ANGULAR_DIAMETERS.keys()),
            state="readonly"
        ).pack()

        # Calibration points
        tk.Label(self.root, text="Calibration Points:").pack(pady=4)
        self.calib_points = tk.IntVar(value=9)
        ttk.Combobox(
            self.root,
            textvariable=self.calib_points,
            values=[5, 9, 13],
            state="readonly"
        ).pack()

        # Stimulus points
        tk.Label(self.root, text="Stimulus Points:").pack(pady=4)
        self.stim_points = tk.IntVar(value=9)
        ttk.Combobox(
            self.root,
            textvariable=self.stim_points,
            values=[5, 9, 13],
            state="readonly"
        ).pack()

        # Screen width
        tk.Label(self.root, text="Screen Width (cm):").pack(pady=4)
        self.screen_width_cm = tk.DoubleVar(value=52.704)
        tk.Entry(self.root, textvariable=self.screen_width_cm).pack()

        # Viewing distance
        tk.Label(self.root, text="Viewing Distance (cm):").pack(pady=4)
        self.viewing_distance_cm = tk.DoubleVar(value=45.0)
        tk.Entry(self.root, textvariable=self.viewing_distance_cm).pack()

        # Pass threshold
        tk.Label(self.root, text="Pass Threshold (px):").pack(pady=4)
        self.threshold_dist = tk.IntVar(value=500)
        tk.Entry(self.root, textvariable=self.threshold_dist).pack()

        # Enable rotation?
        self.enable_rotation = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.root,
            text="Enable Continuous Rotation",
            variable=self.enable_rotation
        ).pack(pady=4)

        # Rotation speed
        tk.Label(self.root, text="Rotation Speed (°/s):").pack(pady=4)
        self.rot_speed = tk.DoubleVar(value=90.0)
        tk.Entry(self.root, textvariable=self.rot_speed).pack()

        # Stimulus image
        tk.Label(self.root, text="Stimulus Image:").pack(pady=4)
        self.stim_path = tk.StringVar(value="./stimulus_image/ball.jpg")
        frame = tk.Frame(self.root); frame.pack(fill="x", padx=10)
        tk.Entry(frame, textvariable=self.stim_path).pack(side="left", expand=True, fill="x")
        tk.Button(frame, text="Browse…", command=self.browse).pack(side="right")

        # --- Calibration folder (optional) ---
        tk.Label(self.root, text="Calibration folder (optional):").pack(pady=4)
        self.calib_dir = tk.StringVar(value="")
        cframe = tk.Frame(self.root); cframe.pack(fill="x", padx=10)
        tk.Entry(cframe, textvariable=self.calib_dir).pack(side="left", expand=True, fill="x")
        tk.Button(cframe, text="Browse…", command=self.browse_calib_dir).pack(side="right")

        # --- Calibration target image & size ---
        tk.Label(self.root, text="Calibration target image (optional):").pack(pady=4)
        self.cali_img_path = tk.StringVar(value="")
        cif = tk.Frame(self.root); cif.pack(fill="x", padx=10)
        tk.Entry(cif, textvariable=self.cali_img_path).pack(side="left", expand=True, fill="x")
        tk.Button(cif, text="Browse…", command=self.browse_cali_img).pack(side="right")

        tk.Label(self.root, text="Calibration image size (px):").pack(pady=4)
        sizef = tk.Frame(self.root); sizef.pack()
        self.cali_w = tk.IntVar(value=170)
        self.cali_h = tk.IntVar(value=170)
        tk.Entry(sizef, textvariable=self.cali_w, width=6).pack(side="left")
        tk.Label(sizef, text=" x ").pack(side="left")
        tk.Entry(sizef, textvariable=self.cali_h, width=6).pack(side="left")
        # --- Gaze marker style ---
        tk.Label(self.root, text="Gaze marker color (R,G,B):").pack(pady=4)
        self.gaze_color = tk.StringVar(value="0,255,0")
        gframe = tk.Frame(self.root); gframe.pack(fill="x", padx=10)
        tk.Entry(gframe, textvariable=self.gaze_color, width=14).pack(side="left")
        tk.Button(gframe, text="Pick", command=self.pick_color).pack(side="left", padx=6)

        tk.Label(self.root, text="Gaze marker radius (px):").pack(pady=4)
        self.gaze_radius = tk.IntVar(value=20)
        tk.Entry(self.root, textvariable=self.gaze_radius).pack()

        tk.Label(self.root, text="Gaze marker line width (0=filled):").pack(pady=4)
        self.gaze_width = tk.IntVar(value=2)
        tk.Entry(self.root, textvariable=self.gaze_width).pack()

        # ▶ 新增：按鈕列（載入上一筆 / 開始）
        btn_row = tk.Frame(self.root); btn_row.pack(pady=12)
        tk.Button(btn_row, text="Use last settings", command=self.on_load_last).pack(side="left", padx=6)
        tk.Button(btn_row, text="Start Test", command=self.on_start).pack(side="left", padx=6)

        self.root.mainloop()
    def on_cancel(self):
        self.cancelled = True
        self.root.destroy()
    def pick_color(self):
        rgb = colorchooser.askcolor()[0]
        if rgb:
            r,g,b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self.gaze_color.set(f"{r},{g},{b}")

    def browse(self):
        f = filedialog.askopenfilename(
            title="Select stimulus image",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")]
        )
        if f:
            self.stim_path.set(f)

    def browse_calib_dir(self):
        d = filedialog.askdirectory(title="Select calibration folder")
        if d:
            self.calib_dir.set(d)

    def browse_cali_img(self):
        f = filedialog.askopenfilename(
            title="Select calibration target image",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All", "*.*")]
        )
        if f:
            self.cali_img_path.set(f)

    def on_start(self):
        try:
            if self.screen_width_cm.get() <= 0 or self.viewing_distance_cm.get() <= 0:
                raise ValueError("Screen width/distance must be > 0")
            if self.enable_rotation.get() and self.rot_speed.get() < 0:
                raise ValueError("Rotate speed must be ≥ 0")
            if not os.path.isfile(self.stim_path.get()):
                raise ValueError("Stimulus image not found")
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        # ✅ 存成 last_settings.json 供下次一鍵載入
        self._save_last_settings()
        self.cancelled = False   # ← 使用者真的要開始

        self.root.destroy()


    def get(self):
        return {
            "user_name":           self.user_name.get().strip().replace(" ", "_"),
            "goldmann_size":       self.size.get(),
            "calib_points":        self.calib_points.get(),
            "stim_points":         self.stim_points.get(),
            "screen_width_cm":     self.screen_width_cm.get(),
            "viewing_distance_cm": self.viewing_distance_cm.get(),
            "threshold_dist":      self.threshold_dist.get(),
            "enable_rotation":     self.enable_rotation.get(),
            "rot_speed":           self.rot_speed.get(),
            "stim_path":           self.stim_path.get(),
            "calib_dir":           self.calib_dir.get().strip(),
            "cali_img_path":       self.cali_img_path.get().strip(),
            "cali_img_size":       (self.cali_w.get(), self.cali_h.get()),
            "gaze_color":          self.gaze_color.get().strip(),
            "gaze_radius":         self.gaze_radius.get(),
            "gaze_width":          self.gaze_width.get(),
        }
    def _save_last_settings(self):
        """把目前 GUI 的設定存到 JSON，供下次載入"""
        data = self.get()
        try:
            LAST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Save last settings failed: {e}")

    def on_load_last(self):
        """按鈕事件：載入上一次設定並套到 GUI"""
        if not LAST_SETTINGS_FILE.exists():
            messagebox.showinfo("No saved settings", "找不到上一筆設定（last_settings.json）。")
            return
        try:
            with open(LAST_SETTINGS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            messagebox.showerror("Load error", f"讀取 last_settings.json 失敗：\n{e}")
            return

        # 逐欄套回 GUI（有值才覆蓋）
        self.user_name.set(d.get("user_name", self.user_name.get()))
        self.size.set(d.get("goldmann_size", self.size.get()))
        self.calib_points.set(d.get("calib_points", self.calib_points.get()))
        self.stim_points.set(d.get("stim_points", self.stim_points.get()))
        self.screen_width_cm.set(float(d.get("screen_width_cm", self.screen_width_cm.get())))
        self.viewing_distance_cm.set(float(d.get("viewing_distance_cm", self.viewing_distance_cm.get())))
        self.threshold_dist.set(int(d.get("threshold_dist", self.threshold_dist.get())))
        self.enable_rotation.set(bool(d.get("enable_rotation", self.enable_rotation.get())))
        self.rot_speed.set(float(d.get("rot_speed", self.rot_speed.get())))
        self.stim_path.set(d.get("stim_path", self.stim_path.get()))
        self.calib_dir.set(d.get("calib_dir", self.calib_dir.get()))
        self.cali_img_path.set(d.get("cali_img_path", self.cali_img_path.get()))

        # cali_img_size 可能是 list/tuple
        cali_size = d.get("cali_img_size", (self.cali_w.get(), self.cali_h.get()))
        try:
            w, h = int(cali_size[0]), int(cali_size[1])
            self.cali_w.set(w); self.cali_h.set(h)
        except Exception:
            pass

        self.gaze_color.set(d.get("gaze_color", self.gaze_color.get()))
        self.gaze_radius.set(int(d.get("gaze_radius", self.gaze_radius.get())))
        self.gaze_width.set(int(d.get("gaze_width", self.gaze_width.get())))

        messagebox.showinfo("Loaded", "已套用上一筆設定。")
# ----------------------------------------------------------------
# SVOP Test (with continuous rotation & end summary)
# ----------------------------------------------------------------
def svop_test(screen, stim_pts, stim_deg_list, diameter_px, gf, stim_img, font, small_font, cfg):
    W,H = screen.get_size()
    cx,cy = W//2,H//2
    threshold  = cfg['threshold_dist']
    rot_speed  = cfg['rot_speed']
    do_rotate  = cfg['enable_rotation']
    orig_stim  = stim_img.copy()
    angle      = 0.0

    # gaze 樣式
    gaze_color = to_rgb_tuple_str(cfg['gaze_color'], (0,255,0))
    gaze_radius= int(cfg.get('gaze_radius', 20))
    gaze_width = int(cfg.get('gaze_width', 2))

    results = []

    # 一一呈現刺激
    for idx, stim in enumerate(stim_pts, start=1):
        target_q   = get_quadrant(stim[0], stim[1], cx, cy)
        dwell_start= None
        passed     = False
        t0         = time.time()
        last_t     = t0

        while True:
            now  = time.time()
            dt   = now - last_t
            last_t = now
            elapsed = now - t0

            # 按 Q 強制結束
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    gf.stop_sampling()
                    gf.release(); pygame.quit(); sys.exit()

            screen.fill(BACKGROUND_COLOR)

            if do_rotate:
                angle = (angle + rot_speed*dt) % 360
                disp = pygame.transform.rotate(orig_stim, angle)
                rect = disp.get_rect(center=stim)
            else:
                disp = orig_stim
                rect = orig_stim.get_rect(center=stim)

            screen.blit(disp, rect)

            # 讀取 gaze
            gi = gf.get_gaze_info()
            if gi and getattr(gi,'status',False):
                coords = getattr(gi, 'filtered_gaze_coordinates', None) or \
                         getattr(gi, 'gaze_coordinates', None)
                if coords:
                    gx,gy = map(int, coords)
                    curr_q= get_quadrant(gx,gy,cx,cy)
                    dist_px = math.hypot(stim[0]-gx, stim[1]-gy)

                    if target_q is None or curr_q is None:
                        inside = (dist_px <= threshold)
                    else:
                        inside = (curr_q == target_q)

                    if inside:
                        dwell_start = dwell_start or now
                    else:
                        dwell_start = None

                    pygame.draw.circle(screen, gaze_color, (gx,gy), gaze_radius, gaze_width)

                    if dwell_start and (now-dwell_start)>=PASS_DWELL_SEC:
                        passed = True
                        break

            if elapsed > TIMEOUT_SEC:
                passed = False
                break

            pygame.display.flip()
            time.sleep(0.01)

        x_deg,y_deg = stim_deg_list[idx-1]
        stim_ecc = math.hypot(x_deg, y_deg)
        # 記錄結果
        results.append({
            "user_name": cfg['user_name'],
            "stim_index": idx,
            "stim_x": stim[0], "stim_y": stim[1],
            "x_deg": x_deg, "y_deg": y_deg,
            "stim_ecc_deg": stim_ecc,
            "target_quadrant": target_q,
            "result": "PASS" if passed else "FAIL"
        })

        # 顯示單次 PASS/FAIL
        screen.fill(BACKGROUND_COLOR)
        msg = "PASS" if passed else "FAIL"
        col = PASS_COLOR if passed else ERROR_COLOR
        txt = font.render(msg, True, col)
        screen.blit(txt, (cx-txt.get_width()//2, cy-txt.get_height()//2))
        pygame.display.flip()
        time.sleep(1)

    # 結束後顯示總數
    pass_count = sum(1 for r in results if r['result']=="PASS")
    fail_count = sum(1 for r in results if r['result']=="FAIL")
    screen.fill(BACKGROUND_COLOR)
    summary1 = font.render(f"Total PASS: {pass_count}", True, PASS_COLOR)
    summary2 = font.render(f"Total FAIL: {fail_count}", True, ERROR_COLOR)
    screen.blit(summary1, (cx-summary1.get_width()//2, cy- summary1.get_height()))
    screen.blit(summary2, (cx-summary2.get_width()//2, cy+10))
    pygame.display.flip()
    time.sleep(3)

    # 儲存 CSV
    df = pd.DataFrame(results)
    Path("VF_output").mkdir(parents=True, exist_ok=True)
    fname = f"VF_output/svop_{cfg['user_name']}_{datetime.datetime.now():%Y%m%d%H%M%S}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    logging.info(f"Results saved to {fname}")
def _init_gf_logger():
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gazefollower_{time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))   # ✅ 一定要在用到 SVRCalibration 前呼叫
    except Exception:
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "GazeFollower" / "gazefollower.log"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        GFLog.init(str(tmp))

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    _init_gf_logger()   # ← 加這行

    gui = ConfigGUI()
    if gui.cancelled:
        logging.info("User cancelled in config GUI. Exit.")
        return  # 或 sys.exit(0)

    cfg = gui.get()
    pygame.init(); pygame.font.init()
    small_font = pygame.font.SysFont(None, 24)
    font = pygame.font.SysFont(None, 72)

    info = pygame.display.Info()
    W,H = info.current_w, info.current_h
    px_per_cm = W / cfg['screen_width_cm']
    dist_cm   = cfg['viewing_distance_cm']
    half_w_cm = cfg['screen_width_cm']/2
    max_deg_horizon = math.degrees(math.atan(half_w_cm/dist_cm))
    screen_height_cm = cfg['screen_width_cm'] * (info.current_h / info.current_w)
    half_h_cm = screen_height_cm / 2
    max_deg_vertical = math.degrees(math.atan(half_h_cm/dist_cm))

    screen = pygame.display.set_mode((W,H), pygame.FULLSCREEN)

    # load stimulus
    if not os.path.isfile(cfg['stim_path']):
        messagebox.showerror("File Error", f"Image not found:\n{cfg['stim_path']}")
        sys.exit(1)
    raw = pygame.image.load(cfg['stim_path'])
    stim_raw = raw.convert_alpha() if raw.get_alpha() else raw.convert()

    angle_deg  = ANGULAR_DIAMETERS[cfg['goldmann_size']]
    diameter_px= angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm)
    stim_img   = pygame.transform.scale(stim_raw, (diameter_px, diameter_px))

    # ---- Calibration checkpoint ----
    dcfg = DefaultConfig(); dcfg.cali_mode = cfg['calib_points']

    # 套用校正圖/尺寸（有填才設定）
    if cfg.get('cali_img_path'):
        dcfg.cali_target_img = cfg['cali_img_path']
    if cfg.get('cali_img_size'):
        dcfg.cali_target_size = tuple(cfg['cali_img_size'])

    # 校正檔資料夾
    if cfg.get("calib_dir"):
        profile_dir = Path(cfg["calib_dir"])
    else:
        profile_dir = _default_profile_dir(cfg, W, H)
    profile_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Calibration folder: {profile_dir}")

    calib = SVRCalibration(model_save_path=str(profile_dir))
    gf = GazeFollower(config=dcfg, calibration=calib)

    # ---- 預覽/校正：前置清理事件 → 預覽/校正 → 恢復事件過濾 ----
    prep_input_for_calibration()
    gf.preview(win=screen)

    if not gf.calibration.has_calibrated:
        logging.info("No checkpoint found → running calibration…")
        gf.calibrate(win=screen)
        ok = gf.calibration.save_model()
        logging.info(f"Calibration saved: {ok}")
    else:
        logging.info("Loaded checkpoint → skip calibrate()")

    restore_event_filter()

    gf.start_sampling(); time.sleep(0.1)

    pts_deg = generate_points(cfg['stim_points'], max_deg_horizon, max_deg_vertical)
    pts_px  = convert_positions_to_pixels(pts_deg, W, H, px_per_cm, dist_cm, diameter_px)

    svop_test(screen, pts_px, pts_deg, diameter_px, gf, stim_img, font, small_font, cfg)

    gf.stop_sampling(); pygame.quit()

if __name__ == "__main__":
    Path("VF_output").mkdir(parents=True, exist_ok=True)
    main()
