import cv2
import numpy as np
import mediapipe as mp
import random
import time
import winsound
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget, QComboBox, QFileDialog, QCheckBox, QLineEdit
from PyQt5.QtCore import Qt
import sys
import os
from collections import deque

print("Initializing Mediapipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
print("Mediapipe initialized.")

# Default parameters
screen_width, screen_height = 1920, 1080
stimulus_duration = 2.0  
blank_screen_duration = 1.0
font = cv2.FONT_HERSHEY_SIMPLEX
background_color_rgb = (127.5, 127.5, 127.5)
background_color_bgr = tuple(reversed(background_color_rgb))
stimulus_color_rgb = (255, 255, 255)
stimulus_color_bgr = tuple(reversed(stimulus_color_rgb))
circle_radius = 250
frequency_min = 10
frequency_max = 900
initial_step = 200
fine_step_1 = 100
fine_step_mid = 150
fine_step_2 = 50
frequency = 100  
calibration_points_count = 3
calibration_image_path = "pikachu01.jpg"
calibration_image = cv2.imread(calibration_image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(calibration_image_path) else None
enable_sound = True
user_id = ""

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

def generate_grating_pixel_level(frequency, screen_width, screen_height):
    x = np.arange(screen_width)
    y = np.arange(screen_height)
    xx, yy = np.meshgrid(x, y)
    grating = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * xx / screen_width)
    grating = (grating * 255).astype(np.float32)
    grating = cv2.GaussianBlur(grating, (5, 5), sigmaX=1.0)
    return grating.astype(np.uint8)

def apply_color_to_stimulus(stimulus, color):
    colored_stimulus = np.zeros((stimulus.shape[0], stimulus.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        colored_stimulus[:, :, i] = stimulus * (color[i] / 255.0)
    return colored_stimulus

def create_circle_mask(radius, point):
    mask = np.zeros((screen_height, screen_width), dtype=np.uint8)
    centers = {
        "center": (screen_width // 2, screen_height // 2),
        "left": (250, screen_height // 2),
        "right": (1650, screen_height // 2),
        "left_up": (250, screen_height // 4),
        "right_up": (1650, screen_height // 4),
        "left_down": (250, 3 * screen_height // 4),
        "right_down": (1650, 3 * screen_height // 4),
        "up": (screen_width // 2, screen_height // 4),
        "down": (screen_width // 2, 3 * screen_height // 4)
    }
    center = centers.get(point, (screen_width // 2, screen_height // 2))
    cv2.circle(mask, center, radius, 255, -1)
    return mask, center

def overlay_image(screen, img, position):
    print("Overlaying image...")
    h, w = img.shape[:2]
    y, x = position[1] - h // 2, position[0] - w // 2
    if y < 0 or x < 0 or y + h > screen_height or x + w > screen_width:
        print("Image out of bounds. Skipping overlay.")
        return screen
    try:
        if img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            img_rgb = img[:, :, :3]
            alpha = alpha[:, :, np.newaxis]
            screen_region = screen[y:y+h, x:x+w]
            blended = (1 - alpha) * screen_region + alpha * img_rgb
            screen[y:y+h, x:x+w] = blended.astype(np.uint8)
        else:
            screen[y:y+h, x:x+w] = img
        print("Image overlaid successfully.")
    except Exception as e:
        print(f"Error overlaying image: {str(e)}")
    return screen

def calibrate_gaze(cap, points_count):
    print("Starting calibration...")
    global calibration_image
    if calibration_image is None:
        print("No calibration image found. Using default red circle.")
        calibration_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(calibration_image, (50, 50), 50, (255, 0, 0), -1)
    else:
        calibration_image = cv2.resize(calibration_image, (100, 100))

    if points_count == 3:
        calibration_points = ["center", "left", "right"]
    elif points_count == 5:
        calibration_points = ["center", "left", "right", "left_up", "right_up"]
    else:
        calibration_points = ["center", "left", "right", "left_up", "right_up", "left_down", "right_down", "up", "down"]
    
    calibration_data = {point: [] for point in calibration_points}
    calibration_duration = 2.0  
    gaze_buffer = deque(maxlen=5)

    centers = {
        "center": (screen_width // 2, screen_height // 2),
        "left": (250, screen_height // 2),
        "right": (1650, screen_height // 2),
        "left_up": (250, screen_height // 4),
        "right_up": (1650, screen_height // 4),
        "left_down": (250, 3 * screen_height // 4),
        "right_down": (1650, 3 * screen_height // 4),
        "up": (screen_width // 2, screen_height // 4),
        "down": (screen_width // 2, 3 * screen_height // 4)
    }

    for point in calibration_points:
        print(f"Calibrating point: {point}")
        if enable_sound:
            winsound.Beep(1000, 200)
        start_time = time.time()
        while time.time() - start_time < calibration_duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            screen = np.full((screen_height, screen_width, 3), background_color_bgr, dtype=np.uint8)
            cv2.putText(screen, f"Look at {point.upper()} point", (screen_width // 3, screen_height // 4), font, 1.5, (0, 255, 0), 2)
            center = centers.get(point, (screen_width // 2, screen_height // 2))
            screen = overlay_image(screen, calibration_image, center)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_eye_inner = landmarks[133]
                right_eye_inner = landmarks[362]
                nose_tip = landmarks[1]
                avg_eye_center = (left_eye_inner.x + right_eye_inner.x) / 2 - nose_tip.x
                gaze_buffer.append(avg_eye_center)
                smoothed_eye_center = np.mean(gaze_buffer) if gaze_buffer else avg_eye_center
                calibration_data[point].append(smoothed_eye_center)

            cv2.imshow("Calibration", screen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Calibration aborted by user.")
                return None, None

    left_points = [data for point in ["left", "left_up", "left_down"] if point in calibration_data for data in calibration_data[point]]
    right_points = [data for point in ["right", "right_up", "right_down"] if point in calibration_data for data in calibration_data[point]]

    left_mean = np.mean(left_points) if left_points else 0.1
    right_mean = np.mean(right_points) if right_points else -0.1
    left_std = np.std(left_points) if left_points else 0.05
    right_std = np.std(right_points) if right_points else 0.05

    left_threshold = left_mean - left_std
    right_threshold = right_mean + right_std

    print(f"Calibration thresholds: left={left_threshold:.3f}, right={right_threshold:.3f}, left_mean={left_mean:.3f}, right_mean={right_mean:.3f}")
    print("Calibration completed.")
    return left_threshold, right_threshold

def determine_gaze_direction(landmarks, left_threshold, right_threshold, gaze_buffer):
    left_eye_inner = landmarks[133]
    right_eye_inner = landmarks[362]
    nose_tip = landmarks[1]
    avg_eye_center = (left_eye_inner.x + right_eye_inner.x) / 2 - nose_tip.x
    gaze_buffer.append(avg_eye_center)
    smoothed_eye_center = np.mean(gaze_buffer) if gaze_buffer else avg_eye_center

    if smoothed_eye_center > left_threshold:
        return "left", smoothed_eye_center
    elif smoothed_eye_center < right_threshold:
        return "right", smoothed_eye_center
    return "center", smoothed_eye_center

class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing PyQt5 Settings Window...")
        self.setWindowTitle("Test Settings")
        self.setFixedSize(600, 800)

        layout = QVBoxLayout()

        def create_label(text):
            label = QLabel(text)
            label.setStyleSheet("font-size: 14px; margin: 5px;")
            return label

        layout.addWidget(create_label("User ID:"))
        self.user_id = QLineEdit()
        self.user_id.setPlaceholderText("Enter User ID")
        self.user_id.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.user_id)

        layout.addWidget(create_label("Frequency Min (cycles/screen):"))
        self.freq_min = QSpinBox()
        self.freq_min.setRange(1, 100)
        self.freq_min.setValue(frequency_min)
        self.freq_min.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.freq_min)

        layout.addWidget(create_label("Frequency Max (cycles/screen):"))
        self.freq_max = QSpinBox()
        self.freq_max.setRange(100, 1000)
        self.freq_max.setValue(frequency_max)
        self.freq_max.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.freq_max)

        layout.addWidget(create_label("Initial Step Size: Fixed at 200"))
        self.initial_step_label = QLabel("200")
        self.initial_step_label.setStyleSheet("font-size: 12px; margin: 5px;")
        layout.addWidget(self.initial_step_label)

        layout.addWidget(create_label("Fine Step 1 (after first error):"))
        self.fine_step_1 = QSpinBox()
        self.fine_step_1.setRange(50, 200)
        self.fine_step_1.setValue(fine_step_1)
        self.fine_step_1.setSingleStep(10)
        self.fine_step_1.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.fine_step_1)

        layout.addWidget(create_label("Fine Step Mid (after 3 correct):"))
        self.fine_step_mid = QSpinBox()
        self.fine_step_mid.setRange(100, 200)
        self.fine_step_mid.setValue(fine_step_mid)
        self.fine_step_mid.setSingleStep(10)
        self.fine_step_mid.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.fine_step_mid)

        layout.addWidget(create_label("Fine Step 2 (after third reversal):"))
        self.fine_step_2 = QSpinBox()
        self.fine_step_2.setRange(50, 100)
        self.fine_step_2.setValue(fine_step_2)
        self.fine_step_2.setSingleStep(10)
        self.fine_step_2.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.fine_step_2)

        layout.addWidget(create_label("Stimulus Duration (seconds):"))
        self.stim_duration = QDoubleSpinBox()
        self.stim_duration.setRange(0.5, 10.0)
        self.stim_duration.setValue(stimulus_duration)
        self.stim_duration.setSingleStep(0.1)
        self.stim_duration.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.stim_duration)

        layout.addWidget(create_label("Blank Screen Duration (seconds):"))
        self.blank_duration = QDoubleSpinBox()
        self.blank_duration.setRange(0.5, 10.0)
        self.blank_duration.setValue(blank_screen_duration)
        self.blank_duration.setSingleStep(0.1)
        self.blank_duration.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.blank_duration)

        layout.addWidget(create_label("Circle Radius (pixels):"))
        self.circle_rad = QSpinBox()
        self.circle_rad.setRange(50, 500)
        self.circle_rad.setValue(circle_radius)
        self.circle_rad.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.circle_rad)

        layout.addWidget(create_label("Calibration Points:"))
        self.calib_points = QComboBox()
        self.calib_points.addItems(["3 points", "5 points", "9 points"])
        self.calib_points.setCurrentText("3 points")
        self.calib_points.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.calib_points)

        layout.addWidget(create_label("Calibration Image:"))
        self.calib_image_label = QLabel("pikachu01.jpg" if calibration_image is not None else "No file selected")
        self.calib_image_label.setStyleSheet("font-size: 12px; margin: 5px;")
        layout.addWidget(self.calib_image_label)
        self.calib_image_button = QPushButton("Select Image")
        self.calib_image_button.setStyleSheet("font-size: 12px; padding: 5px;")
        self.calib_image_button.clicked.connect(self.select_calibration_image)
        layout.addWidget(self.calib_image_button)

        layout.addWidget(create_label("Sound Settings:"))
        self.sound_toggle = QCheckBox("Enable Sound")
        self.sound_toggle.setChecked(True)
        self.sound_toggle.setStyleSheet("font-size: 12px; margin: 5px;")
        layout.addWidget(self.sound_toggle)

        start_button = QPushButton("Start Test")
        start_button.setStyleSheet("font-size: 14px; padding: 10px;")
        start_button.clicked.connect(self.start_test)
        layout.addWidget(start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        print("Settings Window initialized.")

    def select_calibration_image(self):
        global calibration_image_path, calibration_image
        print("Opening file dialog for calibration image...")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            calibration_image_path = file_path
            calibration_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if calibration_image is None:
                self.calib_image_label.setText("Error loading image")
                print(f"Failed to load image: {file_path}")
            else:
                self.calib_image_label.setText(file_path.split('/')[-1])
                print(f"Calibration image selected: {file_path}")

    def start_test(self):
        global frequency_min, frequency_max, initial_step, fine_step_1, fine_step_mid, fine_step_2, stimulus_duration, blank_screen_duration, circle_radius, frequency, calibration_points_count, enable_sound, user_id
        print("Starting test with settings:")
        user_id_raw = self.user_id.text().strip()
        user_id_safe = "".join(c for c in user_id_raw if c.isalnum() or c in ('-', '_'))
        user_id = user_id_safe if user_id_safe else ""
        print(f"user_id={user_id if user_id else 'Anonymous'}, freq_min={self.freq_min.value()}, freq_max={self.freq_max.value()}, stimulus_duration={self.stim_duration.value()}")
        frequency_min = self.freq_min.value()
        frequency_max = self.freq_max.value()
        initial_step = 200
        fine_step_1 = self.fine_step_1.value()
        fine_step_mid = self.fine_step_mid.value()
        fine_step_2 = self.fine_step_2.value()
        stimulus_duration = self.stim_duration.value()
        blank_screen_duration = self.blank_duration.value()
        circle_radius = self.circle_rad.value()
        frequency = 100  # Increased from 10
        calib_text = self.calib_points.currentText()
        calibration_points_count = 3 if calib_text == "3 points" else 5 if calib_text == "5 points" else 9
        enable_sound = self.sound_toggle.isChecked()
        print("Settings applied. Closing settings window.")
        self.close()

class Staircase:
    def __init__(self, freq_min, freq_max, initial_step, fine_step_1, fine_step_mid, fine_step_2):
        self.frequency = 100  # Increased from freq_min
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.initial_step = initial_step
        self.fine_step_1 = fine_step_1
        self.fine_step_mid = fine_step_mid
        self.fine_step_2 = fine_step_2
        self.current_step = initial_step
        self.correct_streak = 0
        self.incorrect_streak = 0
        self.reversals = []
        self.last_correct = True
        self.first_error = False
        self.max_correct_streak = 0
        self.step_history = [initial_step]

    def update(self, is_correct):
        if is_correct:
            self.correct_streak += 1
            self.incorrect_streak = 0
            if self.frequency >= self.freq_max:
                self.max_correct_streak += 1
            if self.correct_streak >= 3 and self.frequency < self.freq_max and not self.first_error:
                self.frequency = min(self.frequency + self.current_step, self.freq_max)
                self.correct_streak = 0
                if not self.last_correct:
                    self.reversals.append(self.frequency)
                    if len(self.reversals) >= 3:
                        self.current_step = self.fine_step_2
                        self.step_history.append(self.current_step)
                self.last_correct = True
            elif self.correct_streak >= 3 and self.frequency < self.freq_max and self.first_error:
                if self.current_step == self.fine_step_1:
                    self.current_step = self.fine_step_mid
                elif self.current_step == self.fine_step_mid:
                    self.current_step = self.initial_step
                self.frequency = min(self.frequency + self.current_step, self.freq_max)
                self.correct_streak = 0
                self.step_history.append(self.current_step)
                if not self.last_correct:
                    self.reversals.append(self.frequency)
                    if len(self.reversals) >= 3:
                        self.current_step = self.fine_step_2
                        self.step_history.append(self.current_step)
                self.last_correct = True
            elif self.correct_streak >= 2 and self.frequency < self.freq_max:
                self.frequency = min(self.frequency + self.current_step, self.freq_max)
                self.correct_streak = 0
                if not self.last_correct:
                    self.reversals.append(self.frequency)
                    if len(self.reversals) >= 3:
                        self.current_step = self.fine_step_2
                        self.step_history.append(self.current_step)
                self.last_correct = True
        else:
            self.incorrect_streak += 1
            self.correct_streak = 0
            self.max_correct_streak = 0
            if self.incorrect_streak >= 1:
                if not self.first_error:
                    self.first_error = True
                    self.current_step = self.fine_step_1
                elif self.current_step == self.initial_step:
                    self.current_step = self.fine_step_mid
                elif self.current_step == self.fine_step_mid:
                    self.current_step = self.fine_step_1
                self.frequency = max(self.frequency - self.current_step, self.freq_min)
                self.incorrect_streak = 0
                self.step_history.append(self.current_step)
                if self.last_correct:
                    self.reversals.append(self.frequency)
                    if len(self.reversals) >= 3:
                        self.current_step = self.fine_step_2
                        self.step_history.append(self.current_step)
                self.last_correct = False

    def is_finished(self):
        return len(self.reversals) >= 4 or (self.frequency >= self.freq_max and self.max_correct_streak >= 3)

    def get_result(self):
        if self.frequency >= self.freq_max and self.max_correct_streak >= 3:
            return self.freq_max
        if self.reversals:
            return np.mean(self.reversals[-4:])
        return self.frequency

def run_test():
    global frequency
    print("Opening camera...")
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera. Exiting.")
        return

    print("Starting gaze calibration...")
    left_threshold, right_threshold = calibrate_gaze(cap, calibration_points_count)
    if left_threshold is None or right_threshold is None:
        print("Calibration failed or aborted. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    gaze_buffer = deque(maxlen=5)

    stimulus = generate_grating_pixel_level(frequency, screen_width, screen_height)
    start_time = time.time()
    is_blank_screen = False
    blank_start_time = 0
    test_ended = False
    va_result = None
    side = random.choice(["left", "right"])
    staircase = Staircase(frequency_min, frequency_max, initial_step, fine_step_1, fine_step_mid, fine_step_2)
    feedback_text = ""
    feedback_start_time = 0
    feedback_duration = 1.0
    is_paused = False

    print("Starting main test loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        screen = np.full((screen_height, screen_width, 3), background_color_bgr, dtype=np.uint8)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            is_paused = True
            if enable_sound:
                winsound.Beep(600, 200)
            print("Test paused.")
        elif key == ord("c"):
            is_paused = False
            if enable_sound:
                winsound.Beep(900, 200)
            print("Test resumed.")
            if not is_blank_screen:
                start_time = time.time() - (elapsed_time if 'elapsed_time' in locals() else 0)
            else:
                blank_start_time = time.time() - (time.time() - blank_start_time if 'blank_start_time' in locals() else 0)
        elif key == ord("q"):
            print("Test aborted by user.")
            break

        if is_paused:
            cv2.putText(screen, "Paused. Press 'c' to continue.", (screen_width // 3, screen_height // 2), font, 1.5, (0, 255, 255), 2)
            cv2.imshow("Stimuli with Gaze Tracking", screen)
            continue

        if results.multi_face_landmarks and not test_ended:
            landmarks = results.multi_face_landmarks[0].landmark
            gaze_direction, smoothed_eye_center = determine_gaze_direction(landmarks, left_threshold, right_threshold, gaze_buffer)

            cv2.putText(screen, f"Gaze: {gaze_direction.upper()}", (10, 50), font, 1, (0, 255, 0), 2)
            cv2.putText(screen, f"Frequency: {frequency:.3f}", (10, 90), font, 1, (255, 0, 0), 2)

            elapsed_time = time.time() - start_time
            if elapsed_time > stimulus_duration and not is_blank_screen:
                is_correct = (gaze_direction == side)
                staircase.update(is_correct)
                frequency = staircase.frequency
                stimulus = generate_grating_pixel_level(frequency, screen_width, screen_height)
                feedback_text = "Correct!" if is_correct else "Incorrect!"
                feedback_start_time = time.time()
                if enable_sound:
                    if is_correct:
                        winsound.Beep(1200, 300)
                    else:
                        winsound.Beep(800, 300)
                side = random.choice(["left", "right"])
                is_blank_screen = True
                blank_start_time = time.time()

                if staircase.is_finished():
                    test_ended = True
                    final_freq = staircase.get_result()
                    va_result = get_va_result(final_freq)
                    print(f"Test finished. User ID: {user_id if user_id else 'Anonymous'}, Final frequency: {final_freq}, VA result: {va_result}")
                    if user_id:
                        output_filename = f"result_{user_id}_{time.strftime('%Y%m%d')}.txt"
                    else:
                        output_filename = f"result_Anonymous_{time.strftime('%Y%m%d')}.txt"
                    try:
                        with open(output_filename, "w", encoding="utf-8") as f:
                            f.write(f"Test Result\n")
                            f.write(f"==========\n")
                            f.write(f"User ID: {user_id if user_id else 'Anonymous'}\n")
                            f.write(f"Final Frequency: {final_freq:.3f}\n")
                            f.write(f"VA Result: {va_result}\n")
                            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        print(f"Results saved to {output_filename}")
                    except Exception as e:
                        print(f"Error saving results to file: {str(e)}")

            elif is_blank_screen and time.time() - blank_start_time > blank_screen_duration:
                is_blank_screen = False
                start_time = time.time()
                if enable_sound:
                    winsound.Beep(1000, 200)

        if is_blank_screen or test_ended:
            if test_ended:
                cv2.putText(screen, f"User ID: {user_id if user_id else 'Anonymous'}", (screen_width // 3, screen_height // 2 - 100), font, 2, (0, 255, 255), 3)
                cv2.putText(screen, "TEST END", (screen_width // 3, screen_height // 2 - 50), font, 2, (0, 0, 255), 3)
                cv2.putText(screen, f"VA Result: {va_result}", (screen_width // 3, screen_height // 2 + 50), font, 2, (0, 255, 0), 3)
        else:
            stimulus_colored = apply_color_to_stimulus(stimulus, stimulus_color_bgr)
            circle_mask, _ = create_circle_mask(circle_radius, side)
            stimulus_circular = cv2.bitwise_and(stimulus_colored, stimulus_colored, mask=circle_mask)
            mask_inv = cv2.bitwise_not(circle_mask)
            screen = cv2.bitwise_and(screen, screen, mask=mask_inv)
            screen = cv2.add(screen, stimulus_circular)
            left_center = (250, screen_height // 2)
            right_center = (1650, screen_height // 2)
            cv2.circle(screen, left_center, circle_radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(screen, right_center, circle_radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        if time.time() - feedback_start_time < feedback_duration:
            text_size, _ = cv2.getTextSize(feedback_text, font, 1.5, 2)
            text_x = (screen_width - text_size[0]) // 2
            text_y = screen_height // 2
            cv2.putText(screen, feedback_text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)

        cv2.imshow("Stimuli with Gaze Tracking", screen)

    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed.")

if __name__ == "__main__":
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("Creating settings window...")
        window = SettingsWindow()
        print("Showing settings window...")
        window.show()
        print("Entering main event loop...")
        app.exec_()
        print("Running test...")
        run_test()
        print("Application finished.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")