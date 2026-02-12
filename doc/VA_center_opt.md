# VA_center_opt - Visual Acuity Test

A visual acuity (VA) testing program using contrast sensitivity gratings with support for both webcam-based and Sol glasses eye tracking.

## Overview

VA_center_opt presents circular sinusoidal gratings at the center of the screen and uses a staircase procedure to measure the user's contrast sensitivity threshold. It supports dual eye-tracking sources (webcam + Sol glasses), records gaze data, screen video, and webcam video throughout the entire experiment.

## Quick Start

1. **Calibrate first** - Run `calibration.exe` to create a user calibration profile (required for webcam tracking)
2. **Launch** - Double-click `VA_center_opt.exe` (or `run_debug.bat` for error output)
3. **Configure** - Enter settings in the GUI
4. **Run** - Click "Start Test" or "Start Practice"

## Settings

### General Tab - User & Tracker

| Setting | Description | Default |
|---------|-------------|---------|
| User Name | Participant identifier (must match calibration profile) | anonymous |
| Trackers | Enable Webcam and/or Sol Glasses | Webcam: ON, Sol: OFF |
| Evaluation Source | Which tracker to use for VA scoring (Webcam or Sol) | Webcam |
| Show Gaze Marker | Display gaze point on screen during test | ON |

### General Tab - Stimulus

| Setting | Description | Default |
|---------|-------------|---------|
| Stimulus Duration (s) | How long each grating is displayed | 5.0 |
| Pass Duration (s) | Time gaze must be on target to count as "pass" | 2.0 |
| Blank Duration (s) | Duration of blank screen between stimulus and response | 1.0 |
| Circle Radius (px) | Radius of the circular grating stimulus | 200 |
| Rotate Stimulus | Enable rotation of the grating | OFF |
| Speed (deg/s) | Rotation speed | 60.0 |

### General Tab - Colors & Display

| Setting | Description | Default |
|---------|-------------|---------|
| Bright Color | Bright bars of the grating (R,G,B) | 255,255,255 |
| Dark Color | Dark bars of the grating (R,G,B) | 0,0,0 |
| Background Color | Screen background color (R,G,B) | 0,0,0 |
| Paper Color Mode | Gray background with black/white grating and white border | OFF |

### General Tab - Screen & Viewing

| Setting | Description | Default |
|---------|-------------|---------|
| Screen Width (cm) | Physical width of the display monitor | 52.6 |
| Viewing Distance (cm) | Distance from user's eyes to the screen | 50 |

### General Tab - Inter-trial

| Setting | Description | Default |
|---------|-------------|---------|
| Image | Optional image displayed between trials | (none) |
| Image Duration (s) | How long the inter-trial image is shown | 1.1 |
| Background Hold (s) | Duration of blank background after inter-trial image | 1.0 |

### Webcam Tab

| Setting | Description | Default |
|---------|-------------|---------|
| Calibration Folder | Path to webcam calibration profiles | calibration_profiles/ |
| Select Camera | Webcam device index | 0 |

> **Note:** The Calibration Folder should point to the `calibration_profiles` directory inside the calibration tool's `_internal` folder (e.g., `calibration/_internal/calibration_profiles/`). This is where `calibration.exe` saves its output profiles.

### Sol Tab

| Setting | Description | Default |
|---------|-------------|---------|
| IP / Port | Sol glasses network address | 192.168.1.121 : 8080 |
| Markers (HxV) | ArUco marker grid layout | 8 x 5 |
| Pattern Size (px) | Size of each ArUco marker on screen | 120 |
| Dict | ArUco dictionary type | DICT_4X4_250 |
| Gaze Method | 3D (ray-plane) or 2D (homography) | 2D |
| Pose Smooth / Gaze Smooth | Smoothing factors (0-1, higher = more smooth) | 1 / 1 |

> **Note:** Before running experiments, the tester should test different ArUco marker counts (HxV) and pattern sizes on the target screen to ensure markers do not overlap each other or the stimulus area. Choosing the right combination for the specific screen size is important for achieving the best gaze tracking accuracy.

### Sol Calib Tab

Offset calibration for Sol glasses. Corrects systematic gaze offset when the Sol built-in calibration is inaccurate.

| Setting | Description | Default |
|---------|-------------|---------|
| Target Image | Image shown as fixation target during calibration | (none) |
| Target Size (px) | Display size of the target | 100 |
| Calibration Points | Number of calibration positions (1-5) | 5 |

### Recording Tab

| Setting | Description | Default |
|---------|-------------|---------|
| Resolution | Recording resolution (Original, 1920x1080, 1280x720) | Original |
| Record Webcam Data | Record webcam video + gaze data | ON |
| Record Sol Data | Record Sol glasses gaze data | ON |
| Export Raw Sol Video | Save raw Sol camera video | ON |

## Keyboard Controls During Test

| Key | Action |
|-----|--------|
| ESC | Abort the test and return to settings |

## Output Data

Experiment data is saved to `VA_output/`:

### Session Folder

Each session creates a timestamped folder:

```
VA_output/
  VA_{username}_sess1_{date}_{time}/
    webcam_gaze_data.csv        - Webcam gaze coordinates per frame
    sol_gaze_data.csv           - Sol gaze coordinates per frame
    webcam_video.mp4            - Webcam recording
    screen_record.mp4           - Screen recording
    sol_video.mp4               - Sol camera recording (if enabled)
    webcam_video_timestamp.csv  - Frame timestamps for webcam video
    screen_video_timestamp.csv  - Frame timestamps for screen video
    sol_video_timestamp.csv     - Frame timestamps for Sol video
```

### Results CSV

Trial-by-trial VA results are saved as a separate CSV at the top level:

```
VA_output/
  VA_{username}_opt.csv         - CPD and pass/fail per trial
```

Gaze data is recorded continuously during all phases (stimulus, feedback, inter-trial).

## Practice Mode

Click "Start Practice" to run a practice session. Practice mode uses the same stimulus but does not record any data. Useful for familiarizing participants with the task.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Calibration not found" | Run calibration.exe first with the same username |
| Camera not detected | Close other camera apps, try different camera index |
| Sol glasses not connecting | Check IP/port, ensure glasses are on the same network |
| Black screen | Press ESC, check camera and display settings |
| Webcam disconnects mid-test | Program auto-reconnects (up to 5 attempts) |
