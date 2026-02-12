# Replayer - Experiment Data Replay Tool

Replays recorded experiment sessions with synchronized webcam video, screen video, Sol camera video, and gaze data overlays.

## Overview

After running an experiment with VA_center_opt, the replayer lets you review the recorded data. It synchronizes all video streams by timestamp and overlays gaze coordinates on the screen recording. Useful for verifying data quality, debugging tracking issues, and reviewing participant behavior.

## Quick Start

1. **Launch** - Double-click `replayer.exe` (or `run_debug.bat` for error output)
2. **Select folder** - Browse to the experiment output folder (e.g., `VA_output/{session}/`)
3. **Playback** - Use keyboard controls to navigate the recording

## Input Data

The replayer expects a session folder from VA_center_opt containing:

```
{session_folder}/
  webcam_video.mp4              - Webcam recording
  webcam_video_timestamp.csv    - Frame timestamps
  webcam_gaze_data.csv          - Webcam gaze data
  screen_record.mp4             - Screen recording
  screen_video_timestamp.csv    - Frame timestamps
  sol_video.mp4                 - Sol camera recording (optional)
  sol_video_timestamp.csv       - Frame timestamps (optional)
  sol_gaze_data.csv             - Sol gaze data (optional)
```

Not all files are required. The replayer will display whichever streams are available.

## Keyboard Controls

| Key | Action |
|-----|--------|
| SPACE | Pause / Resume playback |
| D | Seek forward 1 second |
| A | Seek backward 1 second |
| W | Seek forward 5 seconds |
| S | Seek backward 5 seconds |
| R | Restart from beginning |
| ] or = | Increase playback speed (+0.1x) |
| [ or - | Decrease playback speed (-0.1x) |
| Q / ESC | Quit the replayer |

## Display

The replayer window shows available video streams side by side with:

- Gaze point markers overlaid on the screen recording
- Synchronized timestamps across all streams
- Current playback position and speed

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No video displayed | Ensure the session folder contains .avi files |
| Videos out of sync | Check that timestamp CSV files exist alongside videos |
| "File not found" error | Browse to the correct VA_output session folder |
