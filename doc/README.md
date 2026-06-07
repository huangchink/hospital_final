# NTUH Eye-Tracking Experiment Suite

This package contains three programs for eye-tracking visual acuity experiments.

## Programs

| Program | Description | Documentation |
|---------|-------------|---------------|
| **calibration.exe** | Create webcam eye-tracking calibration profiles | [calibration.md](calibration.md) |
| **VA_center_opt.exe** | Run visual acuity experiments (webcam + Sol glasses) | [VA_center_opt.md](VA_center_opt.md) |
| **replayer.exe** | Replay and review recorded experiment data | [replayer.md](replayer.md) |

## Workflow

```
1. calibration.exe    -->  Create user calibration profile
2. VA_center_opt.exe  -->  Run the VA experiment
3. replayer.exe       -->  Review recorded data
```

### For each participant:

1. Run `calibration.exe` - enter participant name, complete calibration
2. Run `VA_center_opt.exe` - use the **same** participant name, configure settings, run test
3. Collect results from the `VA_output/` folder (next to `VA_center_opt.exe`)
4. (Optional) Use `replayer.exe` to review **and label** recordings

> **Latest changes:** see [20260607_release_note.txt](20260607_release_note.txt) for what is
> new in the current build and how to use it.

## Data location

Each program reads/writes its data **next to its own `.exe`** (not inside `_internal/`):
`VA_output/`, `calibration_profiles/`, `logs/`, and the image folders.

**Sharing calibration profiles:** `calibration.exe` saves to `calibration/calibration_profiles/`,
while `VA_center_opt.exe` reads from `VA_center_opt/calibration_profiles/`. After calibrating,
either copy the new `<name>_<points>pt` folder into `VA_center_opt/calibration_profiles/`, or set
VA_center_opt's **Webcam tab → Calibration Folder** to point at `calibration/calibration_profiles/`.
A default `anonymous_9pt` profile is included in both.

## Building from Source

Run from the repo root (uses `VA_center_opt.spec` / `calibration.spec` / `replayer.spec` +
`stage_release.py`):
```
build_exe.bat
```
This builds all three debug/console exes into `dist/`, copies the default profile, image folders
and these docs next to each exe, and writes a `run_debug.bat` launcher per app.

## Troubleshooting

Use `run_debug.bat` (included alongside each .exe) to see error messages in the console
(the console window staying open is expected for these debug builds).

## Contact

For questions or issues, contact Edan Chen.
