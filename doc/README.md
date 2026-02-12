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
3. Collect results from `VA_output/` folder
4. (Optional) Use `replayer.exe` to review recordings

## Building from Source

See [deprecated/BUILD_EXE_README.md](deprecated/BUILD_EXE_README.md) for PyInstaller build instructions.

To build all three executables:
```
build_exe.bat
```

## Troubleshooting

Use `run_debug.bat` (included alongside each .exe) to see error messages in the console.

## Contact

For questions or issues, contact Edan Chen.
