# Building VA_center_opt.exe - Step by Step Guide

This guide will help you create a standalone Windows executable (.exe) for VA_center_opt.py.

## Prerequisites

1. **Python installed** (with all dependencies working)
2. **All required packages installed**:
   ```bash
   pip install pyinstaller
   ```

## Quick Build (Recommended)

### Option 1: Using the Build Script (Easiest)

1. Double-click `build_exe.bat`
2. Wait for the build to complete (may take 5-10 minutes)
3. Find your executable in `dist\VA_center_opt\VA_center_opt.exe`

### Option 2: Manual Build

Open Command Prompt in the ntuh-eyetracking folder and run:

```bash
pyinstaller VA_center_opt.spec
```

## Build Output

After successful build:
- **Executable**: `dist\VA_center_opt\VA_center_opt.exe`
- **Supporting files**: All required DLLs and data files in `dist\VA_center_opt\`

## Distribution

### To distribute your application:

1. Copy the entire `dist\VA_center_opt` folder
2. Share the folder (you can zip it first)
3. Users can run `VA_center_opt.exe` directly (no Python needed!)

### Important Files in Distribution:

```
dist\VA_center_opt\
├── VA_center_opt.exe          # Main executable
├── calibration_profiles\      # User calibration data (if exists)
├── gazefollower\              # Eye tracking models
├── recorder.py                # Recording module
├── sol_tracker.py             # Sol glasses support
└── _internal\                 # All dependencies (DLLs, packages)
```

## Troubleshooting

### Build Issues

1. **Missing module error during build**:
   ```bash
   # Add the missing module to hiddenimports in VA_center_opt.spec
   # Then rebuild
   ```

2. **Build takes too long**:
   - Normal! First build can take 5-10 minutes
   - Subsequent builds are faster

3. **Build fails with import errors**:
   ```bash
   # Make sure all dependencies are installed
   pip install -r requirements.txt
   ```

### Runtime Issues

1. **Executable crashes immediately**:
   - Run from command prompt to see error messages:
     ```bash
     cd dist\VA_center_opt
     VA_center_opt.exe
     ```

2. **DLL errors**:
   - Make sure to distribute the ENTIRE folder, not just the .exe
   - Windows may need Visual C++ Redistributable

3. **"Failed to execute script" error**:
   - Edit `VA_center_opt.spec` and change `console=True` to see error details
   - Rebuild: `pyinstaller VA_center_opt.spec`

## Customization

### Change Icon

1. Create or download an `.ico` file
2. Edit `VA_center_opt.spec`:
   ```python
   icon='path/to/your/icon.ico'
   ```
3. Rebuild

### Hide Console Window

Edit `VA_center_opt.spec`:
```python
console=False,  # Change True to False
```

Then rebuild.

### Single File Executable (Not Recommended for Large Apps)

If you want a single .exe file instead of a folder:

Edit `VA_center_opt.spec`:
```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,        # Add this
    a.zipfiles,        # Add this
    a.datas,           # Add this
    [],
    exclude_binaries=False,  # Change to False
    name='VA_center_opt',
    # ... rest stays the same
)

# REMOVE or comment out the COLLECT section
```

⚠️ **Warning**: Single file will be slower to start and may have antivirus issues.

## Advanced Configuration

### Reduce File Size

1. Edit `VA_center_opt.spec` and add to excludes:
   ```python
   excludes=['matplotlib', 'IPython', 'jupyter', 'pytest', 'sphinx'],
   ```

2. Use UPX compression (already enabled):
   - Download UPX from https://upx.github.io/
   - Add to PATH
   - Rebuild

### Add Version Info

Create `version.txt`:
```
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'NTUH'),
        StringStruct(u'FileDescription', u'Visual Acuity Testing'),
        StringStruct(u'FileVersion', u'1.0.0'),
        StringStruct(u'ProductName', u'VA Center'),
        StringStruct(u'ProductVersion', u'1.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
```

Then in spec file:
```python
exe = EXE(
    ...
    version='version.txt',
    ...
)
```

## Testing the Executable

1. **Test on build machine first**:
   ```bash
   cd dist\VA_center_opt
   VA_center_opt.exe
   ```

2. **Test on clean Windows VM/machine**:
   - Copy the entire `dist\VA_center_opt` folder
   - Run without Python installed
   - Verify all features work

3. **Test checklist**:
   - [ ] GUI opens correctly
   - [ ] Webcam detection works
   - [ ] Calibration loads
   - [ ] Recording works
   - [ ] Sol glasses connection (if available)
   - [ ] Data export works

## File Size Reference

Typical build sizes:
- **Folder**: 500MB - 1GB (includes all dependencies)
- **Single file**: 300MB - 500MB (slower startup)

This is normal due to:
- MediaPipe models
- OpenCV
- NumPy/SciPy
- TensorFlow Lite

## Support

If you encounter issues:

1. Check the error message carefully
2. Enable console window (`console=True`)
3. Run from command prompt to see full traceback
4. Check PyInstaller logs in `build\VA_center_opt\`

## Reference Links

- [PyInstaller Documentation](https://pyinstaller.org/)
- [PyInstaller Common Issues](https://github.com/pyinstaller/pyinstaller/wiki/How-to-Report-Bugs)
- [UPX Compression](https://upx.github.io/)
