# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VA_center_opt.py
To build: pyinstaller VA_center_opt.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules
hiddenimports = [
    'gazefollower',
    'gazefollower.camera',
    'gazefollower.camera.Camera',
    'gazefollower.camera.WebCamCamera',
    'gazefollower.face_alignment',
    'gazefollower.face_alignment.MediaPipeFaceAlignment',
    'gazefollower.gaze_estimator',
    'gazefollower.gaze_estimator.MGazeNetGazeEstimator',
    'gazefollower.calibration',
    'gazefollower.filter',
    'gazefollower.misc',
    'gazefollower.logger',
    'gazefollower.ui',
    'recorder',
    'sol_tracker',
    'mediapipe',
    'cv2',
    'pygame',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'tkinter',
    'tkinter.ttk',
    'numpy',
    'pandas',
    'scipy',
    # MediaPipe dependencies
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
]

# Collect data files from mediapipe and matplotlib
datas = []
try:
    datas += collect_data_files('mediapipe')
except:
    pass

try:
    datas += collect_data_files('matplotlib')
except:
    pass

# Add gazefollower package
datas += [('gazefollower', 'gazefollower')]

# Add local modules
datas += [('recorder.py', '.'), ('sol_tracker.py', '.')]

# Add calibration profiles if they exist
if os.path.exists('calibration_profiles'):
    datas += [('calibration_profiles', 'calibration_profiles')]

# Binaries (empty, but can add DLLs if needed)
binaries = []

a = Analysis(
    ['VA_center_opt.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter'],  # Exclude unnecessary packages (matplotlib needed by mediapipe)
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VA_center_opt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VA_center_opt',
)
