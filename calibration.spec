# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for calibration.py
To build: python -m PyInstaller --clean -y calibration.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- Apply workarounds (must be before Analysis) ---
import pyinstaller_helpers
pyinstaller_helpers.patch_dll_discovery()

# ===================== Hidden imports =====================
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
    'mediapipe',
    'cv2',
    'pygame',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'numpy',
    'pandas',
    'scipy',
    # MediaPipe dependencies
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
]

# ===================== Data files =====================
datas = []
try:
    datas += collect_data_files('mediapipe')
except Exception:
    pass
try:
    datas += collect_data_files('matplotlib')
except Exception:
    pass

datas += [('gazefollower', 'gazefollower')]

if os.path.exists('calibration_images'):
    datas += [('calibration_images', 'calibration_images')]
if os.path.exists('stimulus_images'):
    datas += [('stimulus_images', 'stimulus_images')]

if os.path.exists('calibration_profiles'):
    datas += [('calibration_profiles', 'calibration_profiles')]

# ===================== Binaries (manual DLL inclusion) =====================
binaries = pyinstaller_helpers.collect_manual_binaries()
datas += pyinstaller_helpers.collect_manual_datas()

# ===================== Analysis =====================
a = Analysis(
    ['calibration.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/runtime_hook_mediapipe.py'],
    excludes=['IPython', 'jupyter', 'MNN', '_mnncengine'],
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
    name='calibration',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='calibration',
)
