# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for replayer.py
To build: python -m PyInstaller --clean -y replayer.spec
"""

import sys
import os

block_cipher = None

# --- Apply workarounds (must be before Analysis) ---
import pyinstaller_helpers
pyinstaller_helpers.patch_dll_discovery()

# ===================== Hidden imports =====================
hiddenimports = [
    'cv2',
    'numpy',
    'pandas',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
]

# ===================== Binaries (manual DLL inclusion) =====================
# Replayer only needs numpy/pandas/cv2, but we use the same helper for consistency
binaries = pyinstaller_helpers.collect_manual_binaries()
datas = pyinstaller_helpers.collect_manual_datas()

# ===================== Analysis =====================
a = Analysis(
    ['replayer.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'matplotlib', 'scipy', 'mediapipe', 'PIL',
              'MNN', '_mnncengine'],
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
    name='replayer',
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
    name='replayer',
)
