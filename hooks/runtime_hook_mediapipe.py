"""
Runtime hook for mediapipe.
Adds the mediapipe/python directory to DLL search paths so that
_framework_bindings.pyd can find opencv_world3410.dll at runtime.
"""
import os
import sys

# In PyInstaller frozen app, sys._MEIPASS is the _internal directory
if getattr(sys, 'frozen', False):
    meipass = sys._MEIPASS
    mp_python_dir = os.path.join(meipass, 'mediapipe', 'python')
    if os.path.isdir(mp_python_dir):
        # Add to DLL search directories (Windows 10+ API)
        try:
            os.add_dll_directory(mp_python_dir)
        except (OSError, AttributeError):
            pass
        # Also add to PATH as fallback
        os.environ['PATH'] = mp_python_dir + os.pathsep + os.environ.get('PATH', '')
