"""Writable-data base directory (APP_DIR) + shared path constants.

APP_DIR is where the app reads/writes settings, calibration profiles, output and logs.
When frozen by PyInstaller it is the .exe directory; otherwise the directory of the entry
script (sys.argv[0]) - the same location the apps used before the reorg.
"""
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(sys.argv[0]).resolve().parent

LAST_SETTINGS_FILE = APP_DIR / "VA_output" / "last_settings_opt.json"
