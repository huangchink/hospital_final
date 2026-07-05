"""Single source of truth for the release version of the three built apps.

Each app shows its version in the window title (e.g. "Calibration (v1.0.0)") so a
user can tell exactly which build they are running.

Release policy
--------------
- Versions are MAJOR.MINOR.PATCH.
    PATCH  -> bug fix / no behaviour change users would notice
    MINOR  -> new user-facing feature or option
    MAJOR  -> breaking change to the workflow or data layout
- Whenever a program is modified for a release, bump *that program's* entry below
  before building, and add a one-line note to its changelog so we know which
  version a given build corresponds to.

Changelog
---------
VA_center_opt
    1.0.0  Baseline: versioning introduced.
calibration
    1.0.0  Baseline: multi-screen selection, screen-width (cm) input, image size
           shown in cm, flexible/auto-sized config window, configurable profile
           output folder, remembered settings, a webcam preview button (identify
           cameras without calibrating), 'q'-to-quit on the calibration screens,
           and English-keyboard switch/restore with crash recovery (all matching
           VA_center_opt).
replayer
    1.0.0  Baseline: versioning introduced.
"""

APP_VERSIONS = {
    "VA_center_opt": "1.0.0",
    "calibration": "1.0.0",
    "replayer": "1.0.0",
}


def get_version(app: str) -> str:
    """Return the 'MAJOR.MINOR.PATCH' string for an app key (see APP_VERSIONS)."""
    return APP_VERSIONS.get(app, "0.0.0")
