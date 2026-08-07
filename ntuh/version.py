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
    1.0.1  Fix: the Sol gaze accuracy test gave the operator no way to tell whether the
           subject was fixating the target. Added an operator-only tester-monitor view
           (schematic of the subject screen with the target, the live gaze dot, and the
           target->gaze offset in px/deg) on the configured Tester Screen, with
           focus-independent SPACE/Q. Nothing is drawn on the subject screen (so the dot
           cannot be chased); no change on a single-monitor setup.
    1.1.0  The "wait for stable valid data" gate can now be overridden: pressing SPACE
           (from the subject or the tester window) starts the trial immediately. The
           dashboard banner shows the hint. VF trials are unaffected.
calibration
    1.0.0  Baseline: versioning introduced.
    1.0.1  Multi-screen selection, screen-width (cm) input, image size shown in
           cm, flexible/auto-sized config window, configurable profile output
           folder, remembered settings, a webcam preview button (identify cameras
           without calibrating), 'q'-to-quit on the calibration screens, and
           English-keyboard switch/restore with crash recovery (all matching
           VA_center_opt).
    1.1.0  A finished profile is now also copied to the sibling
           VA_center_opt/calibration_profiles/ folder (release layout), so it can be used
           without the manual copy step; the "Done" dialog shows both paths.
replayer
    1.0.0  Baseline: versioning introduced.
"""

APP_VERSIONS = {
    "VA_center_opt": "1.1.0",
    "calibration": "1.1.0",
    "replayer": "1.0.0",
}


def get_version(app: str) -> str:
    """Return the 'MAJOR.MINOR.PATCH' string for an app key (see APP_VERSIONS)."""
    return APP_VERSIONS.get(app, "0.0.0")
