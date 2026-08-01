# -*- coding: utf-8 -*-
"""Scripted local release for the NTUH Eye-Tracking Suite.

The build needs Windows + the native stack + the vendored Ganzin wheel and produces ~2 GB, so
releases are cut locally (not in CI). This script:

  1. (default) cleans build/ + dist/ and builds all three apps via their .spec files, then stages
     runtime data/docs with stage_release.py;
  2. zips dist/ into  release/<YYYYMMDD>_<name>.zip  (forward-slash entries, ZIP64-safe), keeping
     only the current archive there;
  3. (--tag) creates per-app git tags  <app>-v<version>  for the versions in ntuh/version.py.

Usage (from the repo root, venv active):
    python packaging/release.py                 # build all 3, stage, and zip
    python packaging/release.py --no-build      # package the existing dist/ only
    python packaging/release.py --tag           # also create per-app git tags
    python packaging/release.py --name Foo      # zip base name (default NTUH_EyeTracking_Suite)

Pushing tags and merging develop -> main are left to a human (see CLAUDE.md).
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# This script lives in packaging/; the repo root (dist/, release/, ntuh/, specs) is its parent.
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
RELEASE = ROOT / "release"
SPECS = ["packaging/VA_center_opt.spec", "packaging/calibration.spec", "packaging/replayer.spec"]
APPS = ["VA_center_opt", "calibration", "replayer"]


def app_versions():
    sys.path.insert(0, str(ROOT))
    from ntuh.version import APP_VERSIONS
    return dict(APP_VERSIONS)


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def build():
    for d in (ROOT / "build", DIST):
        if d.exists():
            shutil.rmtree(d)
    for spec in SPECS:
        run([sys.executable, "-m", "PyInstaller", "--clean", "-y", spec])
    run([sys.executable, "packaging/stage_release.py"])


def make_zip(name):
    if not DIST.is_dir():
        print("ERROR: dist/ not found - build first (omit --no-build).", file=sys.stderr)
        return None
    RELEASE.mkdir(exist_ok=True)
    for old in RELEASE.glob("*.zip"):
        print(f"  removing superseded archive: {old.name}")
        old.unlink()
    zpath = RELEASE / f"{datetime.now():%Y%m%d}_{name}.zip"
    files = [p for p in sorted(DIST.rglob("*")) if p.is_file()]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in files:
            z.write(p, p.relative_to(DIST).as_posix())  # forward-slash entries
    return zpath, len(files)


def git_dirty():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def tag(vers):
    if git_dirty():
        print("  WARNING: working tree is not clean - tags will point at the current HEAD anyway.")
    for app in APPS:
        t = f"{app}-v{vers[app]}"
        exists = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{t}"],
                                cwd=str(ROOT), capture_output=True).returncode == 0
        if exists:
            print(f"  tag already exists, skipping: {t}")
        else:
            run(["git", "tag", "-a", t, "-m", f"{app} v{vers[app]}"])


def main():
    ap = argparse.ArgumentParser(description="Build + package a release of the three apps.")
    ap.add_argument("--no-build", action="store_true", help="package existing dist/ (skip PyInstaller)")
    ap.add_argument("--tag", action="store_true", help="create per-app git tags for the current versions")
    ap.add_argument("--name", default="NTUH_EyeTracking_Suite", help="base name for the zip")
    args = ap.parse_args()

    vers = app_versions()
    print("App versions: " + ", ".join(f"{a} v{vers[a]}" for a in APPS))

    if not args.no_build:
        build()

    for app in APPS:
        if not (DIST / app / f"{app}.exe").exists():
            print(f"  WARNING: dist/{app}/{app}.exe is missing")

    result = make_zip(args.name)
    if result is None:
        return 1
    zpath, n = result
    print(f"\nPackaged {n} files -> {zpath}  ({zpath.stat().st_size / 1e6:.0f} MB)")

    if args.tag:
        tag(vers)

    print("\nNext steps (by hand, see CLAUDE.md):")
    print("  - push tags:      git push --tags")
    print("  - stable release: open a PR to merge develop -> main")
    print(f"  - hand off:       release/{zpath.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
