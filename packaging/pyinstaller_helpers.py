"""
Shared PyInstaller workarounds for this project.

Problem: PyInstaller's "Looking for dynamic libraries" phase spawns an isolated
subprocess that __import__()s every collected package to discover DLL search
paths (os.add_dll_directory / PATH). Some native .pyd files (e.g., MNN's
_mnncengine.pyd) cause an access violation (c0000005 in MSVCP140.dll) that
kills the subprocess and fails the build.

Solution: We monkey-patch find_binary_dependencies to pass an empty package
list, completely skipping the isolated subprocess DLL imports. Then we manually
include the DLLs that would have been discovered:
  - mediapipe native binaries (opencv_world3410.dll, _framework_bindings.pyd)
  - delvewheel .libs directories (numpy.libs, pandas.libs, scipy.libs)
  - MNN native engine (_mnncengine.pyd)

Usage in spec files:
    import pyinstaller_helpers
    pyinstaller_helpers.patch_dll_discovery()
    ...
    binaries += pyinstaller_helpers.collect_manual_binaries()
    datas += pyinstaller_helpers.collect_manual_datas()
"""

import os
import sys


def patch_dll_discovery():
    """
    Replace PyInstaller's isolated-subprocess DLL discovery with a no-op.
    Must be called BEFORE Analysis() in the spec file.
    """
    import PyInstaller.building.build_main as _build_main

    _original = _build_main.find_binary_dependencies

    def _safe_find_binary_dependencies(binaries, import_packages, symlink_suppression_patterns):
        print(f"[pyinstaller_helpers] Skipping isolated subprocess DLL discovery "
              f"({len(import_packages)} packages)")
        return _original(binaries, [], symlink_suppression_patterns)

    _build_main.find_binary_dependencies = _safe_find_binary_dependencies


def _find_site_packages_with(package_name):
    """Find the site-packages directory that contains a given package."""
    import site
    paths = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)
    for p in paths:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, package_name)):
            return p
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def collect_manual_binaries():
    """
    Collect binaries that would normally be discovered by the isolated
    subprocess DLL discovery phase. Call this and add results to binaries=[].
    """
    from PyInstaller.utils.hooks import collect_dynamic_libs, collect_delvewheel_libs_directory

    binaries = []

    # MNN native engine
    sp = _find_site_packages_with('MNN')
    if sp:
        mnn_pyd = os.path.join(sp, '_mnncengine.cp312-win_amd64.pyd')
        if os.path.exists(mnn_pyd):
            binaries.append((mnn_pyd, '.'))
            print(f"[pyinstaller_helpers] Including MNN native engine")

    # MediaPipe native binaries (opencv_world3410.dll, _framework_bindings.pyd)
    try:
        mp_bins = collect_dynamic_libs('mediapipe')
        if mp_bins:
            binaries += mp_bins
            print(f"[pyinstaller_helpers] Including mediapipe binaries: "
                  f"{[os.path.basename(b[0]) for b in mp_bins]}")
    except Exception:
        pass

    # Delvewheel .libs directories (numpy, pandas, scipy)
    for pkg in ['numpy', 'pandas', 'scipy']:
        try:
            _datas, _bins = collect_delvewheel_libs_directory(pkg)
            if _bins:
                binaries += _bins
                print(f"[pyinstaller_helpers] Including {pkg}.libs")
        except Exception:
            pass

    return binaries


def collect_manual_datas():
    """
    Collect data files that need manual inclusion (MNN package).
    Call this and add results to datas=[].
    """
    datas = []

    # MNN Python package
    sp = _find_site_packages_with('MNN')
    if sp:
        mnn_pkg = os.path.join(sp, 'MNN')
        if os.path.isdir(mnn_pkg):
            datas.append((mnn_pkg, 'MNN'))
            print(f"[pyinstaller_helpers] Including MNN package")

    # Delvewheel .libs data files
    from PyInstaller.utils.hooks import collect_delvewheel_libs_directory
    for pkg in ['numpy', 'pandas', 'scipy']:
        try:
            _datas, _bins = collect_delvewheel_libs_directory(pkg)
            if _datas:
                datas += _datas
        except Exception:
            pass

    return datas
