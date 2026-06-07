@echo off
REM ============================================================================
REM Build the 3 release apps as DEBUG (console-visible) onedir exes into dist\.
REM   - VA_center_opt.exe  (VA/VF stimulus + Sol/webcam capture)
REM   - calibration.exe    (webcam SVR calibration tool)
REM   - replayer.exe       (session replay + review labeling)
REM Specs: VA_center_opt.spec / calibration.spec / replayer.spec (all console=True).
REM Post-build staging (defaults, image folders, output dirs, run_debug.bat launchers)
REM is done by stage_release.py.
REM Run from the repo root:  build_exe.bat
REM ============================================================================
echo ========================================
echo Building VA_center_opt, calibration, replayer (debug/console build)
echo ========================================
echo.

python -m PyInstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    python -m pip install pyinstaller || (echo Failed to install PyInstaller & pause & exit /b 1)
)

echo Cleaning previous build\ and dist\ ...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo.

echo [1/3] Building VA_center_opt.exe ...
python -m PyInstaller --clean -y VA_center_opt.spec
set "R1=%errorlevel%"

echo [2/3] Building calibration.exe ...
python -m PyInstaller --clean -y calibration.spec
set "R2=%errorlevel%"

echo [3/3] Building replayer.exe ...
python -m PyInstaller --clean -y replayer.spec
set "R3=%errorlevel%"

if %R1% neq 0 ( echo [FAILED] VA_center_opt build & goto :fail )
if %R2% neq 0 ( echo [FAILED] calibration build & goto :fail )
if %R3% neq 0 ( echo [FAILED] replayer build & goto :fail )

echo.
echo Staging runtime data + debug launchers ...
python stage_release.py
if %errorlevel% neq 0 ( echo [FAILED] staging & goto :fail )

echo.
echo ========================================
echo Build complete. Release folders (copy any one to the target PC):
echo   dist\VA_center_opt\   ^> VA_center_opt.exe  (or run_debug.bat)
echo   dist\calibration\     ^> calibration.exe    (or run_debug.bat)
echo   dist\replayer\        ^> replayer.exe       (or run_debug.bat)
echo ========================================
echo.
pause
exit /b 0

:fail
echo.
echo ========================================
echo BUILD FAILED - see errors above.
echo ========================================
pause
exit /b 1
