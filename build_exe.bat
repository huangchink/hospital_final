@echo off
echo ========================================
echo Building VA_center_opt.exe, calibration.exe, and replayer.exe
echo ========================================
echo.

:: Check if PyInstaller is installed
python -m PyInstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    python -m pip install pyinstaller
    if %errorlevel% neq 0 (
        echo Failed to install PyInstaller
        pause
        exit /b 1
    )
)

:: Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Build VA_center_opt
echo.
echo ========================================
echo [1/3] Building VA_center_opt.exe ...
echo ========================================
python -m PyInstaller --clean -y VA_center_opt.spec
set "BUILD1_RESULT=%errorlevel%"

:: Build calibration
echo.
echo ========================================
echo [2/3] Building calibration.exe ...
echo ========================================
python -m PyInstaller --clean -y calibration.spec
set "BUILD2_RESULT=%errorlevel%"

:: Build replayer
echo.
echo ========================================
echo [3/3] Building replayer.exe ...
echo ========================================
python -m PyInstaller --clean -y replayer.spec
set "BUILD3_RESULT=%errorlevel%"

:: Check results
set "FAILED=0"

if %BUILD1_RESULT% neq 0 (
    echo.
    echo [FAILED] VA_center_opt build FAILED!
    set "FAILED=1"
)

if %BUILD2_RESULT% neq 0 (
    echo.
    echo [FAILED] calibration build FAILED!
    set "FAILED=1"
)

if %BUILD3_RESULT% neq 0 (
    echo.
    echo [FAILED] replayer build FAILED!
    set "FAILED=1"
)

if "%FAILED%"=="1" (
    echo.
    echo ========================================
    echo Some builds FAILED! See errors above.
    echo ========================================
    pause
    exit /b 1
)

:: Create debug launchers
echo.
echo Creating debug launchers...

(
    echo @echo off
    echo echo ========================================
    echo echo Running VA_center_opt.exe in DEBUG mode
    echo echo ========================================
    echo echo.
    echo echo Console will stay open to show errors.
    echo echo.
    echo echo ----------------------------------------
    echo echo.
    echo VA_center_opt.exe
    echo set EXITCODE=%%errorlevel%%
    echo echo.
    echo echo ----------------------------------------
    echo echo Program exited with code: %%EXITCODE%%
    echo echo.
    echo if %%EXITCODE%% neq 0 ^(
    echo     echo ERROR: The program crashed!
    echo     echo Please read the error messages above.
    echo ^)
    echo echo.
    echo pause
) > dist\VA_center_opt\run_debug.bat

(
    echo @echo off
    echo echo ========================================
    echo echo Running calibration.exe in DEBUG mode
    echo echo ========================================
    echo echo.
    echo calibration.exe
    echo set EXITCODE=%%errorlevel%%
    echo echo.
    echo echo ----------------------------------------
    echo echo Program exited with code: %%EXITCODE%%
    echo if %%EXITCODE%% neq 0 ^(
    echo     echo ERROR: The program crashed!
    echo ^)
    echo echo.
    echo pause
) > dist\calibration\run_debug.bat

(
    echo @echo off
    echo echo ========================================
    echo echo Running replayer.exe in DEBUG mode
    echo echo ========================================
    echo echo.
    echo replayer.exe
    echo set EXITCODE=%%errorlevel%%
    echo echo.
    echo echo ----------------------------------------
    echo echo Program exited with code: %%EXITCODE%%
    echo if %%EXITCODE%% neq 0 ^(
    echo     echo ERROR: The program crashed!
    echo ^)
    echo echo.
    echo pause
) > dist\replayer\run_debug.bat

echo.
echo ========================================
echo Build complete!
echo ========================================
echo.
echo VA_center_opt: dist\VA_center_opt\VA_center_opt.exe
echo               dist\VA_center_opt\run_debug.bat
echo.
echo Calibration:   dist\calibration\calibration.exe
echo               dist\calibration\run_debug.bat
echo.
echo Replayer:      dist\replayer\replayer.exe
echo               dist\replayer\run_debug.bat
echo.
echo You can copy each "dist\..." folder to another
echo Windows computer and run the .exe directly.
echo.

pause
