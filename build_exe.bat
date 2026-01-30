@echo off
echo ========================================
echo Building VA_center_opt.exe
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

:: Build the executable
echo.
echo Building executable...
python -m PyInstaller --clean VA_center_opt.spec

if %errorlevel% equ 0 (
    echo.
    echo Creating debug launcher...
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

    echo.
    echo ========================================
    echo Build successful!
    echo ========================================
    echo Executable location: dist\VA_center_opt\VA_center_opt.exe
    echo Debug launcher:      dist\VA_center_opt\run_debug.bat
    echo.
    echo To see error messages if it crashes:
    echo   1. Use run_debug.bat
    echo   2. Or run from command prompt
    echo.
    echo You can now copy the entire "dist\VA_center_opt" folder
    echo to another Windows computer and run VA_center_opt.exe
    echo.
) else (
    echo.
    echo ========================================
    echo Build FAILED!
    echo ========================================
    echo Please check the error messages above
    echo.
)

pause
