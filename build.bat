@echo off
chcp 65001 >nul
echo ============================================
echo   PointCloud Downsampler - Build Installer
echo ============================================
echo.

set BUILD_VENV=.buildvenv
set PYTHON_EXE=%BUILD_VENV%\Scripts\python.exe
set PYINSTALLER_EXE=%BUILD_VENV%\Scripts\pyinstaller.exe

if not exist "%PYTHON_EXE%" (
    echo [0/3] Creating clean build venv...
    py -3.11 -m venv "%BUILD_VENV%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create build venv!
        pause
        exit /b
    )
)

for /f "delims=" %%i in ('%PYTHON_EXE% -c "import version; print(version.APP_NAME)"') do set APP_NAME=%%i
for /f "delims=" %%i in ('%PYTHON_EXE% -c "import version; print(version.APP_VERSION)"') do set APP_VERSION=%%i
for /f "delims=" %%i in ('%PYTHON_EXE% -c "import version; print(version.APP_RELEASE_DATE)"') do set APP_RELEASE_DATE=%%i

set DIST_ROOT=dist\releases\v%APP_VERSION%
set WORK_ROOT=build\releases\v%APP_VERSION%
set PACKAGE_DIR=%DIST_ROOT%\PointCloudDownsampler
set VERSION_FILE=%PACKAGE_DIR%\VERSION.txt

echo [INFO] App:     %APP_NAME%
echo [INFO] Version: %APP_VERSION%
echo [INFO] Release: %APP_RELEASE_DATE%
echo [INFO] Output:  %PACKAGE_DIR%
echo.

echo [1/3] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to upgrade pip!
    pause
    exit /b
)

echo.
echo [2/3] Installing requirements...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install requirements!
    pause
    exit /b
)

echo.
echo [3/3] Running PyInstaller...
"%PYINSTALLER_EXE%" --clean --noconfirm --distpath "%DIST_ROOT%" --workpath "%WORK_ROOT%" PointCloudDownsampler.spec

echo.
if errorlevel 1 (
    echo [ERROR] Build failed. Check the logs.
) else (
    > "%VERSION_FILE%" (
        echo Name: %APP_NAME%
        echo Version: %APP_VERSION%
        echo Release-Date: %APP_RELEASE_DATE%
        echo Build-Time: %DATE% %TIME%
    )
    echo [OK] Build completed!
    echo   Output: %PACKAGE_DIR%
    echo   EXE:    %PACKAGE_DIR%\PointCloudDownsampler.exe
    echo   VER:    %APP_VERSION%
    echo   META:   %VERSION_FILE%
)
echo.
pause
