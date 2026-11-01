@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

:: ============================================================
:: CrowdGPT GUI bootstrap installer — Windows
::
:: Flow:
::   1. Ensure git (winget if needed)
::   2. Clone / sync repo
::   3. Find logo files, pass to installer.py
::   4. Get Python 3.11 (uv, python-build-standalone)
::   5. Installer venv + PyQt6 WebEngine + qtpy
::   6. Verify Qt backend imports
::   7. Launch installer.py (GUI or headless)
:: ============================================================

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
set "REPO_DIR=CrowdGPT"
set "PASS_ARGS="
set "LOGO_LIGHT="
set "LOGO_DARK="

:: ── Arg parsing ──
:parse
if "%~1"=="" goto parse_done
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h"     goto usage
if /i "%~1"=="--dir" (
    if "%~2"=="" ( echo --dir requires a path & exit /b 1 )
    set "REPO_DIR=%~2"
    shift & shift & goto parse
)
if /i "%~1"=="--backend" (
    if "%~2"=="" ( echo --backend requires a value & exit /b 1 )
    set "PASS_ARGS=!PASS_ARGS! --backend "%~2""
    shift & shift & goto parse
)
if /i "%~1"=="--no-launch" (
    set "PASS_ARGS=!PASS_ARGS! --no-launch"
    shift & goto parse
)
if /i "%~1"=="--no-integrate" (
    set "PASS_ARGS=!PASS_ARGS! --no-integrate"
    shift & goto parse
)
if /i "%~1"=="--headless" (
    set "PASS_ARGS=!PASS_ARGS! --headless"
    shift & goto parse
)
if /i "%~1"=="--debug" (
    set "DEBUG=1"
    shift & goto parse
)
echo Unknown argument: %1
echo Use --help for usage
exit /b 1

:usage
echo CrowdGPT installer (Windows GUI bootstrap)
echo.
echo Options:
echo   --backend ^<name^>   Force: nvidia ^| amd ^| intel ^| cpu
echo   --dir ^<path^>       Directory to install into (default: .\CrowdGPT)
echo   --no-launch        Install only, don't start the client
echo   --no-integrate     Skip desktop / command integration
echo   --headless         Skip GUI, use text-mode installer
echo   --debug            Verbose output
exit /b 0

:parse_done

echo.
echo ============================================================
echo               CrowdGPT Installer (GUI)
echo ============================================================
echo.

:: ── 1. git ──
where git >nul 2>&1
if not errorlevel 1 goto git_ok
if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
    goto git_ok
)
where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found. Install git from https://git-scm.com/download/win
    pause
    exit /b 1
)
echo Installing git via winget ...
winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
    goto git_ok
)
echo ERROR: git install failed
pause
exit /b 1
:git_ok
for /f "tokens=3" %%v in ('git --version') do echo git: %%v

:: ── 2. clone/sync repo ──
if exist "client.py" if exist ".git" (
    set "REPO_DIR=."
    goto have_repo
)
if exist "%REPO_DIR%\.git" goto have_repo
if exist "%REPO_DIR%" (
    echo ERROR: %REPO_DIR% exists but is not a git repo
    pause
    exit /b 1
)
echo Cloning %REPO_URL% into %REPO_DIR% ...
git clone "%REPO_URL%" "%REPO_DIR%"
if errorlevel 1 (
    echo ERROR: git clone failed
    pause
    exit /b 1
)

:have_repo
cd /d "%REPO_DIR%"
if not exist ".git" (
    echo ERROR: %CD% is not a git repository
    pause
    exit /b 1
)

for /f "tokens=*" %%b in ('git symbolic-ref --quiet --short HEAD 2^>nul') do set "BRANCH=%%b"
if not defined BRANCH set "BRANCH=main"
echo Syncing to origin/!BRANCH! ...
git fetch --all --prune --tags
if errorlevel 1 ( echo ERROR: git fetch failed & pause & exit /b 1 )
git reset --hard origin/!BRANCH!
if errorlevel 1 ( echo ERROR: git reset failed & pause & exit /b 1 )
for /f "tokens=*" %%c in ('git log -1 --format^="%%h %%s"') do echo Synced to %%c

:: ── 3. installer.py present? ──
if not exist "installer.py" (
    echo ERROR: installer.py not found in %CD%
    echo        The repo must contain installer.py
    pause
    exit /b 1
)
findstr /C:"--logo-dark" installer.py >nul
if errorlevel 1 (
    echo ERROR: installer.py is outdated ^(no --logo-dark flag^)
    echo        Pull the latest installer.py and rerun
    pause
    exit /b 1
)

:: ── 4. Find logo ──
if exist "docs\logo-app.svg"   set "LOGO_LIGHT=%CD%\docs\logo-app.svg"
if not defined LOGO_LIGHT if exist "docs\logo-app.png"   set "LOGO_LIGHT=%CD%\docs\logo-app.png"
if not defined LOGO_LIGHT if exist "docs\logo-black.svg" set "LOGO_LIGHT=%CD%\docs\logo-black.svg"
if exist "docs\logo-app.svg"   set "LOGO_DARK=%CD%\docs\logo-app.svg"
if not defined LOGO_DARK if exist "docs\logo-app.png"    set "LOGO_DARK=%CD%\docs\logo-app.png"
if not defined LOGO_DARK if exist "docs\logo-white.svg"  set "LOGO_DARK=%CD%\docs\logo-white.svg"

if defined LOGO_LIGHT (
    echo Logo: !LOGO_LIGHT:%CD%\=!
) else (
    echo WARNING: no logo under docs\ — installer will show 'C' fallback
)

:: ── 5. uv ──
where uv >nul 2>&1
if not errorlevel 1 goto uv_ok
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    goto uv_ok
)
if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    goto uv_ok
)
echo Installing uv ...
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    goto uv_ok
)
if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    goto uv_ok
)
echo ERROR: uv install failed
pause
exit /b 1
:uv_ok
for /f "tokens=2" %%v in ('uv --version') do echo uv: %%v

:: ── 6. Python 3.11 ──
echo Fetching Python 3.11 ...
uv python install 3.11 >nul 2>&1
if errorlevel 1 (
    echo ERROR: failed to install Python 3.11 via uv
    pause
    exit /b 1
)

:: ── 7. Installer venv + WebView ──
set "INSTALLER_VENV=%CD%\.installer-venv"
echo Creating installer venv at %INSTALLER_VENV% ...
if exist "%INSTALLER_VENV%" rmdir /s /q "%INSTALLER_VENV%"
uv venv --python 3.11 --clear "%INSTALLER_VENV%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: failed to create installer venv
    pause
    exit /b 1
)

echo Downloading WebView runtime ^(PyQt6 + WebEngine + qtpy, ~120 MB^)...
uv pip install --python "%INSTALLER_VENV%\Scripts\python.exe" ^
    pywebview "pywebview[qt]" qtpy pyqt6 pyqt6-webengine
if errorlevel 1 (
    echo ERROR: failed to install WebView dependencies
    pause
    exit /b 1
)

:: ── 8. Verify Qt backend ──
"%INSTALLER_VENV%\Scripts\python.exe" -c "from qtpy import QtCore; from PyQt6 import QtWidgets, QtWebEngineWidgets; import webview.platforms.qt" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Qt WebEngine could not load — falling back to headless
    set "PASS_ARGS=!PASS_ARGS! --headless"
) else (
    echo Qt WebEngine OK
)

:: ── 9. Launch ──
echo.
echo Launching installer.py ...
echo.

"%INSTALLER_VENV%\Scripts\python.exe" "installer.py" ^
    --repo-dir "%CD%" ^
    --logo "!LOGO_LIGHT!" ^
    --logo-dark "!LOGO_DARK!" ^
    !PASS_ARGS!
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo Installer exited with code %RC%
    echo Re-run with --headless for a text log:
    echo   install.bat --headless
)

pause
exit /b %RC%
