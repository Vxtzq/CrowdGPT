@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM CrowdGPT GUI bootstrap installer - Windows (native WebView)
REM
REM Uses pywebview's Windows-native backend (EdgeChromium via
REM pythonnet / WebView2 runtime). No PyQt6, no 220MB download.
REM
REM ASCII-only. CRLF line endings. Do not add emoji/em-dash.
REM ============================================================

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
if not defined REPO_DIR set "REPO_DIR=%USERPROFILE%\CrowdGPT"
set "PASS_ARGS="
set "LOGO_LIGHT="
set "LOGO_DARK="

:parse
if "%~1"=="" goto parse_done
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h"     goto usage
if /i "%~1"=="--dir" (
    if "%~2"=="" ( echo --dir requires a path & exit /b 1 )
    set "REPO_DIR=%~2" & shift & shift & goto parse
)
if /i "%~1"=="--backend" (
    if "%~2"=="" ( echo --backend requires a value & exit /b 1 )
    set PASS_ARGS=!PASS_ARGS! --backend "%~2" & shift & shift & goto parse
)
if /i "%~1"=="--no-launch"    ( set PASS_ARGS=!PASS_ARGS! --no-launch & shift & goto parse )
if /i "%~1"=="--no-integrate" ( set PASS_ARGS=!PASS_ARGS! --no-integrate & shift & goto parse )
if /i "%~1"=="--headless"     ( set PASS_ARGS=!PASS_ARGS! --headless & shift & goto parse )
if /i "%~1"=="--debug"        ( set DEBUG=1 & shift & goto parse )
echo Unknown argument: %1
exit /b 1

:usage
echo CrowdGPT installer - Windows
echo.
echo   --backend NAME     nvidia ^| amd ^| intel ^| cpu
echo   --dir PATH         default: %%USERPROFILE%%\CrowdGPT
echo   --no-launch        install only
echo   --no-integrate     skip Start Menu / shortcuts
echo   --headless         text mode (no GUI)
echo   --debug            verbose
exit /b 0
:parse_done

echo.
echo ============================================================
echo                CrowdGPT Installer - Windows
echo ============================================================
echo.

REM ---- 1. git ----
where git >nul 2>&1
if not errorlevel 1 goto git_ok
if exist "%ProgramFiles%\Git\cmd\git.exe" ( set "PATH=%ProgramFiles%\Git\cmd;%PATH%" & goto git_ok )
where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found. Install from https://git-scm.com/download/win
    pause & exit /b 1
)
echo Installing git via winget ...
winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
if exist "%ProgramFiles%\Git\cmd\git.exe" ( set "PATH=%ProgramFiles%\Git\cmd;%PATH%" & goto git_ok )
echo ERROR: git install failed & pause & exit /b 1
:git_ok
for /f "tokens=3" %%v in ('git --version') do echo git: %%v

REM ---- 2. clone/sync ----
if exist "client.py" if exist ".git" ( set "REPO_DIR=." & goto have_repo )
if exist "%REPO_DIR%\.git" goto have_repo
if exist "%REPO_DIR%" ( echo ERROR: %REPO_DIR% is not a git repo & pause & exit /b 1 )
echo Cloning %REPO_URL% into %REPO_DIR% ...
git clone "%REPO_URL%" "%REPO_DIR%"
if errorlevel 1 ( echo ERROR: git clone failed & pause & exit /b 1 )
:have_repo
cd /d "%REPO_DIR%"
if not exist ".git" ( echo ERROR: %CD% is not a git repo & pause & exit /b 1 )
for /f "tokens=*" %%b in ('git symbolic-ref --quiet --short HEAD 2^>nul') do set "BRANCH=%%b"
if not defined BRANCH set "BRANCH=main"
echo Syncing to origin/!BRANCH! ...
git fetch --all --prune --tags
if errorlevel 1 ( echo ERROR: git fetch failed & pause & exit /b 1 )
git reset --hard origin/!BRANCH!
if errorlevel 1 ( echo ERROR: git reset failed & pause & exit /b 1 )
for /f "tokens=*" %%c in ('git log -1 --format^="%%h %%s"') do echo Synced to %%c

REM ---- 3. installer.py ----
if not exist "installer.py" (
    echo ERROR: installer.py not found in %CD% & pause & exit /b 1
)
findstr /C:"--logo-dark" installer.py >nul
if errorlevel 1 (
    echo ERROR: installer.py is outdated ^(no --logo-dark^) & pause & exit /b 1
)

REM ---- 4. logo ----
if exist "docs\logo-app.svg"   set LOGO_LIGHT=%CD%\docs\logo-app.svg
if not defined LOGO_LIGHT if exist "docs\logo-app.png"   set LOGO_LIGHT=%CD%\docs\logo-app.png
if not defined LOGO_LIGHT if exist "docs\logo-black.svg" set LOGO_LIGHT=%CD%\docs\logo-black.svg
if exist "docs\logo-app.svg"   set LOGO_DARK=%CD%\docs\logo-app.svg
if not defined LOGO_DARK if exist "docs\logo-app.png"    set LOGO_DARK=%CD%\docs\logo-app.png
if not defined LOGO_DARK if exist "docs\logo-white.svg"  set LOGO_DARK=%CD%\docs\logo-white.svg
if defined LOGO_LIGHT ( echo Logo: !LOGO_LIGHT:%CD%\=! ) else ( echo WARNING: no logo under docs\ )

REM ---- 5. uv ----
where uv >nul 2>&1
if not errorlevel 1 goto uv_ok
if exist "%USERPROFILE%\.local\bin\uv.exe" ( set "PATH=%USERPROFILE%\.local\bin;%PATH%" & goto uv_ok )
if exist "%USERPROFILE%\.cargo\bin\uv.exe" ( set "PATH=%USERPROFILE%\.cargo\bin;%PATH%" & goto uv_ok )
echo Installing uv ...
powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if exist "%USERPROFILE%\.local\bin\uv.exe" ( set "PATH=%USERPROFILE%\.local\bin;%PATH%" & goto uv_ok )
if exist "%USERPROFILE%\.cargo\bin\uv.exe" ( set "PATH=%USERPROFILE%\.cargo\bin;%PATH%" & goto uv_ok )
echo ERROR: uv install failed & pause & exit /b 1
:uv_ok
for /f "tokens=2" %%v in ('uv --version') do echo uv: %%v

REM ---- 6. Python 3.11 ----
echo Fetching Python 3.11 ...
uv python install 3.11 >nul 2>&1
if errorlevel 1 ( echo ERROR: failed to install Python 3.11 & pause & exit /b 1 )

REM ---- 7. Installer venv ----
set "INSTALLER_VENV=%CD%\.installer-venv"
echo Creating installer venv at %INSTALLER_VENV% ...
if exist "%INSTALLER_VENV%" rmdir /s /q "%INSTALLER_VENV%"
uv venv --python 3.11 --clear "%INSTALLER_VENV%" >nul 2>&1
if errorlevel 1 ( echo ERROR: failed to create installer venv & pause & exit /b 1 )

REM ---- 8. WebView: EdgeChromium first, NO Qt ----
echo Installing pywebview + pythonnet ^(native WebView2 / EdgeChromium^)...
uv pip install --python "%INSTALLER_VENV%\Scripts\python.exe" pywebview pythonnet
if errorlevel 1 ( echo ERROR: pywebview install failed & pause & exit /b 1 )

REM Verify EdgeChromium backend loads (create QApplication not needed for this path)
echo Verifying EdgeChromium backend ...
set "WEBVIEW_OK=0"
"%INSTALLER_VENV%\Scripts\python.exe" -c "import webview; from webview.platforms import edgechromium; print('EDGE_OK')" 2>nul | findstr /C:"EDGE_OK" >nul
if not errorlevel 1 (
    set "WEBVIEW_OK=1"
    echo EdgeChromium OK
) else (
    echo EdgeChromium backend did not load - checking WebView2 runtime ...
)

REM Check if WebView2 runtime is present (registry)
if "%WEBVIEW_OK%"=="0" (
    reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
    if errorlevel 1 (
        reg query "HKCU\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
        if errorlevel 1 (
            echo WARNING: WebView2 runtime not found
            where winget >nul 2>&1
            if not errorlevel 1 (
                echo Installing WebView2 runtime via winget ...
                winget install --id Microsoft.EdgeWebView2Runtime -e --silent --accept-package-agreements --accept-source-agreements
                "%INSTALLER_VENV%\Scripts\python.exe" -c "from webview.platforms import edgechromium; print('EDGE_OK')" 2>nul | findstr /C:"EDGE_OK" >nul
                if not errorlevel 1 ( set "WEBVIEW_OK=1" & echo EdgeChromium OK after WebView2 install )
            )
        ) else (
            set "WEBVIEW_OK=1" & echo WebView2 runtime found in registry
        )
    ) else (
        set "WEBVIEW_OK=1" & echo WebView2 runtime found in registry
    )
)

if "%WEBVIEW_OK%"=="0" (
    echo.
    echo WARNING: No native WebView backend available.
    echo Falling back to headless text mode.
    echo To fix, install WebView2 runtime:
    echo   winget install Microsoft.EdgeWebView2Runtime
    set PASS_ARGS=!PASS_ARGS! --headless
)

REM ---- 9. Launch ----
echo.
echo Launching installer.py ...
echo.

set "LAUNCH_ARGS=--repo-dir "%CD%""
if defined LOGO_LIGHT set LAUNCH_ARGS=!LAUNCH_ARGS! --logo "!LOGO_LIGHT!"
if defined LOGO_DARK  set LAUNCH_ARGS=!LAUNCH_ARGS! --logo-dark "!LOGO_DARK!"

"%INSTALLER_VENV%\Scripts\python.exe" "installer.py" !LAUNCH_ARGS! !PASS_ARGS!
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo Installer exited with code %RC%
    echo Re-run with --headless for a text log:
    echo   install.bat --headless
)

pause
exit /b %RC%
