@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT universal installer / launcher for Windows
REM Detects Python + Git + GPU backend, installs the matching
REM PyTorch wheel, sets up a `crowdgpt` command and a Start Menu
REM shortcut, then launches client.py.
REM
REM Usage:
REM   install.bat [--backend nvidia|amd|intel|cpu]
REM               [--dir PATH] [--no-launch] [--no-integrate]
REM ============================================================

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
set "REPO_DIR=CrowdGPT"
set "PYTHON_EXE="
set "VENV_DIR="
set "BACKEND="
set "TORCH_INDEX="
set "TORCH_EXTRA="
set "WHEEL_NOTE="
set "LAUNCH=1"
set "INTEGRATE=1"
set "BACKEND_OVERRIDE="

REM ---------- arg parsing ----------
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--backend" (
    set "BACKEND_OVERRIDE=%~2"
    shift & shift
    goto parse_args
)
if /I "%~1"=="--dir" (
    set "REPO_DIR=%~2"
    shift & shift
    goto parse_args
)
if /I "%~1"=="--no-launch" (
    set "LAUNCH=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-integrate" (
    set "INTEGRATE=0"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h" goto show_help
echo [ERROR] Unknown argument: %~1
echo         Run install.bat --help for usage.
exit /b 1

:show_help
echo.
echo CrowdGPT installer (Windows)
echo.
echo Options:
echo   --backend ^<name^>   Force backend: nvidia ^| amd ^| intel ^| cpu
echo   --dir ^<path^>       Directory to clone into (default: .\CrowdGPT)
echo   --no-launch        Install only, don't launch client.py
echo   --no-integrate     Skip Start Menu ^& PATH integration
echo   --help             Show this message
echo.
exit /b 0

:args_done

echo.
echo ============================================================
echo                 CrowdGPT Installer
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 1. Locate Python
REM ------------------------------------------------------------
where python >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python') do (
        set "PYTHON_EXE=%%P"
        goto :python_found
    )
)

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    goto :python_found
)

echo [INFO] Python not found. Installing uv, then the latest stable Python...

where uv >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if not errorlevel 1 (
        echo [INFO] Installing uv with winget...
        winget install --id=astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
        if errorlevel 1 goto :fatal
        set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv;%PATH%"
    ) else (
        echo [INFO] Installing uv via official installer...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if errorlevel 1 goto :fatal
        set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv;%PATH%"
    )
)

if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%LOCALAPPDATA%\uv\uv.exe"        set "PATH=%LOCALAPPDATA%\uv;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv installed but not on PATH. Restart terminal and rerun.
    goto :fatal
)

echo [INFO] Downloading the latest stable Python...
uv python install --default
if errorlevel 1 goto :fatal

for /f "delims=" %%P in ('uv python find') do (
    set "PYTHON_EXE=%%P"
    goto :python_found
)

:python_found
echo [OK] Python: !PYTHON_EXE!

REM ------------------------------------------------------------
REM 2. Locate / install Git
REM ------------------------------------------------------------
where git >nul 2>nul
if errorlevel 1 (
    echo [INFO] Git not found. Installing Git for Windows...

    where winget >nul 2>nul
    if not errorlevel 1 (
        winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
        if errorlevel 1 goto :git_fallback
    ) else (
        goto :git_fallback
    )

    set "PATH=%ProgramFiles%\Git\cmd;%ProgramFiles%\Git\bin;%PATH%"
)

where git >nul 2>nul
if errorlevel 1 goto :git_fallback
goto :git_found

:git_fallback
echo [INFO] winget unavailable/failed. Downloading Git for Windows directly...
set "GIT_INSTALLER=%TEMP%\Git-64-bit.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='https://github.com/git-for-windows/git/releases/latest/download/Git-64-bit.exe'; Invoke-WebRequest -Uri $u -OutFile '%GIT_INSTALLER%'"
if errorlevel 1 goto :fatal

echo [INFO] Installing Git silently...
"%GIT_INSTALLER%" /VERYSILENT /NORESTART
if errorlevel 1 goto :fatal
del /q "%GIT_INSTALLER%" >nul 2>nul
set "PATH=%ProgramFiles%\Git\cmd;%ProgramFiles%\Git\bin;%PATH%"

where git >nul 2>nul
if errorlevel 1 goto :fatal

:git_found
echo [OK] Git found.

REM ------------------------------------------------------------
REM 3. Clone / update repo
REM ------------------------------------------------------------
if exist ".git" if exist "client.py" (
    set "REPO_DIR=."
    goto :repo_ready
)

if exist "%REPO_DIR%\.git" (
    echo [INFO] CrowdGPT already exists. Updating...
    git -C "%REPO_DIR%" pull --ff-only
    if errorlevel 1 echo [WARN] git pull failed; continuing with existing checkout.
    goto :repo_ready
)

if exist "%REPO_DIR%" (
    echo [ERROR] "%REPO_DIR%" exists but is not a Git repository.
    echo         Move/delete it and rerun.
    goto :fatal
)

echo [INFO] Cloning CrowdGPT...
git clone "%REPO_URL%" "%REPO_DIR%"
if errorlevel 1 goto :fatal

:repo_ready
cd /d "%REPO_DIR%"
if not exist "client.py" (
    echo [ERROR] client.py not found in the CrowdGPT repository.
    goto :fatal
)

REM ------------------------------------------------------------
REM 4. Hardware detection
REM ------------------------------------------------------------
echo.
echo [INFO] Detecting hardware backend...

call :detect_backend
if errorlevel 1 goto :fatal

if not "!BACKEND_OVERRIDE!"=="" (
    if /I "!BACKEND_OVERRIDE!"=="nvidia" (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: nvidia/cu124"
    ) else if /I "!BACKEND_OVERRIDE!"=="amd" (
        set "BACKEND=amd"
        set "TORCH_INDEX="
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: amd/directml"
    ) else if /I "!BACKEND_OVERRIDE!"=="intel" (
        set "BACKEND=intel"
        set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
        set "TORCH_EXTRA=https://pypi.org/simple"
        set "WHEEL_NOTE=forced: intel/xpu"
    ) else if /I "!BACKEND_OVERRIDE!"=="cpu" (
        set "BACKEND=cpu"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: cpu"
    ) else (
        echo [ERROR] --backend must be one of: nvidia ^| amd ^| intel ^| cpu
        goto :fatal
    )
)

echo.
echo ============================================================
echo                       Backend
echo ============================================================
echo   Detected:    !BACKEND!
if defined TORCH_INDEX echo   Torch index: !TORCH_INDEX!
if defined TORCH_EXTRA echo   Extra index: !TORCH_EXTRA!
echo   Note:        !WHEEL_NOTE!
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 5. Install paths
REM ------------------------------------------------------------
set "APP_DIR=%LOCALAPPDATA%\CrowdGPT"
set "BIN_DIR=%APP_DIR%"
set "START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

REM ------------------------------------------------------------
REM 6. Build venv in the source dir
REM ------------------------------------------------------------
set "BUILD_VENV=%CD%\.venv"

echo [INFO] Creating build venv...
if exist "%BUILD_VENV%" rmdir /s /q "%BUILD_VENV%" >nul 2>nul
"!PYTHON_EXE!" -m venv "%BUILD_VENV%"
if errorlevel 1 (
    echo [WARN] venv creation failed. Trying uv...
    where uv >nul 2>nul
    if errorlevel 1 goto :fatal
    uv venv "%BUILD_VENV%"
    if errorlevel 1 goto :fatal
)

set "BUILD_PY=%BUILD_VENV%\Scripts\python.exe"
if not exist "%BUILD_PY%" (
    echo [ERROR] Build venv Python missing.
    goto :fatal
)

echo [INFO] Upgrading pip...
"%BUILD_PY%" -m pip install --upgrade pip >nul
if errorlevel 1 goto :fatal

REM ------------------------------------------------------------
REM 7. Install PyTorch from correct index
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                   Installing PyTorch
echo ============================================================

"%BUILD_PY%" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>nul

if /I "!BACKEND!"=="amd" (
    echo [INFO] pip install torch-directml
    "%BUILD_PY%" -m pip install torch-directml
    if errorlevel 1 goto :fatal
) else (
    if defined TORCH_EXTRA (
        echo [INFO] pip install torch torchvision torchaudio --index-url !TORCH_INDEX! --extra-index-url !TORCH_EXTRA!
        "%BUILD_PY%" -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX! --extra-index-url !TORCH_EXTRA!
    ) else (
        echo [INFO] pip install torch torchvision torchaudio --index-url !TORCH_INDEX!
        "%BUILD_PY%" -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX!
    )
    if errorlevel 1 goto :fatal
)

REM ------------------------------------------------------------
REM 8. Install project requirements (torch lines stripped)
REM ------------------------------------------------------------
if exist "requirements.txt" (
    echo.
    echo ============================================================
    echo               Installing project dependencies
    echo ============================================================
    set "TMP_REQ=%TEMP%\crowdgpt-req-%RANDOM%-%RANDOM%.txt"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Get-Content -LiteralPath 'requirements.txt' | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|torch-directml)(\s|[<>=!~]|$)' } | Set-Content -LiteralPath '%TMP_REQ%'"
    if errorlevel 1 goto :fatal

    for %%A in ("%TMP_REQ%") do set "TMP_SIZE=%%~zA"
    if "!TMP_SIZE!"=="0" (
        echo [WARN] requirements.txt contained only torch packages; nothing else to install.
    ) else (
        "%BUILD_PY%" -m pip install -r "%TMP_REQ%"
        if errorlevel 1 goto :fatal
    )
    del /q "%TMP_REQ%" >nul 2>nul
) else (
    echo [WARN] requirements.txt not found; skipping project deps.
)

REM ------------------------------------------------------------
REM 9. GUI backend check (WebView2 runtime on Windows)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo           GUI backend for pywebview (WebView2)
echo ============================================================

REM WebView2 ships with Windows 11 and Windows 10 21H2+.
REM For older builds we warn and offer the installer URL.
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
if errorlevel 1 (
    reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
)
if errorlevel 1 (
    echo [WARN] Microsoft Edge WebView2 Runtime not detected.
    echo        pywebview needs it to display the GUI window.
    echo        Windows 11 and recent Windows 10 already include it.
    echo        If the client fails to open a window, install it from:
    echo          https://developer.microsoft.com/microsoft-edge/webview2/
    echo.
) else (
    echo [OK] WebView2 Runtime present.
)

REM ------------------------------------------------------------
REM 10. Verify the venv
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                    Verifying install
echo ============================================================
"%BUILD_PY%" -c "import sys, torch; print('  Python:', sys.version.split()[0]); print('  Torch:  ', torch.__version__); print('  CUDA:   ', torch.cuda.is_available(), end=''); print('  (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else '')"
if errorlevel 1 echo [WARN] Verification failed, but install may still work.

"%BUILD_PY%" -c "import webview; print('  pywebview: OK')" 2>nul
if errorlevel 1 echo [WARN] pywebview import failed.

REM ------------------------------------------------------------
REM 11. Desktop + command integration
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    echo.
    echo ============================================================
    echo                Installing desktop integration
    echo ============================================================
    echo [INFO] Install location: !APP_DIR!

    if not exist "!APP_DIR!" mkdir "!APP_DIR!"

    REM Copy repo (skip .git, venv, caches)
    echo [INFO] Copying app files...
    robocopy "%CD%" "!APP_DIR!" /E /XD .git .venv __pycache__ checkpoints pending_uploads /NFL /NDL /NJH /NJS /NC /NS >nul
    if errorlevel 8 (
        echo [ERROR] Failed to copy files to !APP_DIR!
        goto :fatal
    )

    REM Move build venv into app dir
    if exist "!APP_DIR!\.venv" rmdir /s /q "!APP_DIR!\.venv" >nul 2>nul
    move "!BUILD_VENV!" "!APP_DIR!\.venv" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to move venv into app dir.
        goto :fatal
    )

    set "APP_PY=!APP_DIR!\.venv\Scripts\python.exe"
    if not exist "!APP_PY!" (
        echo [ERROR] Installed venv Python missing at !APP_PY!
        goto :fatal
    )

    REM Rewrite shebangs/paths that still point at the old venv location
    echo [INFO] Rewriting venv paths...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$old = '%CD%\.venv'.Replace('\','\\');" ^
        "$new = '!APP_DIR!\.venv'.Replace('\','\\');" ^
        "$root = '!APP_DIR!\.venv';" ^
        "Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {" ^
        "  if ($_.Length -gt 200000) { return }" ^
        "  try { $c = [System.IO.File]::ReadAllText($_.FullName) } catch { return }" ^
        "  if ($c -like ('*' + $old + '*')) {" ^
        "    $c = $c.Replace($old, $new);" ^
        "    try { [System.IO.File]::WriteAllText($_.FullName, $c) } catch {}" ^
        "  }" ^
        "}"

    REM ---- CMD shim ----
    > "!BIN_DIR!\crowdgpt.cmd" echo @echo off
    >>"!BIN_DIR!\crowdgpt.cmd" echo "!APP_PY!" "!APP_DIR!\client.py" %%*

    REM ---- GUI launcher (no console window) ----
    > "!BIN_DIR!\crowdgpt-gui.vbs" echo Set WshShell = CreateObject("WScript.Shell")
    >>"!BIN_DIR!\crowdgpt-gui.vbs" echo WshShell.Run """!APP_PY!"" ""!APP_DIR!\client.py""", 0, False

    REM ---- Add to user PATH (HKCU, via PowerShell, no setx truncation) ----
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$p = [Environment]::GetEnvironmentVariable('Path', 'User');" ^
        "if ($null -eq $p) { $p = '' }" ^
        "if ($p -notlike '*!BIN_DIR!*') {" ^
        "  $new = if ($p) { $p + ';!BIN_DIR!' } else { '!BIN_DIR!' };" ^
        "  [Environment]::SetEnvironmentVariable('Path', $new, 'User');" ^
        "  Write-Host 'Added to user PATH';" ^
        "}"

    REM ---- Icon path ----
    set "APP_ICON="
    if exist "!APP_DIR!\docs\logo-app.ico" set "APP_ICON=!APP_DIR!\docs\logo-app.ico"
    if not defined APP_ICON if exist "!APP_DIR!\docs\logo-app.svg" set "APP_ICON=!APP_DIR!\docs\logo-app.svg"

    REM ---- Start Menu shortcut ----
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ws = New-Object -ComObject WScript.Shell;" ^
        "$sc = $ws.CreateShortcut('!START_MENU!\CrowdGPT.lnk');" ^
        "$sc.TargetPath = '!APP_DIR!\crowdgpt-gui.vbs';" ^
        "$sc.WorkingDirectory = '!APP_DIR!';" ^
        "if ('!APP_ICON!') { $sc.IconLocation = '!APP_ICON!' }" ^
        "$sc.Description = 'Decentralized AI Training Client';" ^
        "$sc.Save()"

    REM ---- Add/Remove Programs registry entry ----
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$k = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT';" ^
        "New-Item -Path $k -Force | Out-Null;" ^
        "Set-ItemProperty -Path $k -Name 'DisplayName' -Value 'CrowdGPT';" ^
        "Set-ItemProperty -Path $k -Name 'DisplayVersion' -Value '0.5';" ^
        "Set-ItemProperty -Path $k -Name 'Publisher' -Value 'CrowdGPT Project';" ^
        "Set-ItemProperty -Path $k -Name 'InstallLocation' -Value '!APP_DIR!';" ^
        "Set-ItemProperty -Path $k -Name 'UninstallString' -Value ('\"' + '!APP_DIR!\uninstall.bat' + '\"');" ^
        "Set-ItemProperty -Path $k -Name 'NoModify' -Value 1 -Type DWord;" ^
        "Set-ItemProperty -Path $k -Name 'NoRepair' -Value 1 -Type DWord;"

    REM ---- Uninstaller ----
    > "!APP_DIR!\uninstall.bat" echo @echo off
    >>"!APP_DIR!\uninstall.bat" echo setlocal
    >>"!APP_DIR!\uninstall.bat" echo echo Uninstalling CrowdGPT...
    >>"!APP_DIR!\uninstall.bat" echo del /q "!START_MENU!\CrowdGPT.lnk" 2^>nul
    >>"!APP_DIR!\uninstall.bat" echo reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT" /f ^>nul 2^>nul
    >>"!APP_DIR!\uninstall.bat" echo powershell -NoProfile -Command "$p = [Environment]::GetEnvironmentVariable('Path','User'); if ($p) { $p = ($p -split ';' ^| Where-Object { $_ -ne '!BIN_DIR!' }) -join ';'; [Environment]::SetEnvironmentVariable('Path', $p, 'User') }"
    >>"!APP_DIR!\uninstall.bat" echo cd /d "%%TEMP%%"
    >>"!APP_DIR!\uninstall.bat" echo rmdir /s /q "!APP_DIR!"
    >>"!APP_DIR!\uninstall.bat" echo echo Done. Log out and back in for PATH changes to take effect.
    >>"!APP_DIR!\uninstall.bat" echo pause

    echo.
    echo [OK] Installed command: crowdgpt
    echo [OK] Start Menu shortcut: CrowdGPT
    echo [OK] Uninstall via: Settings ^> Apps ^> CrowdGPT
    echo.
    echo [NOTE] A new terminal is required for the "crowdgpt" command to work.
)

REM ------------------------------------------------------------
REM 12. Launch
REM ------------------------------------------------------------
if "!LAUNCH!"=="1" (
    echo Launching client.py...
    echo.
    if "!INTEGRATE!"=="1" (
        "!APP_PY!" "!APP_DIR!\client.py"
        set "EXITCODE=!ERRORLEVEL!"
    ) else (
        "!BUILD_PY!" "%CD%\client.py"
        set "EXITCODE=!ERRORLEVEL!"
    )
    echo.
    echo CrowdGPT exited with code !EXITCODE!.
    exit /b !EXITCODE!
) else (
    if "!INTEGRATE!"=="1" (
        echo To launch later:  crowdgpt
    ) else (
        echo To launch later:  "!BUILD_PY!" "%CD%\client.py"
    )
    exit /b 0
)

REM ============================================================
REM Subroutine: detect_backend
REM ============================================================
:detect_backend

REM --- NVIDIA first ---
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    set "NVIDIA_DRIVER="
    for /f "delims=" %%V in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul') do (
        if not defined NVIDIA_DRIVER set "NVIDIA_DRIVER=%%V"
    )
    if defined NVIDIA_DRIVER (
        for /f "tokens=1 delims= " %%A in ("!NVIDIA_DRIVER!") do set "NVIDIA_DRIVER=%%A"
    )

    if not defined NVIDIA_DRIVER (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "WHEEL_NOTE=NVIDIA GPU - could not read driver, defaulting to cu124"
        exit /b 0
    )

    set "DRIVER_MAJOR="
    for /f "tokens=1 delims=." %%M in ("!NVIDIA_DRIVER!") do set "DRIVER_MAJOR=%%M"

    if not defined DRIVER_MAJOR (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! - defaulting to cu124"
        exit /b 0
    )

    if !DRIVER_MAJOR! GEQ 525 (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -^> CUDA 12.4 wheel"
    ) else if !DRIVER_MAJOR! GEQ 470 (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
        set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -^> CUDA 11.8 wheel (older driver)"
    ) else (
        echo [ERROR] NVIDIA driver !NVIDIA_DRIVER! is too old for any PyTorch wheel.
        echo         Update your GPU driver and rerun.
        exit /b 1
    )
    exit /b 0
)

REM --- Intel Arc / XPU ---
powershell -NoProfile -Command "try { $g = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; if ($g -match 'Intel.*Arc') { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 (
    set "BACKEND=intel"
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
    set "TORCH_EXTRA=https://pypi.org/simple"
    set "WHEEL_NOTE=Intel Arc detected -^> XPU wheel"
    exit /b 0
)

REM --- AMD discrete GPU -> DirectML ---
powershell -NoProfile -Command "try { $g = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterCompatibility; if ($g -match 'AMD|Advanced Micro Devices') { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 (
    set "BACKEND=amd"
    set "TORCH_INDEX="
    set "TORCH_EXTRA="
    set "WHEEL_NOTE=AMD GPU detected -^> DirectML (torch-directml). ROCm is Linux-only; use WSL2 for the ROCm path."
    exit /b 0
)

REM --- CPU fallback ---
set "BACKEND=cpu"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_EXTRA="
set "WHEEL_NOTE=No usable GPU detected -^> CPU wheel (training will be very slow)"
exit /b 0

:fatal
echo.
echo ============================================================
echo [ERROR] Installation failed.
echo ============================================================
pause
exit /b 1
