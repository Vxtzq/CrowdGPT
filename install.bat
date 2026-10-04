@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT universal installer / launcher for Windows
REM Detects Python + Git + GPU backend, installs the matching
REM PyTorch wheel from the correct index, then launches client.py.
REM
REM Usage:
REM   install.bat [--backend nvidia|amd|intel|cpu]
REM               [--dir PATH] [--no-launch]
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
REM 3. Clone / reuse CrowdGPT
REM ------------------------------------------------------------
if exist ".git" if exist "client.py" (
    set "REPO_DIR=."
    goto :repo_ready
)

if exist "%REPO_DIR%\.git" (
    echo [INFO] CrowdGPT already exists. Updating it...
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
REM
REM Windows backend matrix:
REM   NVIDIA  -> CUDA (cu124 if driver ^>= 525, else cu118)
REM   Intel Arc -> XPU (Intel's official Windows path for Arc)
REM   AMD/Intel iGPU -> DirectML (torch-directml) OR CPU
REM   Nothing -> CPU
REM
REM ROCm does not exist on Windows; AMD users who want ROCm must
REM use WSL2 with an Ubuntu install, then run the Linux installer.
REM

echo.
echo [INFO] Detecting hardware backend...

call :detect_backend
if errorlevel 1 goto :fatal

REM Apply --backend override if given
if not "%BACKEND_OVERRIDE%"=="" (
    if /I "%BACKEND_OVERRIDE%"=="nvidia" (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: nvidia/cu124"
    ) else if /I "%BACKEND_OVERRIDE%"=="amd" (
        set "BACKEND=amd"
        set "TORCH_INDEX="
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: amd/directml"
    ) else if /I "%BACKEND_OVERRIDE%"=="intel" (
        set "BACKEND=intel"
        set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
        set "TORCH_EXTRA=https://pypi.org/simple"
        set "WHEEL_NOTE=forced: intel/xpu"
    ) else if /I "%BACKEND_OVERRIDE%"=="cpu" (
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
REM 5. Create virtual environment
REM ------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    "!PYTHON_EXE!" -m venv .venv
    if errorlevel 1 (
        echo [WARN] venv creation failed. Trying uv...
        where uv >nul 2>nul
        if errorlevel 1 goto :fatal
        uv venv .venv
        if errorlevel 1 goto :fatal
    )
)

set "VENV_DIR=%CD%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual-environment Python was not created.
    goto :fatal
)

echo [INFO] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul
if errorlevel 1 goto :fatal

REM ------------------------------------------------------------
REM 6. Install PyTorch from the correct index
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                   Installing PyTorch
echo ============================================================

REM Remove any pre-existing torch so we never end up with a
REM mixed install (e.g. torch==2.4.0+cpu alongside +cu124).
"%PYTHON_EXE%" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>nul

if /I "!BACKEND!"=="amd" (
    REM DirectML path: separate package, no index URL.
    echo [INFO] pip install torch-directml
    "%PYTHON_EXE%" -m pip install torch-directml
    if errorlevel 1 goto :fatal
) else (
    REM Standard torch wheel from a hardware-specific index.
    if defined TORCH_EXTRA (
        echo [INFO] pip install torch torchvision torchaudio --index-url !TORCH_INDEX! --extra-index-url !TORCH_EXTRA!
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX! --extra-index-url !TORCH_EXTRA!
    ) else (
        echo [INFO] pip install torch torchvision torchaudio --index-url !TORCH_INDEX!
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX!
    )
    if errorlevel 1 goto :fatal
)

REM ------------------------------------------------------------
REM 7. Install the rest of requirements.txt (torch lines stripped)
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
        "%PYTHON_EXE%" -m pip install -r "%TMP_REQ%"
        if errorlevel 1 goto :fatal
    )
    del /q "%TMP_REQ%" >nul 2>nul
) else (
    echo [WARN] requirements.txt not found; skipping project dependencies.
)

REM ------------------------------------------------------------
REM 8. Verify
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                    Verifying install
echo ============================================================
"%PYTHON_EXE%" -c "import sys, torch; print('  Python:', sys.version.split()[0]); print('  Torch:  ', torch.__version__); print('  CUDA:   ', torch.cuda.is_available(), end=''); print('  (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''); print('  DirectML:', 'yes' if 'torch_directml' in sys.modules else 'no')"
if errorlevel 1 (
    echo [WARN] Verification failed, but install may still work.
)
REM Quick DirectML check separately (import is heavyweight, do it lazily)
if /I "!BACKEND!"=="amd" (
    "%PYTHON_EXE%" -c "try:\n    import torch_directml as dml\n    print('  DirectML device:', dml.device_name(0))\nexcept Exception as e:\n    print('  DirectML not available:', e)" 2>nul
)

REM ------------------------------------------------------------
REM 9. Launch
REM ------------------------------------------------------------
echo.
echo ============================================================
echo             Installation complete  OK
echo ============================================================
echo   Backend: !BACKEND!
echo   Venv:    .venv
echo.

if "%LAUNCH%"=="1" (
    echo Launching client.py...
    echo.
    "%PYTHON_EXE%" client.py
    set "EXITCODE=%ERRORLEVEL%"
    echo.
    echo CrowdGPT exited with code !EXITCODE!.
    exit /b !EXITCODE!
) else (
    echo To launch later:  "%PYTHON_EXE%" client.py
    exit /b 0
)

REM ============================================================
REM Subroutine: detect_backend
REM Sets BACKEND, TORCH_INDEX, TORCH_EXTRA, WHEEL_NOTE
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
        REM Trim whitespace
        for /f "tokens=1 delims= " %%A in ("!NVIDIA_DRIVER!") do set "NVIDIA_DRIVER=%%A"
    )

    if not defined NVIDIA_DRIVER (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "WHEEL_NOTE=NVIDIA GPU - could not read driver, defaulting to cu124"
        exit /b 0
    )

    REM Parse major version
    set "DRIVER_MAJOR="
    for /f "tokens=1 delims=." %%M in ("!NVIDIA_DRIVER!") do set "DRIVER_MAJOR=%%M"

    if not defined DRIVER_MAJOR (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! - defaulting to cu124"
        exit /b 0
    )

    REM Windows NVIDIA drivers: 525+ is CUDA 12.x capable
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

REM --- Intel Arc / XPU (Intel's official Windows path) ---
REM Heuristic: look for Intel(R) Arc in the adapter string.
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

REM --- Anything else with no usable accelerator -> CPU ---
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
