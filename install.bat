@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT Universal Installer for Windows
REM v10.3
REM  Changes vs v10.2:
REM  - transformers 5.x needs torch >= 2.6, but the cu121 wheel index
REM    stops at torch 2.5.1. NVIDIA now uses cu126 (torch 2.6+ up to
REM    the newest release); the old-driver fallback stays cu118
REM    (torch 2.7.1 max, still >= 2.6).
REM  - Step 7b threshold raised from torch 2.5 to 2.6.
REM  - DirectML: torch-directml pins torch 2.4.1, which cannot satisfy
REM    torch>=2.6 in requirements.txt, so torch lines are left out of
REM    the requirements install on the AMD backend only.
REM
REM  Changes in v10.2:
REM  - Project requirements are now installed directly with
REM    pip install -r requirements.txt. The PowerShell/regex torch
REM    filter is gone (it could fail silently and skip ALL deps,
REM    including transformers). torch is already installed at this
REM    point, so "torch>=x" lines are satisfied and left alone.
REM  - A failure while installing requirements is now fatal
REM    instead of being skipped with a warning.
REM  - Step 7b now guarantees transformers is installed even if
REM    something upstream went wrong.
REM
REM  Changes in v10.1:
REM  - uv / Git no longer require a terminal restart. After any
REM    install the PATH is re-read from the registry and merged
REM    into the running session, and known install dirs are probed
REM    as a fallback (winget Links, ~/.local/bin, etc).
REM  - uv: official installer with a fixed install dir first,
REM    winget as fallback.
REM  - Git direct-download fallback resolves the real asset
REM    through the GitHub API.
REM  - DirectML verify snippets contain no '%' characters.
REM  - Uninstaller: PATH cleanup and self-delete fixed.
REM  - Non-integrated install: venv is created inside the repo.
REM  - Start Menu icon: only .ico is used (.svg is not valid).
REM ============================================================

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
set "REPO_DIR=CrowdGPT"
set "BACKEND="
set "TORCH_INDEX="
set "WHEEL_NOTE="
set "LAUNCH=1"
set "INTEGRATE=1"
set "BACKEND_OVERRIDE="
set "LOCAL_INSTALLER_DIR=%~dp0"
set "AMD_GPU_NAME="
set "ROCM_GFX="
set "FRESH=1"
set "PY_VER=3.11"

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
if /I "%~1"=="--no-fresh" (
    set "FRESH=0"
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
echo CrowdGPT installer - Windows
echo.
echo Options:
echo   --backend ^<name^>   Force backend: nvidia ^| amd ^| rocm ^| intel ^| cpu
echo   --dir ^<path^>       Directory to clone into. Default: .\CrowdGPT
echo   --no-launch        Install only, don't launch client.py
echo   --no-integrate     Skip Start Menu and PATH integration
echo   --no-fresh         Keep existing install (default: wipe and reinstall)
echo   --help             Show this message
echo.
echo Notes:
echo   - AMD automatic selection: recognized GPU -> ROCm; otherwise DirectML.
echo   - Experimental ROCm: --backend rocm (requires recognized AMD GPU).
echo   - Requires a compatible AMD driver and TheRock PyTorch wheels.
echo   - Intel Arc uses the XPU wheel. NVIDIA uses CUDA.
echo.
exit /b 0

:args_done

echo.
echo ============================================================
echo                 CrowdGPT Installer  (v10.3)
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 0. Paths
REM ------------------------------------------------------------
set "APP_DIR=%LOCALAPPDATA%\CrowdGPT"
set "BIN_DIR=%APP_DIR%"
set "START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

REM VENV_DIR / APP_PY for the non-integrated case are set after
REM the repo is located (see SRC_DIR below).
if "!INTEGRATE!"=="1" (
    set "VENV_DIR=!APP_DIR!\.venv"
    set "APP_PY=!APP_DIR!\.venv\Scripts\python.exe"
)

REM ------------------------------------------------------------
REM 0a. Explicit ROCm: validate BEFORE the previous install is removed.
if /I "!BACKEND_OVERRIDE!"=="rocm" (
    call :choose_rocm_gpu
    if not defined ROCM_GFX (
        echo [ERROR] No recognized AMD GPU target for Windows ROCm.
        echo [ERROR] Existing installation was not modified.
        exit /b 1
    )
)

REM 0b. Wipe previous install (preserve caches)
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" if "!FRESH!"=="1" if exist "!APP_DIR!" (
    echo [INFO] Removing previous installation at !APP_DIR!...
    echo        ^(caches in checkpoints\ and pending_uploads\ are preserved^)

    set "KEEP_DIR=%TEMP%\crowdgpt-keep-%RANDOM%"
    if not exist "!KEEP_DIR!" mkdir "!KEEP_DIR!" >nul 2>nul

    if exist "!APP_DIR!\checkpoints" (
        robocopy "!APP_DIR!\checkpoints" "!KEEP_DIR!\checkpoints" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )
    if exist "!APP_DIR!\pending_uploads" (
        robocopy "!APP_DIR!\pending_uploads" "!KEEP_DIR!\pending_uploads" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )

    rmdir /s /q "!APP_DIR!" >nul 2>nul
    if exist "!APP_DIR!" (
        echo [WARN] Could not fully remove !APP_DIR! ^(files may be locked^).
        echo        Close any running CrowdGPT windows and rerun for a clean install.
    )

    mkdir "!APP_DIR!" >nul 2>nul
    if exist "!KEEP_DIR!\checkpoints" (
        robocopy "!KEEP_DIR!\checkpoints" "!APP_DIR!\checkpoints" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )
    if exist "!KEEP_DIR!\pending_uploads" (
        robocopy "!KEEP_DIR!\pending_uploads" "!APP_DIR!\pending_uploads" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )
    rmdir /s /q "!KEEP_DIR!" >nul 2>nul
    echo [OK] Previous install cleared.
    echo.
)

REM ------------------------------------------------------------
REM 1. uv (no terminal restart needed)
REM ------------------------------------------------------------
call :ensure_uv
if errorlevel 1 (
    echo [ERROR] Could not install uv automatically.
    echo         Install it manually from:
    echo         https://docs.astral.sh/uv/getting-started/installation/
    goto :fatal
)
echo [OK] uv is ready.

REM ------------------------------------------------------------
REM 2. Git (no terminal restart needed)
REM ------------------------------------------------------------
call :ensure_git
if errorlevel 1 (
    echo [ERROR] Could not install Git automatically.
    echo         Install it from https://git-scm.com/download/win and rerun.
    goto :fatal
)
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
set "SRC_DIR=%CD%"

if not "!INTEGRATE!"=="1" (
    set "VENV_DIR=!SRC_DIR!\.venv"
    set "APP_PY=!SRC_DIR!\.venv\Scripts\python.exe"
)

REM ------------------------------------------------------------
REM 4. Hardware detection
REM ------------------------------------------------------------
echo.
echo [INFO] Detecting hardware backend...

call :detect_backend
if errorlevel 1 goto :fatal

if not "!BACKEND_OVERRIDE!"=="" call :apply_override
if errorlevel 1 goto :fatal

echo.
echo ============================================================
echo                       Backend
echo ============================================================
echo   Detected:    !BACKEND!
if defined TORCH_INDEX echo   Torch index: !TORCH_INDEX!
echo   Note:        !WHEEL_NOTE!
echo   Python:      !PY_VER!
echo ============================================================
echo.

if /I "!BACKEND!"=="amd" call :note_amd
if /I "!BACKEND!"=="rocm" (
    echo [NOTE] Experimental Windows ROCm via TheRock, !ROCM_GFX!.
    echo [NOTE] This is NOT a guarantee that CrowdGPT training will work.
    echo [NOTE] Failed ROCm install / GPU smoke test switches to DirectML.
    REM Never overwrite a newer repository client.py with an old local overlay by default.
    if /I "!CROWDGPT_USE_LEGACY_ROCM_OVERLAY!"=="1" (
        if exist "!LOCAL_INSTALLER_DIR!client_rocm.py" (
            echo [WARN] Applying explicitly enabled legacy client_rocm.py overlay.
            copy /Y "!LOCAL_INSTALLER_DIR!client_rocm.py" "!SRC_DIR!\client.py" >nul
            if errorlevel 1 goto :fatal
        ) else (
            echo [INFO] Legacy overlay requested but client_rocm.py was not found.
        )
    ) else (
        echo [INFO] Using repository client.py; legacy ROCm overlay disabled by default.
    )
)

REM ------------------------------------------------------------
REM 5. Create venv
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    if not exist "!APP_DIR!" mkdir "!APP_DIR!"
)

echo [INFO] Creating venv with Python !PY_VER! at !VENV_DIR!...
if exist "!VENV_DIR!" rmdir /s /q "!VENV_DIR!" >nul 2>nul

uv venv --python !PY_VER! --seed "!VENV_DIR!"
if errorlevel 1 (
    echo [ERROR] Failed to create venv with uv.
    goto :fatal
)

if not exist "!APP_PY!" (
    echo [ERROR] venv Python missing at !APP_PY!
    goto :fatal
)

"!APP_PY!" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [WARN] pip missing from venv, bootstrapping with ensurepip...
    "!APP_PY!" -m ensurepip --upgrade >nul 2>nul
    "!APP_PY!" -m pip --version >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not bootstrap pip into the venv.
        goto :fatal
    )
)

echo [INFO] Upgrading pip...
"!APP_PY!" -m pip install --upgrade pip >nul 2>nul

REM ------------------------------------------------------------
REM 6. Install PyTorch
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                   Installing PyTorch
echo ============================================================

"!APP_PY!" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>nul

if /I "!BACKEND!"=="nvidia" goto :pytorch_nvidia
if /I "!BACKEND!"=="amd"    goto :pytorch_amd
if /I "!BACKEND!"=="rocm"   goto :pytorch_rocm
if /I "!BACKEND!"=="intel"  goto :pytorch_intel
goto :pytorch_cpu

REM Install only the architecture selected by detect_amd_gpu.ps1.
:pytorch_rocm
echo [INFO] Installing native Windows PyTorch ROCm through AMD TheRock...
echo [INFO] ROCm 10.1.0 stable, GPU target !ROCM_GFX!.
"!APP_PY!" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-!ROCM_GFX!]==2.13.0+rocm10.1.0"
if errorlevel 1 (
    echo [WARN] ROCm wheel installation failed.
    goto :rocm_to_directml
)
REM Verify HIP runtime, discover discrete Radeon among devices, then launch a kernel.
"!APP_PY!" -c "import torch; assert torch.version.hip, 'Not a HIP build'; assert torch.cuda.is_available(), 'No HIP devices'; gs=[(i, torch.cuda.get_device_name(i), int(torch.cuda.get_device_properties(i).total_memory)) for i in range(torch.cuda.device_count())]; ds=[g for g in gs if 'radeon rx' in g[1].lower()]; assert ds, 'No discrete Radeon RX visible via HIP'; g=max(ds, key=lambda v:v[2]); torch.cuda.set_device(g[0]); print('  ROCm GPU:',g[1]); x=torch.ones((256,256),device='cuda:'+str(g[0])); y=x@x; torch.cuda.synchronize(g[0]); assert float(y[0,0].cpu())==256.0, 'Wrong matrix result'; print('  ROCm matrix test OK:',float(y[0,0].cpu()))"
if errorlevel 1 (
    echo [WARN] ROCm runtime test failed on the detected AMD card.
    goto :rocm_to_directml
)
goto :pytorch_done

:rocm_to_directml
echo.
echo [WARN] Switching from experimental ROCm to AMD DirectML.
echo [WARN] The final Python environment will not be a HIP build, so ROCm will appear disabled in the GUI.
echo [WARN] See the ROCm wheel-install or GPU smoke-test error above to learn why ROCm was rejected.
REM Recreate the isolated venv to remove incompatible/partially installed ROCm.
if exist "!VENV_DIR!" rmdir /s /q "!VENV_DIR!" >nul 2>nul
if exist "!VENV_DIR!" (
    echo [ERROR] Could not remove failed ROCm environment.
    goto :fatal
)
uv venv --python !PY_VER! --seed "!VENV_DIR!"
if errorlevel 1 goto :fatal
if not exist "!APP_PY!" goto :fatal
set "BACKEND=amd"
set "TORCH_INDEX="
set "WHEEL_NOTE=ROCm failed; DirectML fallback"
goto :pytorch_amd

:pytorch_nvidia
echo [INFO] Trying PyTorch CUDA 12.6 cu126...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] cu126 failed. Trying CUDA 11.8 cu118...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

:pytorch_amd
echo [INFO] Installing PyTorch DirectML ^(torch-directml^)...
echo        torch-directml pins torch==2.4.1. transformers will be
echo        pinned to 4.x automatically below to match.
"!APP_PY!" -m pip install torch-directml
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] torch-directml install failed. Falling back to CPU.
goto :pytorch_cpu

:pytorch_intel
echo [INFO] Installing PyTorch XPU...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

:pytorch_cpu
set "BACKEND=cpu"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "WHEEL_NOTE=CPU-only fallback"
echo.
echo [WARN] Installing CPU-only PyTorch. Training will be very slow.
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fatal

:pytorch_done

echo.
echo [INFO] Installed torch version:
"!APP_PY!" -c "import torch; print('       ' + torch.__version__)"

REM ------------------------------------------------------------
REM 7. Install project requirements
REM    torch is already installed above; "torch>=2.6.0" in
REM    requirements.txt is satisfied by it, so pip leaves it alone.
REM    (If requirements.txt ever pins torch with "==", pip would
REM    replace the GPU build - keep it as ">=".)
REM ------------------------------------------------------------
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in !SRC_DIR!
    goto :fatal
)

echo.
echo ============================================================
echo               Installing project dependencies
echo ============================================================

set "REQ_FILE=!SRC_DIR!\requirements.txt"
set "TMP_REQ="
if /I "!BACKEND!"=="rocm" (
    echo [INFO] ROCm build: protecting AMD PyTorch wheel from requirements changes.
    set "TMP_REQ=%TEMP%\crowdgpt-req-%RANDOM%.txt"
    findstr /V /B /I /C:"torch>" /C:"torch=" /C:"torch<" /C:"torch~" /C:"torch " /C:"torchvision" /C:"torchaudio" /C:"torch-directml" "!SRC_DIR!\requirements.txt" > "!TMP_REQ!"
    set "REQ_FILE=!TMP_REQ!"
)
if /I "!BACKEND!"=="amd" (
    echo [INFO] DirectML build: leaving torch lines out ^(torch-directml pins its own torch^).
    set "TMP_REQ=%TEMP%\crowdgpt-req-%RANDOM%.txt"
    findstr /V /B /I /C:"torch>" /C:"torch=" /C:"torch<" /C:"torch~" /C:"torch " /C:"torchvision" /C:"torchaudio" /C:"torch-directml" "!SRC_DIR!\requirements.txt" > "!TMP_REQ!"
    set "REQ_FILE=!TMP_REQ!"
)

"!APP_PY!" -m pip install -r "!REQ_FILE!"
set "PIP_RC=!ERRORLEVEL!"
if defined TMP_REQ del "!TMP_REQ!" >nul 2>nul
if not "!PIP_RC!"=="0" (
    echo [ERROR] pip failed to install project requirements.
    goto :fatal
)

:reqs_done

REM ------------------------------------------------------------
REM 7b. transformers pin, based on installed torch version
REM     DirectML ships torch 2.4.1; transformers 5.x requires
REM     torch >= 2.6. Any torch < 2.6 gets transformers 4.x.
REM     Otherwise, make sure transformers is present at all.
REM ------------------------------------------------------------
"!APP_PY!" -c "import sys, torch; v = torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]) > 2 or (int(v[0]) == 2 and int(v[1]) >= 6)) else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo [INFO] torch ^< 2.6 detected; pinning transformers to 4.x...
    "!APP_PY!" -m pip install --upgrade "transformers>=4.45,<5.0"
    if errorlevel 1 (
        echo [ERROR] Failed to pin transformers to 4.x.
        goto :fatal
    )
    echo [OK] transformers pinned to 4.x
) else (
    echo.
    echo [INFO] torch ^>= 2.6; using latest transformers
    "!APP_PY!" -c "import transformers" >nul 2>nul
    if errorlevel 1 (
        echo [WARN] transformers missing, installing it directly...
        "!APP_PY!" -m pip install transformers
        if errorlevel 1 (
            echo [ERROR] Failed to install transformers.
            goto :fatal
        )
    )
)

REM ------------------------------------------------------------
REM 8. GUI backend check - WebView2
REM ------------------------------------------------------------
echo.
echo ============================================================
echo           GUI backend for pywebview - WebView2
echo ============================================================

set "WV2_FOUND=0"
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
if not errorlevel 1 set "WV2_FOUND=1"
if "!WV2_FOUND!"=="0" (
    reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
    if not errorlevel 1 set "WV2_FOUND=1"
)
if "!WV2_FOUND!"=="1" (
    echo [OK] WebView2 Runtime present.
) else (
    echo [WARN] Microsoft Edge WebView2 Runtime not detected.
    echo        pywebview needs it to display the GUI window.
    echo        Windows 11 and recent Windows 10 already include it.
    echo        If the client fails to open a window, install it from:
    echo          https://developer.microsoft.com/microsoft-edge/webview2/
    echo.
)

REM ------------------------------------------------------------
REM 9. Verify
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                    Verifying install
echo ============================================================
"!APP_PY!" -c "import sys, torch; print('  Python:', sys.version.split()[0]); print('  Torch:  ', torch.__version__)"
if errorlevel 1 echo [WARN] Torch import failed.

"!APP_PY!" -c "import transformers, torch; assert transformers.utils.is_torch_available(), 'transformers refuses to use installed torch'; print('  transformers:', transformers.__version__, '-> torch', torch.__version__, 'linked OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] transformers cannot use the installed torch.
    echo         Try: pip install "transformers>=4.45,<5.0"
    goto :fatal
)

if /I "!BACKEND!"=="amd" call :verify_directml
if /I "!BACKEND!"=="rocm" (
    "!APP_PY!" -c "import torch; assert torch.version.hip and torch.cuda.is_available(), 'ROCm unavailable after requirements'; gs=[(i,torch.cuda.get_device_name(i),int(torch.cuda.get_device_properties(i).total_memory)) for i in range(torch.cuda.device_count())]; ds=[g for g in gs if 'radeon rx' in g[1].lower()]; g=max(ds or gs,key=lambda x:x[2]); print('  HIP:',torch.version.hip); print('  ROCm GPU:',g[1]); assert ds, 'No discrete Radeon RX visible after requirements'"
    if errorlevel 1 (
        echo [ERROR] ROCm unavailable after dependencies installation.
        goto :fatal
    )
)

if /I "!BACKEND!"=="nvidia" (
    "!APP_PY!" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('  CUDA device:', torch.cuda.get_device_name(0))"
    if errorlevel 1 echo [WARN] CUDA wheel installed but no CUDA device is available.
)

"!APP_PY!" -c "import webview; print('  pywebview: OK')" 2>nul
if errorlevel 1 echo [WARN] pywebview import failed.

REM ------------------------------------------------------------
REM 10. Desktop integration
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    call :do_integration
    if errorlevel 1 goto :fatal
)

REM ------------------------------------------------------------
REM 11. Launch
REM ------------------------------------------------------------
if "!LAUNCH!"=="1" (
    echo Launching client.py...
    echo.
    if "!INTEGRATE!"=="1" (
        "!APP_PY!" "!APP_DIR!\client.py"
    ) else (
        "!APP_PY!" "!SRC_DIR!\client.py"
    )
    set "EXITCODE=!ERRORLEVEL!"
    echo.
    echo CrowdGPT exited with code !EXITCODE!.
    exit /b !EXITCODE!
) else (
    if "!INTEGRATE!"=="1" (
        echo To launch later:  Start Menu ^> CrowdGPT   ^(or "crowdgpt" in a new terminal^)
    ) else (
        echo To launch later:  "!APP_PY!" "!SRC_DIR!\client.py"
    )
    exit /b 0
)

REM ============================================================
REM Subroutine: refresh_path
REM   Re-read Machine + User PATH from the registry and merge it
REM   with the current session PATH (deduplicated). This is what
REM   makes freshly installed tools usable without a new terminal.
REM ============================================================
:refresh_path
set "RP_PS=%TEMP%\crowdgpt-path-%RANDOM%.ps1"
set "NEW_PATH="
> "!RP_PS!"  echo $m = [Environment]::GetEnvironmentVariable('Path', 'Machine')
>> "!RP_PS!" echo $u = [Environment]::GetEnvironmentVariable('Path', 'User')
>> "!RP_PS!" echo $all = ($m + ';' + $u + ';' + $env:Path) -split ';' ^| Where-Object { $_ } ^| Select-Object -Unique
>> "!RP_PS!" echo Write-Output ($all -join ';')
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "!RP_PS!"`) do set "NEW_PATH=%%P"
del "!RP_PS!" >nul 2>nul
if defined NEW_PATH set "PATH=!NEW_PATH!"
exit /b 0

REM ============================================================
REM Subroutine: probe_uv  /  probe_git
REM   Fallback: if the exe exists in a well-known dir, prepend it
REM   to the session PATH even if the registry was not updated.
REM ============================================================
:probe_uv
for %%D in ("%LOCALAPPDATA%\uv\bin" "%USERPROFILE%\.local\bin" "%LOCALAPPDATA%\Microsoft\WinGet\Links" "%USERPROFILE%\.cargo\bin") do (
    if exist "%%~D\uv.exe" set "PATH=%%~D;!PATH!"
)
exit /b 0

:probe_git
for %%D in ("%ProgramFiles%\Git\cmd" "%LOCALAPPDATA%\Programs\Git\cmd") do (
    if exist "%%~D\git.exe" set "PATH=%%~D;!PATH!"
)
exit /b 0

REM ============================================================
REM Subroutine: ensure_uv
REM   Returns 0 if uv is usable in THIS session afterwards.
REM ============================================================
:ensure_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0

call :probe_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0

echo [INFO] uv not found. Installing uv...

REM Method 1: official installer into a known directory
set "UV_INSTALL_DIR=%LOCALAPPDATA%\uv\bin"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; irm https://astral.sh/uv/install.ps1 | iex"
set "UV_INSTALL_DIR="
call :refresh_path
call :probe_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0

REM Method 2: winget
echo [WARN] Official installer did not make uv available. Trying winget...
where winget >nul 2>nul
if errorlevel 1 exit /b 1
winget install --id=astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
call :refresh_path
call :probe_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0
exit /b 1

REM ============================================================
REM Subroutine: ensure_git
REM   Returns 0 if git is usable in THIS session afterwards.
REM ============================================================
:ensure_git
where git >nul 2>nul
if not errorlevel 1 exit /b 0

call :probe_git
where git >nul 2>nul
if not errorlevel 1 exit /b 0

echo [INFO] Git not found. Installing Git for Windows...

where winget >nul 2>nul
if errorlevel 1 goto :git_direct
winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
call :refresh_path
call :probe_git
where git >nul 2>nul
if not errorlevel 1 exit /b 0
echo [WARN] winget did not make git available. Trying direct download...

:git_direct
set "GIT_PS=%TEMP%\crowdgpt-git-%RANDOM%.ps1"
set "GIT_INSTALLER=%TEMP%\Git-64-bit-%RANDOM%.exe"
> "!GIT_PS!"  echo [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
>> "!GIT_PS!" echo $ProgressPreference = 'SilentlyContinue'
>> "!GIT_PS!" echo $r = Invoke-RestMethod -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest' -Headers @{ 'User-Agent' = 'CrowdGPT-Installer' }
>> "!GIT_PS!" echo $a = $r.assets ^| Where-Object { $_.name -like 'Git-*-64-bit.exe' } ^| Select-Object -First 1
>> "!GIT_PS!" echo if (-not $a) { exit 1 }
>> "!GIT_PS!" echo Invoke-WebRequest -Uri $a.browser_download_url -OutFile '!GIT_INSTALLER!'
powershell -NoProfile -ExecutionPolicy Bypass -File "!GIT_PS!"
set "GIT_DL_RC=!ERRORLEVEL!"
del "!GIT_PS!" >nul 2>nul
if not "!GIT_DL_RC!"=="0" exit /b 1
if not exist "!GIT_INSTALLER!" exit /b 1

echo [INFO] Installing Git silently...
start "" /wait "!GIT_INSTALLER!" /VERYSILENT /NORESTART /SP- /SUPPRESSMSGBOXES
del /q "!GIT_INSTALLER!" >nul 2>nul

call :refresh_path
call :probe_git
where git >nul 2>nul
if not errorlevel 1 exit /b 0
exit /b 1

REM ============================================================
REM Subroutine: note_amd
REM ============================================================
:note_amd
echo [NOTE] AMD DirectML backend selected.
echo        torch-directml pins torch 2.4.1, so transformers gets
echo        pinned to 4.x automatically. Some model operations may
echo        not be implemented on DirectML.
echo.
echo [NOTE] If your system has both an iGPU and a dGPU, the client
echo        prefers the device whose name contains a model number
echo        (e.g. "RX 7800 XT" over "Radeon(TM) Graphics").
echo        To list DirectML devices:
echo          .venv\Scripts\python.exe -c "import torch_directml as d; print([d.device_name(i) for i in range(d.device_count())])"
echo.
exit /b 0

REM ============================================================
REM Subroutine: verify_directml
REM   NOTE: no '%' characters in the python snippets - cmd would
REM   treat them as variable expansions and corrupt the code.
REM ============================================================
:verify_directml
"!APP_PY!" -c "import torch_directml as d; [print('  DirectML[' + str(i) + ']: ' + str(d.device_name(i))) for i in range(d.device_count())]" 2>nul
if errorlevel 1 (
    echo [WARN] Could not enumerate DirectML devices.
    exit /b 0
)
"!APP_PY!" -c "import torch; import torch_directml as d; assert d.is_available(), 'no DirectML device'; ds=[(i,str(d.device_name(i))) for i in range(d.device_count())]; rx=[g for g in ds if 'radeon rx' in g[1].lower()]; i=(rx[0] if rx else ds[0])[0]; dev=d.device(i); a=torch.randn(64,64).to(dev); b=torch.randn(64,64).to(dev); c=(a@b).sum().cpu().item(); print('  DirectML['+str(i)+'] matmul OK, sum =',round(c,4))" 2>nul
if errorlevel 1 echo [WARN] DirectML device present but matmul failed. Some ops may be unsupported.
exit /b 0

REM ============================================================
REM Subroutine: do_integration
REM ============================================================
:do_integration
echo.
echo ============================================================
echo                Installing desktop integration
echo ============================================================
echo [INFO] Install location: !APP_DIR!

if not exist "!APP_DIR!" mkdir "!APP_DIR!"

echo [INFO] Copying app files (venv excluded; already in place)...
robocopy "!SRC_DIR!" "!APP_DIR!" /E /XD .git .venv __pycache__ checkpoints pending_uploads /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 (
    echo [ERROR] Failed to copy files to !APP_DIR!
    exit /b 1
)

if not exist "!APP_PY!" (
    echo [ERROR] Installed venv Python missing at !APP_PY!
    exit /b 1
)

REM ---- CMD shim ----
> "!BIN_DIR!\crowdgpt.cmd" echo @echo off
>>"!BIN_DIR!\crowdgpt.cmd" echo "!APP_PY!" "!APP_DIR!\client.py" %%*

REM ---- GUI launcher (no console window) ----
set "GUI_VBS=!BIN_DIR!\crowdgpt-gui.vbs"
> "!GUI_VBS!"  echo Set WshShell = CreateObject("WScript.Shell")
>> "!GUI_VBS!" echo WshShell.Run """!APP_PY!"" ""!APP_DIR!\client.py""", 0, False

REM ---- Add to user PATH (persistent, for future terminals) ----
set "ENV_PS=%TEMP%\crowdgpt-env-%RANDOM%.ps1"
> "!ENV_PS!"  echo $p = [Environment]::GetEnvironmentVariable('Path', 'User')
>> "!ENV_PS!" echo if ($null -eq $p) { $p = '' }
>> "!ENV_PS!" echo if ($p -notlike '*!BIN_DIR!*') {
>> "!ENV_PS!" echo   if ($p) { $new = $p + ';!BIN_DIR!' } else { $new = '!BIN_DIR!' }
>> "!ENV_PS!" echo   [Environment]::SetEnvironmentVariable('Path', $new, 'User')
>> "!ENV_PS!" echo   Write-Host 'Added to user PATH'
>> "!ENV_PS!" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "!ENV_PS!"
del "!ENV_PS!" >nul 2>nul

REM ---- Icon (.ico only; .lnk cannot use .svg) ----
set "APP_ICON="
if exist "!APP_DIR!\docs\logo-app.ico" set "APP_ICON=!APP_DIR!\docs\logo-app.ico"

REM ---- Start Menu shortcut ----
set "SC_PS=%TEMP%\crowdgpt-sc-%RANDOM%.ps1"
> "!SC_PS!"  echo $ws = New-Object -ComObject WScript.Shell
>> "!SC_PS!" echo $sc = $ws.CreateShortcut('!START_MENU!\CrowdGPT.lnk')
>> "!SC_PS!" echo $sc.TargetPath = '!GUI_VBS!'
>> "!SC_PS!" echo $sc.WorkingDirectory = '!APP_DIR!'
if defined APP_ICON >>"!SC_PS!" echo $sc.IconLocation = '!APP_ICON!'
>> "!SC_PS!" echo $sc.Description = 'Decentralized AI Training Client'
>> "!SC_PS!" echo $sc.Save()
powershell -NoProfile -ExecutionPolicy Bypass -File "!SC_PS!"
del "!SC_PS!" >nul 2>nul

REM ---- Add/Remove Programs entry ----
set "REG_PS=%TEMP%\crowdgpt-reg-%RANDOM%.ps1"
> "!REG_PS!"  echo $k = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT'
>> "!REG_PS!" echo New-Item -Path $k -Force ^| Out-Null
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayName'     -Value 'CrowdGPT'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayVersion'  -Value '0.10.3'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'Publisher'       -Value 'CrowdGPT Project'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'InstallLocation' -Value '!APP_DIR!'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'UninstallString' -Value ('"' + '!APP_DIR!\uninstall.bat' + '"')
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'NoModify'        -Value 1 -Type DWord
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'NoRepair'        -Value 1 -Type DWord
powershell -NoProfile -ExecutionPolicy Bypass -File "!REG_PS!"
del "!REG_PS!" >nul 2>nul

REM ---- Uninstaller ----
REM The PATH cleanup avoids pipes/carets (they get mangled by echo inside
REM quotes), and the last line is the "(goto) 2>nul" self-delete trick:
REM the batch context ends, releasing the file, then rmdir runs.
set "UNINST=!APP_DIR!\uninstall.bat"
> "!UNINST!"  echo @echo off
>> "!UNINST!" echo setlocal
>> "!UNINST!" echo echo Uninstalling CrowdGPT...
>> "!UNINST!" echo del /q "!START_MENU!\CrowdGPT.lnk" 2^>nul
>> "!UNINST!" echo reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT" /f ^>nul 2^>nul
>> "!UNINST!" echo powershell -NoProfile -Command "$p = [Environment]::GetEnvironmentVariable('Path','User'); if ($p) { $p = [string]::Join(';', @($p.Split(';') -ne '!BIN_DIR!')); [Environment]::SetEnvironmentVariable('Path', $p, 'User') }"
>> "!UNINST!" echo echo Done. Log out and back in for PATH changes to take effect.
>> "!UNINST!" echo pause
>> "!UNINST!" echo cd /d "%TEMP%"
>> "!UNINST!" echo ^(goto^) 2^>nul ^& rmdir /s /q "!APP_DIR!"

echo.
echo [OK] Installed command: crowdgpt
echo [OK] Start Menu shortcut: CrowdGPT  ^(works immediately^)
echo [OK] Uninstall via: Settings ^> Apps ^> CrowdGPT
echo.
echo [NOTE] The "crowdgpt" command is only available in NEW terminals
echo        ^(Windows limitation for permanent PATH changes^).
exit /b 0

REM ============================================================
REM Subroutine: apply_override
REM ============================================================
:apply_override
if /I "!BACKEND_OVERRIDE!"=="nvidia" (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "WHEEL_NOTE=forced: nvidia/cu126"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="rocm" (
    if not defined ROCM_GFX call :choose_rocm_gpu
    if not defined ROCM_GFX exit /b 1
    set "BACKEND=rocm"
    set "TORCH_INDEX=https://stable.repo.amd.com/rocm/whl-next/"
    set "WHEEL_NOTE=forced: Windows ROCm via TheRock !ROCM_GFX!"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="amd" (
    set "BACKEND=amd"
    set "TORCH_INDEX="
    set "WHEEL_NOTE=forced: amd/directml"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="intel" (
    set "BACKEND=intel"
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
    set "WHEEL_NOTE=forced: intel/xpu"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="cpu" (
    set "BACKEND=cpu"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "WHEEL_NOTE=forced: cpu"
    exit /b 0
)
echo [ERROR] --backend must be one of: nvidia, amd, rocm, intel, cpu
exit /b 1

REM ============================================================
REM Subroutine: choose_rocm_gpu
REM The PowerShell helper emits ONLY a gfx target string for a recognized
REM discrete AMD card. Integrated GPUs are not promoted automatically.
REM ============================================================
:choose_rocm_gpu
set "ROCM_GFX="
if not exist "!LOCAL_INSTALLER_DIR!detect_amd_gpu.ps1" (
    echo [WARN] detect_amd_gpu.ps1 missing; AMD will use DirectML.
    exit /b 0
)
for /f "usebackq delims=" %%G in (`powershell -NoProfile -ExecutionPolicy Bypass -File "!LOCAL_INSTALLER_DIR!detect_amd_gpu.ps1" 2^>nul`) do (
    if not defined ROCM_GFX set "ROCM_GFX=%%G"
)
REM Defense in depth: never use unexpected text in pip extras.
set "ROCM_GFX_OK="
for %%G in (gfx1200 gfx1201 gfx1100 gfx1101 gfx1102 gfx1030 gfx1031 gfx1032 gfx1034 gfx1010) do (
    if /I "!ROCM_GFX!"=="%%G" set "ROCM_GFX_OK=1"
)
if not defined ROCM_GFX_OK set "ROCM_GFX="
exit /b 0

REM ============================================================
REM Subroutine: detect_backend
REM ============================================================
:detect_backend

REM --- NVIDIA ---
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :detect_no_nvidia

set "NVIDIA_DRIVER="
for /f "delims=" %%V in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul') do (
    if not defined NVIDIA_DRIVER set "NVIDIA_DRIVER=%%V"
)
if defined NVIDIA_DRIVER (
    for /f "tokens=1 delims= " %%A in ("!NVIDIA_DRIVER!") do set "NVIDIA_DRIVER=%%A"
)

if not defined NVIDIA_DRIVER (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "WHEEL_NOTE=NVIDIA GPU - could not read driver, defaulting to cu126"
    exit /b 0
)

set "DRIVER_MAJOR="
for /f "tokens=1 delims=." %%M in ("!NVIDIA_DRIVER!") do set "DRIVER_MAJOR=%%M"

if not defined DRIVER_MAJOR (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! - defaulting to cu126"
    exit /b 0
)

if !DRIVER_MAJOR! GEQ 525 (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -> CUDA 12.6 wheel"
    exit /b 0
)
if !DRIVER_MAJOR! GEQ 470 (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
    set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -> CUDA 11.8 wheel - older driver"
    exit /b 0
)
echo [ERROR] NVIDIA driver !NVIDIA_DRIVER! is too old for any PyTorch wheel.
echo         Update your GPU driver and rerun.
exit /b 1

:detect_no_nvidia

REM --- Intel Arc ---
powershell -NoProfile -Command "try { $g = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; if ($g -match 'Intel.*Arc') { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 (
    set "BACKEND=intel"
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
    set "WHEEL_NOTE=Intel Arc detected -> XPU wheel"
    exit /b 0
)

REM --- AMD (auto ROCm if mapped, DirectML otherwise) ---
set "AMD_GPU_NAME="
set "AMD_GPU_TMP=%TEMP%\crowdgpt-amd-%RANDOM%.txt"
powershell -NoProfile -Command "((Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -ExpandProperty Name) -replace '\s+', ' ') -join ' / '" > "%AMD_GPU_TMP%" 2>nul
if exist "%AMD_GPU_TMP%" (
    for /f "usebackq delims=" %%A in ("%AMD_GPU_TMP%") do (
        if not defined AMD_GPU_NAME set "AMD_GPU_NAME=%%A"
    )
    del "%AMD_GPU_TMP%" >nul 2>nul
)

if not defined AMD_GPU_NAME goto :detect_cpu

REM Only a known discrete AMD model is eligible for the experimental ROCm build.
REM The source names may also include an integrated AMD GPU.
call :choose_rocm_gpu
if defined ROCM_GFX (
    set "BACKEND=rocm"
    set "TORCH_INDEX=https://stable.repo.amd.com/rocm/whl-next/"
    set "WHEEL_NOTE=AMD dGPU !ROCM_GFX! -> ROCm TheRock (experimental)"
    exit /b 0
)

set "BACKEND=amd"
set "TORCH_INDEX="
set "WHEEL_NOTE=!AMD_GPU_NAME! -> DirectML (torch-directml)"
exit /b 0

:detect_cpu
set "BACKEND=cpu"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "WHEEL_NOTE=No usable GPU detected -> CPU wheel"
exit /b 0

:fatal
echo.
echo ============================================================
echo [ERROR] Installation failed.
echo ============================================================
pause
exit /b 1
