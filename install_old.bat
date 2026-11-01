@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT Universal Installer for Windows
REM v13
REM  Changes vs v12:
REM   [FIX v13-a] :check_blackwell now validates that nvidia-smi
REM               returns a NUMERIC compute capability, and falls back
REM               to PowerShell CIM (Win32_VideoController) when
REM               nvidia-smi is broken/stale. Broken nvidia-smi was
REM               returning "ERROR: ..." and silently preventing
REM               Blackwell detection → cu126 install → client refusal.
REM   [FIX v13-b] :detect_backend validates that nvidia-smi's
REM               driver_version is numeric (digits.digits). Previously
REM               "ERROR:" got compared with GEQ 525 as a string and
REM               took the "modern driver" branch → cu126.
REM   [FIX v13-c] Added :verify_torch_supports_gpu — queries torch's
REM               own get_device_capability()/get_arch_list() AFTER
REM               install, and auto-upgrades cu126 → cu128 → cu130 if
REM               the wheel has no matching kernels. This is the real
REM               safety net: even if detection misfires, we fix it
REM               before the user launches the client.
REM   [FIX v13-d] Verify block (step 10) rewritten as flat sequential
REM               goto-based checks. The nested if-blocks around
REM               Python -c "..." strings were the source of the v11/v12
REM               "present was unexpected at this time" parse error,
REM               which also skipped the Blackwell verification.
REM   [FIX v13-e] apply_override also validates NB_MAJOR is numeric
REM               before GEQ 580, and calls the same enhanced
REM               :check_blackwell (was: only nvidia-smi).
REM   [FIX v13-f] Better bitsandbytes warning (was misleadingly about
REM               VRAM; it's about optimizer-mode compatibility).
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
set "BLACKWELL_FOUND=0"
set "BLACKWELL_MAJOR="
set "EARLY_INTEGRATE_DONE=0"

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
echo   - AMD automatic selection: recognized GPU -^> ROCm; otherwise DirectML.
echo   - Experimental ROCm: --backend rocm (requires recognized AMD GPU).
echo   - Requires a compatible AMD driver and TheRock PyTorch wheels.
echo   - Intel Arc uses the XPU wheel. NVIDIA uses CUDA.
echo   - Blackwell (RTX 50-series, B100/B200) auto-routes to cu128/cu130.
echo   - Post-install check verifies installed torch has kernels for the GPU.
echo.
exit /b 0

:args_done

echo.
echo ============================================================
echo                 CrowdGPT Installer  (v13)
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 0. Paths
REM ------------------------------------------------------------
set "APP_DIR=%LOCALAPPDATA%\CrowdGPT"
set "BIN_DIR=%APP_DIR%"
set "START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

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
if "!BLACKWELL_FOUND!"=="1" echo   Blackwell:   yes (compute capability !BLACKWELL_MAJOR!.x)
echo   Python:      !PY_VER!
echo ============================================================
echo.

if /I "!BACKEND!"=="amd" call :note_amd
if /I "!BACKEND!"=="rocm" (
    echo [NOTE] Experimental Windows ROCm via TheRock, !ROCM_GFX!.
    echo [NOTE] This is NOT a guarantee that CrowdGPT training will work.
    echo [NOTE] Failed ROCm install / GPU smoke test switches to DirectML.
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
if exist "!VENV_DIR!" (
    echo [INFO] Stopping CrowdGPT and Python processes using this venv...
    echo [WARN] Any unsaved work in CrowdGPT or scripts using this venv may be lost.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$v=[IO.Path]::GetFullPath('!VENV_DIR!').TrimEnd('\')+'\'; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^crowdgpt(\.exe)?$' -or ($_.Name -match '^pythonw?\.exe$' -and $_.ExecutablePath -and $_.ExecutablePath.StartsWith($v,[StringComparison]::OrdinalIgnoreCase)) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul
    timeout /t 2 /nobreak >nul
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q "!VENV_DIR!" >nul 2>nul
)

if exist "!VENV_DIR!" (
    echo [ERROR] Could not remove the existing venv.
    echo         Close CrowdGPT and any process using its Python environment, then retry.
    goto :fatal
)

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

:pytorch_rocm
echo [INFO] Installing native Windows PyTorch ROCm through AMD TheRock...
echo [INFO] ROCm 10.1.0 stable, GPU target !ROCM_GFX!.
"!APP_PY!" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-!ROCM_GFX!]==2.13.0+rocm10.1.0"
if errorlevel 1 (
    echo [WARN] ROCm wheel installation failed.
    goto :rocm_to_directml
)
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
if not defined TORCH_INDEX (
    echo [WARN] TORCH_INDEX not set by detection; defaulting to cu126.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
)
echo [INFO] Trying PyTorch from !TORCH_INDEX!...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url "!TORCH_INDEX!"
if not errorlevel 1 (
    "!APP_PY!" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'" 2>nul
    if not errorlevel 1 goto :pytorch_done
    echo [WARN] torch installed but CUDA is not available on the primary index.
)
echo.
echo [WARN] Primary install failed or CUDA unavailable. Trying CUDA 11.8 cu118...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
if not errorlevel 1 (
    "!APP_PY!" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after cu118 install'" 2>nul
    if not errorlevel 1 (
        set "WHEEL_NOTE=!WHEEL_NOTE! (cu118 fallback)"
        goto :pytorch_done
    )
)
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

REM ============================================================
REM [FIX v13-c] Post-torch capability check + auto-upgrade.
REM Ask torch itself whether the wheel it just installed has
REM kernels for the detected GPU. If not, force-reinstall from
REM cu128 (fallback cu130). This is the load-bearing fix: even
REM if detection misfired, we self-heal before launch.
REM ============================================================
if /I "!BACKEND!"=="nvidia" call :ensure_torch_matches_gpu
if errorlevel 1 goto :fatal

REM ------------------------------------------------------------
REM 7. Install project requirements
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
REM 7a. bitsandbytes (8-bit Adam) on NVIDIA
REM [FIX v13-f] Clarify the warning — it's about optimizer-mode
REM              compatibility in heterogeneous pools, not VRAM.
REM ------------------------------------------------------------
if /I "!BACKEND!"=="nvidia" (
    echo.
    echo [INFO] Installing bitsandbytes ^(8-bit optimizer^)...
    "!APP_PY!" -m pip install bitsandbytes
    if errorlevel 1 (
        echo [WARN] bitsandbytes failed to install.
        echo        The client will fall back to adamw_fp32. This can
        echo        cause cross-check flags in mixed-optimizer pools
        echo        because fp32 AdamW and 8-bit AdamW produce
        echo        structurally different deltas. Not fatal, but
        echo        expect more flag events until this is resolved.
    ) else (
        echo [OK] bitsandbytes installed
    )
)

REM ------------------------------------------------------------
REM 7b. transformers pin, based on installed torch version
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
REM 9. Desktop integration (BEFORE verify, so a verify crash
REM    still leaves a working launcher)
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    call :do_integration
    if errorlevel 1 goto :fatal
    set "EARLY_INTEGRATE_DONE=1"
)

REM ============================================================
REM 10. Verify
REM [FIX v13-d] Flat sequential goto-based checks. The v11/v12
REM nested if-blocks around -c "..." strings were the source of
REM "present was unexpected at this time", which also SKIPPED the
REM Blackwell verification that would have caught the mismatch.
REM ============================================================
echo.
echo ============================================================
echo                    Verifying install
echo ============================================================

"!APP_PY!" -c "import sys, torch; print('  Python:', sys.version.split()[0]); print('  Torch:  ', torch.__version__)"
if errorlevel 1 goto :verify_torch_import_failed

"!APP_PY!" -c "import transformers, torch; assert transformers.utils.is_torch_available(), 'transformers refuses to use installed torch'; print('  transformers:', transformers.__version__, '-> torch', torch.__version__, 'linked OK')"
if errorlevel 1 goto :verify_transformers_failed

if /I "!BACKEND!"=="nvidia" goto :verify_nvidia
if /I "!BACKEND!"=="amd" goto :verify_directml
if /I "!BACKEND!"=="rocm" goto :verify_rocm
goto :verify_pywebview

:verify_torch_import_failed
echo [WARN] Torch import failed.
goto :verify_pywebview

:verify_transformers_failed
echo.
echo [ERROR] transformers cannot use the installed torch.
echo         Try: pip install "transformers>=4.45,<5.0"
goto :fatal

:verify_nvidia
"!APP_PY!" -c "import torch, sys; sys.exit(0 if torch.version.cuda else 2)"
if errorlevel 1 goto :verify_cuda_missing

"!APP_PY!" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 1 goto :verify_cuda_unavailable

call :verify_torch_supports_gpu
if "!TORCH_GPU_RC!"=="3" goto :verify_sm_missing

"!APP_PY!" -c "import torch; print('  CUDA device:', torch.cuda.get_device_name(0)); print('  CUDA version:', torch.version.cuda); print('  Arch list:', ', '.join(torch.cuda.get_arch_list()))"
goto :verify_pywebview

:verify_cuda_missing
echo.
echo [ERROR] BACKEND=nvidia but installed torch is CPU-only (torch.version.cuda is None).
echo         This should not happen — pytorch_nvidia always installs a CUDA wheel.
echo         Retry: install.bat --backend nvidia
goto :fatal

:verify_cuda_unavailable
echo [WARN] CUDA wheel installed but no CUDA device is visible to torch.
echo        If you have an NVIDIA GPU, check the driver in Device Manager.
echo        If this is a headless/VM environment, this warning is expected.
goto :verify_pywebview

:verify_sm_missing
echo.
echo [ERROR] Installed torch wheel has no kernels for this GPU.
echo         Client will refuse to train at task-fetch time.
echo.
echo         Fix manually:
echo           "!APP_PY!" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo.
echo         Or rerun: install.bat --backend nvidia
goto :fatal

:verify_rocm
"!APP_PY!" -c "import torch; assert torch.version.hip and torch.cuda.is_available(), 'ROCm unavailable after requirements'; gs=[(i,torch.cuda.get_device_name(i),int(torch.cuda.get_device_properties(i).total_memory)) for i in range(torch.cuda.device_count())]; ds=[g for g in gs if 'radeon rx' in g[1].lower()]; g=max(ds or gs,key=lambda x:x[2]); print('  HIP:',torch.version.hip); print('  ROCm GPU:',g[1]); assert ds, 'No discrete Radeon RX visible after requirements'"
if errorlevel 1 (
    echo [ERROR] ROCm unavailable after dependencies installation.
    goto :fatal
)
goto :verify_pywebview

:verify_directml
"!APP_PY!" -c "import torch_directml as d; [print('  DirectML[' + str(i) + ']: ' + str(d.device_name(i))) for i in range(d.device_count())]" 2>nul
if errorlevel 1 echo [WARN] Could not enumerate DirectML devices.
"!APP_PY!" -c "import torch; import torch_directml as d; assert d.is_available(), 'no DirectML device'; ds=[(i,str(d.device_name(i))) for i in range(d.device_count())]; rx=[g for g in ds if 'radeon rx' in g[1].lower()]; i=(rx[0] if rx else ds[0])[0]; dev=d.device(i); a=torch.randn(64,64).to(dev); b=torch.randn(64,64).to(dev); c=(a@b).sum().cpu().item(); print('  DirectML['+str(i)+'] matmul OK, sum =',round(c,4))" 2>nul
if errorlevel 1 echo [WARN] DirectML device present but matmul failed. Some ops may be unsupported.
goto :verify_pywebview

:verify_pywebview
"!APP_PY!" -c "import webview; print('  pywebview: OK')" 2>nul
if errorlevel 1 echo [WARN] pywebview import failed.

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
REM Subroutine: probe_uv / probe_git
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
REM ============================================================
:ensure_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0

call :probe_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0

echo [INFO] uv not found. Installing uv...
set "UV_INSTALL_DIR=%LOCALAPPDATA%\uv\bin"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; irm https://astral.sh/uv/install.ps1 | iex"
set "UV_INSTALL_DIR="
call :refresh_path
call :probe_uv
where uv >nul 2>nul
if not errorlevel 1 exit /b 0

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
REM Subroutine: check_blackwell
REM [FIX v13-a] Layered detection. Tries nvidia-smi compute_cap
REM (validated numeric), then nvidia-smi name, then PowerShell
REM CIM (Win32_VideoController). Validates that the returned value
REM is actually a Blackwell family value before setting
REM BLACKWELL_MAJOR. Rejects garbage like "ERROR:".
REM Sets BLACKWELL_FOUND=1, BLACKWELL_MAJOR to the compute
REM capability major if any GPU is Blackwell (10..16 range).
REM ============================================================
:check_blackwell
set "BLACKWELL_FOUND=0"
set "BLACKWELL_MAJOR="
set "BK_CAP_RAW="
set "BK_NAME_RAW="

REM --- Method 1: nvidia-smi compute_cap (must be numeric) ---
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%C in (`nvidia-smi --query-gpu^=compute_cap --format^=csv,noheader 2^>nul`) do (
        if not defined BK_CAP_RAW set "BK_CAP_RAW=%%C"
    )
    if defined BK_CAP_RAW (
        set "BK_CAP_RAW=!BK_CAP_RAW: =!"
        REM Only accept N.N or N.NN — reject "ERROR:" / "[N/A]" / empty.
        echo !BK_CAP_RAW!| findstr /R /C:"^[0-9][0-9]*\.[0-9][0-9]*$" >nul 2>nul
        if not errorlevel 1 (
            for /f "tokens=1 delims=." %%M in ("!BK_CAP_RAW!") do (
                set "BK_MAJ=%%M"
                if "!BK_MAJ!"=="10" set "BLACKWELL_MAJOR=10"
                if "!BK_MAJ!"=="11" set "BLACKWELL_MAJOR=11"
                if "!BK_MAJ!"=="12" set "BLACKWELL_MAJOR=12"
                if "!BK_MAJ!"=="13" set "BLACKWELL_MAJOR=13"
                if "!BK_MAJ!"=="14" set "BLACKWELL_MAJOR=14"
                if "!BK_MAJ!"=="15" set "BLACKWELL_MAJOR=15"
                if "!BK_MAJ!"=="16" set "BLACKWELL_MAJOR=16"
            )
        )
    )
)

REM --- Method 2: nvidia-smi name (works with old nvidia-smi, and ---
REM --- with new nvidia-smi that fails compute_cap for some reason) ---
if not defined BLACKWELL_MAJOR (
    if not errorlevel 1 (
        for /f "usebackq delims=" %%N in (`nvidia-smi --query-gpu^=name --format^=csv,noheader 2^>nul`) do (
            if not defined BK_NAME_RAW set "BK_NAME_RAW=%%N"
        )
        if defined BK_NAME_RAW (
            echo !BK_NAME_RAW!| findstr /I /R /C:"RTX 50[0-9][0-9]" /C:"B100" /C:"B200" /C:"GB200" /C:"GB100" /C:"Blackwell" >nul 2>nul
            if not errorlevel 1 set "BLACKWELL_MAJOR=12"
        )
    )
)

REM --- Method 3: PowerShell CIM (Win32_VideoController name) ---
REM Does NOT use nvidia-smi at all. Fires when nvidia-smi is broken
REM or missing entirely. Output is one device name per line.
if not defined BLACKWELL_MAJOR (
    for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue ^| ForEach-Object { $_.Name }" 2^>nul`) do (
        echo %%P| findstr /I /R /C:"RTX 50[0-9][0-9]" /C:"B100" /C:"B200" /C:"GB200" /C:"GB100" /C:"Blackwell" >nul 2>nul
        if not errorlevel 1 if not defined BLACKWELL_MAJOR set "BLACKWELL_MAJOR=12"
    )
)

if defined BLACKWELL_MAJOR (
    set "BLACKWELL_FOUND=1"
    exit /b 0
)
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
exit /b 0

REM ============================================================
REM Subroutine: verify_directml
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

> "!BIN_DIR!\crowdgpt.cmd" echo @echo off
>>"!BIN_DIR!\crowdgpt.cmd" echo "!APP_PY!" "!APP_DIR!\client.py" %%*

set "GUI_VBS=!BIN_DIR!\crowdgpt-gui.vbs"
> "!GUI_VBS!"  echo Set WshShell = CreateObject("WScript.Shell")
>> "!GUI_VBS!" echo WshShell.Run """!APP_PY!"" ""!APP_DIR!\client.py""", 0, False

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

set "APP_ICON="
if exist "!APP_DIR!\docs\logo-app.ico" set "APP_ICON=!APP_DIR!\docs\logo-app.ico"

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

set "REG_PS=%TEMP%\crowdgpt-reg-%RANDOM%.ps1"
> "!REG_PS!"  echo $k = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT'
>> "!REG_PS!" echo New-Item -Path $k -Force ^| Out-Null
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayName'     -Value 'CrowdGPT'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayVersion'  -Value '0.13.0'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'Publisher'       -Value 'CrowdGPT Project'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'InstallLocation' -Value '!APP_DIR!'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'UninstallString' -Value ('"' + '!APP_DIR!\uninstall.bat' + '"')
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'NoModify'        -Value 1 -Type DWord
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'NoRepair'        -Value 1 -Type DWord
powershell -NoProfile -ExecutionPolicy Bypass -File "!REG_PS!"
del "!REG_PS!" >nul 2>nul

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
REM [FIX v13-e] Validate NB_MAJOR is numeric before GEQ 580.
REM             Call enhanced :check_blackwell for detection.
REM ============================================================
:apply_override
if /I "!BACKEND_OVERRIDE!"=="nvidia" (
    set "BACKEND=nvidia"
    call :check_blackwell
    if "!BLACKWELL_FOUND!"=="1" (
        set "NB_DRIVER="
        for /f "usebackq tokens=1 delims=," %%V in (`nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul`) do (
            if not defined NB_DRIVER set "NB_DRIVER=%%V"
        )
        if defined NB_DRIVER (
            for /f "tokens=1 delims= " %%V in ("!NB_DRIVER!") do set "NB_DRIVER=%%V"
        )
        set "NB_MAJOR="
        if defined NB_DRIVER (
            echo !NB_DRIVER!| findstr /R /C:"^[0-9][0-9]*\.[0-9][0-9]*" >nul 2>nul
            if not errorlevel 1 (
                for /f "tokens=1 delims=." %%M in ("!NB_DRIVER!") do set "NB_MAJOR=%%M"
            )
        )
        if defined NB_MAJOR (
            if !NB_MAJOR! GEQ 580 (
                set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
                set "WHEEL_NOTE=forced: nvidia/Blackwell -> cu130 (driver !NB_DRIVER!)"
                exit /b 0
            )
        )
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        set "WHEEL_NOTE=forced: nvidia/Blackwell -> cu128"
        exit /b 0
    )
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
set "ROCM_GFX_OK="
for %%G in (gfx1200 gfx1201 gfx1100 gfx1101 gfx1102 gfx1030 gfx1031 gfx1032 gfx1034 gfx1010) do (
    if /I "!ROCM_GFX!"=="%%G" set "ROCM_GFX_OK=1"
)
if not defined ROCM_GFX_OK set "ROCM_GFX="
exit /b 0

REM ============================================================
REM Subroutine: detect_backend
REM [FIX v13-b] Validate nvidia-smi driver_version is numeric.
REM             Reject "ERROR:" / non-numeric before any GEQ
REM             comparison (which falls back to string compare and
REM             silently misroutes). Blackwell check runs BEFORE
REM             the driver-version gate — Blackwell always needs
REM             cu128/cu130 regardless of what nvidia-smi thinks.
REM ============================================================
:detect_backend
set "BLACKWELL_FOUND=0"
set "BLACKWELL_MAJOR="

REM --- NVIDIA ---
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :detect_no_nvidia

set "NVIDIA_DRIVER_RAW="
for /f "usebackq delims=" %%V in (`nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul`) do (
    if not defined NVIDIA_DRIVER_RAW set "NVIDIA_DRIVER_RAW=%%V"
)
set "NVIDIA_DRIVER="
if defined NVIDIA_DRIVER_RAW (
    for /f "tokens=1 delims=, " %%A in ("!NVIDIA_DRIVER_RAW!") do set "NVIDIA_DRIVER=%%A"
)
set "DRIVER_MAJOR="
if defined NVIDIA_DRIVER (
    echo !NVIDIA_DRIVER!| findstr /R /C:"^[0-9][0-9]*\.[0-9][0-9]*" >nul 2>nul
    if not errorlevel 1 (
        for /f "tokens=1 delims=." %%M in ("!NVIDIA_DRIVER!") do set "DRIVER_MAJOR=%%M"
    ) else (
        echo [WARN] nvidia-smi returned non-numeric driver version "!NVIDIA_DRIVER!" — ignoring it.
        set "NVIDIA_DRIVER="
    )
)

REM --- Blackwell check FIRST (before driver-version gate) ---
REM Every Blackwell GPU needs cu128 or cu130. If we detected one,
REM route there regardless of what the driver reports, because
REM broken nvidia-smi cannot be trusted and a stale driver version
REM on a Blackwell card is a separate problem the user must fix.
call :check_blackwell
if "!BLACKWELL_FOUND!"=="1" (
    set "BACKEND=nvidia"
    if defined DRIVER_MAJOR (
        if !DRIVER_MAJOR! GEQ 580 (
            set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
            set "WHEEL_NOTE=Blackwell -> cu130 (driver !NVIDIA_DRIVER!)"
            exit /b 0
        )
    )
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    if defined DRIVER_MAJOR (
        set "WHEEL_NOTE=Blackwell -> cu128 (driver !NVIDIA_DRIVER!; cu130 needs 580+)"
    ) else (
        set "WHEEL_NOTE=Blackwell -> cu128 (driver version unreadable)"
    )
    exit /b 0
)

REM --- Non-Blackwell NVIDIA: driver version gate ---
if not defined DRIVER_MAJOR (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "WHEEL_NOTE=NVIDIA GPU - could not read driver, defaulting to cu126"
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

REM ============================================================
REM Subroutine: verify_torch_supports_gpu
REM [FIX v13-c] Ask torch itself whether the installed wheel has
REM kernels for the detected GPU. Writes the check to a temp .py
REM file so no batch string-parse issues with () [] in the source.
REM Returns:
REM   0 = okay (CUDA unavailable, or kernels match)
REM   3 = GPU visible, wheel has no matching kernels → upgrade
REM ============================================================
:verify_torch_supports_gpu
set "TORCH_GPU_RC=0"
set "TORCH_CHECK_PY=%TEMP%\crowdgpt-torch-check-%RANDOM%.py"
> "!TORCH_CHECK_PY!"  echo import sys
>>"!TORCH_CHECK_PY!" echo try:
>>"!TORCH_CHECK_PY!" echo     import torch
>>"!TORCH_CHECK_PY!" echo except Exception as e:
>>"!TORCH_CHECK_PY!" echo     print('  torch import failed:', e)
>>"!TORCH_CHECK_PY!" echo     sys.exit(0)
>>"!TORCH_CHECK_PY!" echo if not torch.cuda.is_available():
>>"!TORCH_CHECK_PY!" echo     sys.exit(0)
>>"!TORCH_CHECK_PY!" echo try:
>>"!TORCH_CHECK_PY!" echo     cap = torch.cuda.get_device_capability(0)
>>"!TORCH_CHECK_PY!" echo     archs = list(torch.cuda.get_arch_list())
>>"!TORCH_CHECK_PY!" echo except Exception as e:
>>"!TORCH_CHECK_PY!" echo     print('  capability query failed:', e)
>>"!TORCH_CHECK_PY!" echo     sys.exit(0)
>>"!TORCH_CHECK_PY!" echo need = 'sm_' + str(cap[0]) + str(cap[1])
>>"!TORCH_CHECK_PY!" echo prefix_sm = 'sm_' + str(cap[0])
>>"!TORCH_CHECK_PY!" echo prefix_compute = 'compute_' + str(cap[0])
>>"!TORCH_CHECK_PY!" echo same_major = [a for a in archs if a.startswith(prefix_sm) or a.startswith(prefix_compute)]
>>"!TORCH_CHECK_PY!" echo ok = (need in archs) or (len(same_major) ^> 0)
>>"!TORCH_CHECK_PY!" echo print('  GPU CC:', need)
>>"!TORCH_CHECK_PY!" echo print('  wheel archs:', ', '.join(archs) if archs else '(none)')
>>"!TORCH_CHECK_PY!" echo sys.exit(0 if ok else 3)
"!APP_PY!" "!TORCH_CHECK_PY!"
set "TORCH_GPU_RC=!ERRORLEVEL!"
del "!TORCH_CHECK_PY!" >nul 2>nul
exit /b 0

REM ============================================================
REM Subroutine: ensure_torch_matches_gpu
REM [FIX v13-c] Called from :pytorch_done for BACKEND=nvidia.
REM If verify_torch_supports_gpu returns 3, force-reinstall from
REM cu128 (then cu130 as last resort). Uses the current installed
REM torch version so we don't accidentally downgrade/upgrade
REM versions — we only swap the CUDA runtime build.
REM ============================================================
:ensure_torch_matches_gpu
call :verify_torch_supports_gpu
if "!TORCH_GPU_RC!"=="0" (
    echo [OK] Installed torch has kernels for the detected GPU.
    exit /b 0
)

echo.
echo ============================================================
echo   Auto-upgrade: installed torch has NO kernels for this GPU
echo ============================================================
echo [WARN] Detected kernel mismatch. The client would refuse to train.
echo [WARN] Reinstalling torch from cu128 (this takes ~2.6 GB / a few min)...
"!APP_PY!" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERROR] cu128 reinstall failed. Network or index issue?
    echo         Manual fix: "!APP_PY!" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    exit /b 1
)
call :verify_torch_supports_gpu
if "!TORCH_GPU_RC!"=="0" (
    echo [OK] cu128 wheel installed and has kernels for this GPU.
    set "WHEEL_NOTE=!WHEEL_NOTE! (auto-upgraded to cu128)"
    exit /b 0
)

echo [WARN] cu128 still lacks kernels for this GPU. Trying cu130...
"!APP_PY!" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo [ERROR] cu130 reinstall failed.
    exit /b 1
)
call :verify_torch_supports_gpu
if "!TORCH_GPU_RC!"=="0" (
    echo [OK] cu130 wheel installed and has kernels for this GPU.
    set "WHEEL_NOTE=!WHEEL_NOTE! (auto-upgraded to cu130)"
    exit /b 0
)

echo.
echo [ERROR] Neither cu128 nor cu130 provides kernels for this GPU.
echo         Driver may be too old. Update your NVIDIA driver and rerun.
echo         Current: driver !NVIDIA_DRIVER!
"!APP_PY!" -c "import torch; print('  torch:', torch.__version__, '| cuda:', torch.version.cuda, '| arch:', torch.cuda.get_arch_list())"
exit /b 1

:fatal
echo.
echo ============================================================
echo [ERROR] Installation failed.
echo ============================================================
pause
exit /b 1
