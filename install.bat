@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM CrowdGPT universal installer / launcher for Windows
REM Detects Python + Git + GPU backend, installs the matching requirements,
REM and launches client.py.

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
set "REPO_DIR=CrowdGPT"
set "PYTHON_EXE="
set "VENV_DIR="
set "REQ_FILE="
set "BACKEND="

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

echo [INFO] Python was not found. Installing uv, then the latest stable Python...

where uv >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if not errorlevel 1 (
        echo [INFO] Installing uv with winget...
        winget install --id=astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
        if errorlevel 1 goto :fatal
        set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv;%PATH%"
    ) else (
        echo [INFO] Installing uv using the official installer...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if errorlevel 1 goto :fatal
        set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv;%PATH%"
    )
)

if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%LOCALAPPDATA%\uv\uv.exe" set "PATH=%LOCALAPPDATA%\uv;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv was installed but cannot be found in PATH.
    echo         Restart this terminal and run install.bat again.
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
    echo [INFO] Git was not found. Installing Git for Windows...

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
if errorlevel 1 (
    echo [ERROR] Git is still unavailable.
    goto :fatal
)
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
    if errorlevel 1 (
        echo [WARN] git pull failed; continuing with the existing checkout.
    )
    goto :repo_ready
)

if exist "%REPO_DIR%" (
    echo [ERROR] "%REPO_DIR%" exists but is not a Git repository.
    echo         Move/delete it and run this installer again.
    goto :fatal
)

echo [INFO] Cloning CrowdGPT...
git clone "%REPO_URL%" "%REPO_DIR%"
if errorlevel 1 goto :fatal

:repo_ready
cd /d "%REPO_DIR%"
if not exist "client.py" (
    echo [ERROR] client.py was not found in the CrowdGPT repository.
    goto :fatal
)

REM ------------------------------------------------------------
REM 4. Detect GPU / backend
REM ------------------------------------------------------------
echo.
echo [INFO] Detecting hardware backend...

REM NVIDIA takes priority when nvidia-smi is available.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    set "BACKEND=cuda"
    set "REQ_FILE=requirements_cuda.txt"
    goto :backend_found
)

REM On Windows, use DirectML only when an AMD/Intel GPU is actually present.
REM Otherwise use the CPU requirements.
powershell -NoProfile -Command "$g=Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterCompatibility; if ($g -match 'AMD|Advanced Micro Devices|Intel') { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    set "BACKEND=directml"
    set "REQ_FILE=requirements_directml.txt"
) else (
    set "BACKEND=cpu"
    set "REQ_FILE=requirements.txt"
)

:backend_found
if not exist "%REQ_FILE%" (
    echo [WARN] %REQ_FILE% is missing from this checkout.
    echo [INFO] Falling back to requirements.txt (CPU/default).
    set "BACKEND=cpu"
    set "REQ_FILE=requirements.txt"
)

echo [OK] Backend: !BACKEND!
echo [OK] Requirements: !REQ_FILE!

REM ------------------------------------------------------------
REM 5. Create isolated virtual environment
REM ------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    "!PYTHON_EXE!" -m venv .venv
    if errorlevel 1 (
        echo [WARN] venv creation failed. Trying uv to create it...
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
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :fatal

REM ROCm has no normal PyPI torch wheel; use the ROCm index on Unix only.
REM Windows always uses the CUDA or DirectML requirements here.
echo [INFO] Installing CrowdGPT dependencies...
"%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :fatal

REM ------------------------------------------------------------
REM 6. Launch client.py
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                 Installation complete
echo ============================================================
echo Backend: !BACKEND!
echo Starting client.py...
echo.

"%PYTHON_EXE%" client.py
set "EXITCODE=%ERRORLEVEL%"

echo.
echo CrowdGPT exited with code %EXITCODE%.
exit /b %EXITCODE%

:fatal
echo.
echo ============================================================
echo [ERROR] Installation failed.
echo ============================================================
pause
exit /b 1
