@echo off
rem Start logo-gen on Windows (cmd or PowerShell): sets up uv, Python packages and the
rem frontend, then runs the server. Usage: logo-gen [--port 7860] [--with-local] [--dev]
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "UV_LOCAL=%ROOT%\.tools\uv\uv.exe"
set "UV="

for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV if exist "%UV_LOCAL%" set "UV=%UV_LOCAL%"
if defined UV goto :run

echo   uv not found, installing it into .tools\uv (nothing outside this folder is changed)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_UNMANAGED_INSTALL='%ROOT%\.tools\uv'; irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 exit /b 1
set "UV=%UV_LOCAL%"

:run
if /i "%UV%"=="%UV_LOCAL%" (
  set "UV_PYTHON_INSTALL_DIR=%ROOT%\.tools\python"
  set "UV_CACHE_DIR=%ROOT%\.tools\uv-cache"
)
cd /d "%ROOT%"
"%UV%" run --quiet --no-project --python ">=3.12,<3.14" "%ROOT%\scripts\launch.py" --uv "%UV%" %*
exit /b %errorlevel%
