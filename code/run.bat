@echo off
rem Shared launcher: run.bat <script.py> [args...]. Finds Python, installs packages if needed, runs the script.
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -c "import sys" >nul 2>&1 && set "PY=py"
if not defined PY python -c "import sys" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [Error] Python not found. Install Python 3.11 from https://www.python.org/ first.
    goto end
)

%PY% -c "import numpy, scipy, sklearn, torch" >nul 2>&1
if errorlevel 1 (
    echo Installing numpy, scipy, scikit-learn and torch, this may take several minutes...
    %PY% -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [Error] Package install failed. Check the network and try again.
        goto end
    )
)

echo Running %*, please wait...
echo.
%PY% %*
if errorlevel 3 (
    echo.
    echo [Holdout BLOCKED] The independent test set was already viewed, so nothing was trained. See the options printed above: --holdout-batches, --skip-holdout, --debug-holdout.
    goto end
)
if errorlevel 2 (
    echo.
    echo [Self-check FAILED] Possible information leak, or cannot be confirmed. See the log and output\runs\*\config.json. Do not use this run's results. No model was saved.
)

:end
echo.
echo Done. Press any key to close.
pause >nul
