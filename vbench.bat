@echo off
setlocal enabledelayedexpansion

set PY_SCRIPT=%~dp0vbench_runner.py
set BASE_CONFIG=%~dp0configs\longlive_inference.yaml

cd /d "%~dp0"
set USE_LIBUV=0

:: Default paths relative to this repo — override by passing them as arguments:
::   vbench.bat [VBENCH_JSON] [BASE_CONFIG]
if not "%~1"=="" (set VBENCH_JSON=%~1) else (set VBENCH_JSON=%~dp0..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json)
if not "%~2"=="" (set BASE_CONFIG=%~2)

echo Output:    %CD%\outputs\vbench
echo VBench:    %VBENCH_JSON%
echo Config:    %BASE_CONFIG%

python "%PY_SCRIPT%" ^
    --vbench-json "%VBENCH_JSON%" ^
    --base-config "%BASE_CONFIG%" ^
    --work-dir "%CD%"

exit /b %ERRORLEVEL%
