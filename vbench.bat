@echo off
setlocal enabledelayedexpansion

set VBENCH_JSON=C:\workspace\world\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json
set BASE_CONFIG=%~dp0configs\longlive_inference.yaml
set PY_SCRIPT=%~dp0vbench_runner.py

cd /d "%~dp0"
set USE_LIBUV=0

for /f "tokens=*" %%i in ('python -c "import random; print(random.randint(0,1999999999))"') do set BASE_SEED=%%i

echo Output:    %CD%\outputs\vbench
echo VBench:    %VBENCH_JSON%
echo Base seed: %BASE_SEED%

python "%PY_SCRIPT%" ^
    --base-seed %BASE_SEED% ^
    --vbench-json "%VBENCH_JSON%" ^
    --base-config "%BASE_CONFIG%" ^
    --work-dir "%CD%"

exit /b %ERRORLEVEL%
