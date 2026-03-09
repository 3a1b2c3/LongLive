@echo off
cd /d "%~dp0"

:: Optional: pass VBench JSON path as first argument, or set VBENCH_ROOT env var
:: Default: look for VBench relative to this bat's parent directory
if not "%~1"=="" (
    set VBENCH_JSON=%~1
) else if not "%VBENCH_ROOT%"=="" (
    set VBENCH_JSON=%VBENCH_ROOT%\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json
) else (
    set VBENCH_JSON=%~dp0..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json
)
set PROMPTS_TXT=%~dp0_vbench_interactive_prompts.txt
set TMP_CFG=%~dp0_vbench_interactive_config.yaml
set BASE_CFG=%~dp0configs\longlive_interactive_inference.yaml

python _vbench_write_prompts.py "%VBENCH_JSON%" "%PROMPTS_TXT%" "%BASE_CFG%" "%TMP_CFG%"
if errorlevel 1 exit /b 1

set PYTHONPATH=%~dp0
set LOCAL_RANK=
set RANK=
set WORLD_SIZE=
set MASTER_ADDR=
set MASTER_PORT=
python interactive_inference.py --config_path "%TMP_CFG%"
