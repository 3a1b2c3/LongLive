@echo off
cd /d "%~dp0"

set VBENCH_JSON=C:\workspace\world\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json
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
