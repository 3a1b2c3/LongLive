@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
set LOCAL_RANK=
set RANK=
set WORLD_SIZE=
set MASTER_ADDR=
set MASTER_PORT=
python interactive_inference.py --config_path configs/longlive_interactive_inference.yaml
