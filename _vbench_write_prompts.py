"""Helper: extract VBench indoor/scenery captions and write temp config."""
import json
import sys
from pathlib import Path

import yaml

ALLOWED_TYPES = {'indoor', 'scenery'}

vbench_json  = Path(sys.argv[1])
prompts_txt  = Path(sys.argv[2])
base_cfg_path = Path(sys.argv[3])
tmp_cfg_path  = Path(sys.argv[4])

entries = json.loads(vbench_json.read_text(encoding='utf-8'))
seen, caps = set(), []
for e in entries:
    fn = e.get('file_name', '')
    if fn not in seen and e.get('type') in ALLOWED_TYPES:
        seen.add(fn)
        caps.append(e.get('caption') or Path(fn).stem)

prompts_txt.write_text('\n'.join(caps), encoding='utf-8')
print(f'[vbench] {len(caps)} prompts -> {prompts_txt}')

cfg = yaml.safe_load(base_cfg_path.read_text(encoding='utf-8'))
cfg['data_path'] = str(prompts_txt)
tmp_cfg_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding='utf-8')
print(f'[vbench] config  -> {tmp_cfg_path}')
