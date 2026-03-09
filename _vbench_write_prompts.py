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

cfg = yaml.safe_load(base_cfg_path.read_text(encoding='utf-8'))

# Determine number of segments from switch_frame_indices
switch_indices = cfg.get('switch_frame_indices', '')
if isinstance(switch_indices, str):
    num_switches = len([x for x in switch_indices.split(',') if x.strip()])
elif isinstance(switch_indices, (list, tuple)):
    num_switches = len(switch_indices)
else:
    num_switches = 0
num_segments = num_switches + 1

entries = json.loads(vbench_json.read_text(encoding='utf-8'))
seen, caps = set(), []
for e in entries:
    fn = e.get('file_name', '')
    if fn not in seen and e.get('type') in ALLOWED_TYPES:
        seen.add(fn)
        caps.append(e.get('caption') or Path(fn).stem)

# Write JSONL: each line repeats the same caption for all segments
prompts_jsonl = prompts_txt.with_suffix('.jsonl')
lines = [json.dumps({'prompts': [cap] * num_segments}) for cap in caps]
prompts_jsonl.write_text('\n'.join(lines), encoding='utf-8')
print(f'[vbench] {len(caps)} prompts ({num_segments} segments each) -> {prompts_jsonl}')

cfg['data_path'] = str(prompts_jsonl)
tmp_cfg_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding='utf-8')
print(f'[vbench] config  -> {tmp_cfg_path}')
