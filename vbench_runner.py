"""
vbench_runner.py  --  VBench T2V runner for LongLive
Runs inference.py once per prompt (61 prompts x 5 samples = 305 runs).
Each prompt gets its own reproducible seed; 5 diverse samples are drawn in one pass.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ALLOWED_TYPES = ['indoor', 'scenery']
NUM_SAMPLES   = 5
NUM_FRAMES    = 161


# ---------- helpers ----------

def _poll_vram(stop_event, readings):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, text=True
            ).strip().splitlines()[0].strip()
            if out.isdigit():
                readings.append(int(out))
        except Exception:
            pass
        time.sleep(5)

def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().used / (1024 ** 3), 2)
    except Exception:
        return ''

def _vram_peak_gb(readings):
    if not readings:
        return ''
    return round(max(readings) / 1024.0, 2)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-seed',    required=True, type=int)
    ap.add_argument('--vbench-json',  required=True)
    ap.add_argument('--base-config',  required=True)
    ap.add_argument('--work-dir',     required=True)
    args = ap.parse_args()

    work_dir   = Path(args.work_dir)
    out_base   = work_dir / 'outputs' / 'vbench' / 'videos'
    stats_path = work_dir / 'outputs' / 'vbench' / 'stats.csv'

    if not Path(args.vbench_json).exists():
        sys.exit(f'[vbench] ERROR: JSON not found: {args.vbench_json}')
    if not Path(args.base_config).exists():
        sys.exit(f'[vbench] ERROR: base config not found: {args.base_config}')

    with open(args.vbench_json, encoding='utf-8') as f:
        entries = json.load(f)

    seen, prompts = set(), []
    for e in entries:
        fn = e.get('file_name', '')
        if fn in seen or e.get('type') not in ALLOWED_TYPES:
            continue
        seen.add(fn)
        prompts.append({'caption': e.get('caption') or Path(fn).stem, 'type': e['type']})

    total = len(prompts) * NUM_SAMPLES
    print(f'\n=== VBench ({len(prompts)} prompts, types: {", ".join(ALLOWED_TYPES)}) x {NUM_SAMPLES} samples = {total} runs ===\n')
    for i, p in enumerate(prompts):
        print(f'  [{i+1:2}/{len(prompts)}] ({p["type"]:<8}) {p["caption"][:100]}')
    print()

    out_base.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # load base config once
    with open(args.base_config, encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    env = {k: v for k, v in os.environ.items()
           if k not in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE',
                        'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RESTART_COUNT',
                        'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_RUN_ID')}
    env.update({
        'TOKENIZERS_PARALLELISM': 'false',
        'TF_ENABLE_ONEDNN_OPTS':  '0',
        'USE_LIBUV': '0',
        'PYTHONPATH': str(work_dir),
    })

    prompt_file = work_dir / '_vbench_prompt.txt'
    tmp_cfg     = work_dir / '_vbench_config.yaml'

    stats_is_new = not stats_path.exists()
    stats_f = open(stats_path, 'a', newline='', encoding='utf-8')
    writer  = csv.writer(stats_f)
    if stats_is_new:
        writer.writerow(['prompt_idx', 'prompt', 'type', 'sample_idx', 'seed',
                         'duration_s', 'gen_fps', 'vram_gb', 'ram_gb', 'out_path', 'status'])
        stats_f.flush()

    generated = skipped = errors = 0
    t_start = time.time()

    for ti, p in enumerate(prompts):
        cap  = p['caption']
        seed = random.Random(args.base_seed ^ hash(cap)).randint(0, 2**31 - 1)

        # skip if all 5 samples already exist
        existing = sorted(out_base.glob(f'rank0-{ti}-*_*.mp4'))
        if len(existing) >= NUM_SAMPLES:
            print(f'[vbench] skip prompt {ti+1}/{len(prompts)}: all {NUM_SAMPLES} samples present')
            for si, f in enumerate(existing[:NUM_SAMPLES]):
                writer.writerow([ti, cap, p['type'], si, seed, '', '', '', '', str(f), 'skipped'])
            stats_f.flush()
            skipped += NUM_SAMPLES
            continue

        pct = round(100 * ti / len(prompts))
        eta = ''
        if ti > 0:
            elapsed = time.time() - t_start
            rem = int(elapsed / ti * (len(prompts) - ti))
            eta = f' ETA {rem//3600}:{(rem%3600)//60:02d}:{rem%60:02d}'
        print(f'[vbench] [{ti+1}/{len(prompts)} {pct}%{eta}]  ({p["type"]})  seed {seed} : {cap[:70]}')

        # write single-prompt file
        prompt_file.write_text(cap, encoding='utf-8')

        # build per-prompt config
        cfg = dict(base_cfg)
        cfg['data_path']           = str(prompt_file)
        cfg['output_folder']       = str(out_base)
        cfg['seed']                = seed
        cfg['num_samples']         = NUM_SAMPLES
        cfg['num_output_frames']   = NUM_FRAMES
        cfg['save_with_index']     = True
        cfg['inference_iter']      = -1
        cfg['height']              = 720
        cfg['width']               = 960
        cfg['fps']                 = 24
        cfg['denoising_step_list'] = [1000, 800, 600, 400, 200]
        with open(tmp_cfg, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # VRAM polling
        vram_readings = []
        stop_evt = threading.Event()
        vram_thread = threading.Thread(target=_poll_vram, args=(stop_evt, vram_readings), daemon=True)
        vram_thread.start()

        t0 = time.time()
        status = 'error'
        try:
            ret = subprocess.run([
                sys.executable, 'inference.py',
                '--config_path', str(tmp_cfg),
            ], cwd=str(work_dir), env=env)
            if ret.returncode == 0:
                status = 'ok'
        except Exception as exc:
            print(f'  EXCEPTION: {exc}', file=sys.stderr)
        finally:
            stop_evt.set()
            vram_thread.join(timeout=10)

        dur  = round(time.time() - t0, 2)
        fps  = round(NUM_FRAMES * NUM_SAMPLES / dur, 2) if dur else ''
        vram = _vram_peak_gb(vram_readings)
        ram  = _ram_gb()

        # collect outputs: inference.py saves rank0-{idx}-{si}_{model_type}.mp4
        # idx is always 0 since we pass a single-prompt file
        new_files = sorted(out_base.glob(f'rank0-0-*_*.mp4'),
                           key=lambda f: f.stat().st_mtime)

        # rename to include prompt index and seed so filenames are unique and traceable
        renamed = []
        for f in new_files:
            stem = f.stem.replace('rank0-0-', f'rank0-{ti}-', 1)
            dst = f.with_name(f'{stem}_seed{seed}{f.suffix}')
            if dst != f:
                f.rename(dst)
            renamed.append(dst)

        if status == 'ok' and len(renamed) == NUM_SAMPLES:
            print(f'  done in {dur}s  ({fps} gen-fps)  VRAM {vram}GB  RAM {ram}GB')
            for si, mp4 in enumerate(renamed):
                writer.writerow([ti, cap, p['type'], si, seed,
                                 dur if si == 0 else '', fps if si == 0 else '',
                                 vram if si == 0 else '', ram if si == 0 else '',
                                 str(mp4), 'ok'])
            stats_f.flush()
            generated += NUM_SAMPLES
        else:
            print(f'  ERROR (returncode={ret.returncode}, files={len(renamed)}) - stopping')
            writer.writerow([ti, cap, p['type'], '', seed, dur, '', vram, ram, '', 'error'])
            stats_f.flush()
            stats_f.close()
            prompt_file.unlink(missing_ok=True)
            tmp_cfg.unlink(missing_ok=True)
            sys.exit(1)

    stats_f.close()
    prompt_file.unlink(missing_ok=True)
    tmp_cfg.unlink(missing_ok=True)

    elapsed_m = round((time.time() - t_start) / 60, 1)
    print(f'\n[vbench] done - generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_m}m')
    print(f'[vbench] stats -> {stats_path}')


if __name__ == '__main__':
    main()
