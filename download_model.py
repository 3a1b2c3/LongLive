"""
Download LongLive-1.3B weights and base Wan2.1-T2V-1.3B model from HuggingFace.
Run from the repo root: python download_model.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

root = Path(__file__).parent

DOWNLOADS = [
    {
        'repo_id':   'Efficient-Large-Model/LongLive-1.3B',
        'local_dir': root / 'longlive_models',
        'verify':    ['models/longlive_base.pt',
                      'models/lora.pt'],
        'label':     'LongLive-1.3B',
    },
    {
        'repo_id':   'Wan-AI/Wan2.1-T2V-1.3B',
        'local_dir': root / 'wan_models' / 'Wan2.1-T2V-1.3B',
        'verify':    [],
        'label':     'Wan2.1-T2V-1.3B (base model)',
    },
]


def _check(ckpt: Path, label: str):
    if ckpt.exists():
        size_gb = ckpt.stat().st_size / 1024**3
        if size_gb < 0.01:
            print(f'  ERROR: {ckpt.name} is only {ckpt.stat().st_size} bytes — LFS pointer?')
            print('  Install git-lfs and run: git lfs pull')
        else:
            print(f'  OK: {label}  ({size_gb:.2f} GB)')
    else:
        print(f'  ERROR: {label} still missing after retry')


for d in DOWNLOADS:
    repo    = d['repo_id']
    loc     = Path(d['local_dir'])
    label   = d['label']
    print(f'\n=== Downloading {label} ===')
    print(f'    repo      : {repo}')
    print(f'    local_dir : {loc}')
    loc.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(loc),
            ignore_patterns=['*.metadata', '.cache/*'],
        )
        print(f'  snapshot_download complete')
    except Exception as exc:
        print(f'  ERROR during snapshot_download: {exc}')

    for rel in d['verify']:
        ckpt = loc / rel
        if not ckpt.exists():
            print(f'  WARNING: {rel} not found — retrying with hf_hub_download ...')
            try:
                hf_hub_download(repo_id=repo, filename=rel, local_dir=str(loc))
            except Exception as exc:
                print(f'  ERROR during hf_hub_download: {exc}')
        _check(ckpt, rel)

print('\n=== Summary ===')
for d in DOWNLOADS:
    loc = Path(d['local_dir'])
    print(f'\n{d["label"]} -> {loc}')
    if loc.exists():
        files = sorted(f for f in loc.rglob('*') if f.is_file())
        for f in files[:20]:
            print(f'  {f.relative_to(loc)}  ({f.stat().st_size / 1024**2:.1f} MB)')
        if len(files) > 20:
            print(f'  ... and {len(files) - 20} more files')
    else:
        print('  (directory not found)')
