"""Project-relative runtime paths and checkpoint discovery helpers."""

import os
import os.path as osp

import torch

PACKAGE_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
PROJECT_ROOT = osp.normpath(osp.join(PACKAGE_ROOT, '..', '..'))
LOG_DIR = osp.join(PROJECT_ROOT, 'logs')
MODEL_DIR = osp.join(PROJECT_ROOT, 'model')


def find_best_checkpoint(run_dir: str) -> str:
    """Locate the best-monitor checkpoint of a run via ModelCheckpoint state."""
    ckpt_dir = osp.join(run_dir, 'checkpoints')
    candidates = sorted(
        osp.join(ckpt_dir, filename)
        for filename in os.listdir(ckpt_dir)
        if filename.endswith('.ckpt')
    )
    if not candidates:
        raise FileNotFoundError(f'No *.ckpt files in {ckpt_dir}.')

    last_path = osp.join(ckpt_dir, 'last.ckpt')
    source = last_path if last_path in candidates else candidates[0]

    state = torch.load(source, map_location='cpu', weights_only=False)
    for key, cb_state in state.get('callbacks', {}).items():
        if 'ModelCheckpoint' not in key:
            continue
        best_path = cb_state.get('best_model_path', '')
        if not best_path:
            continue
        local = osp.join(ckpt_dir, osp.basename(best_path))
        if osp.isfile(local):
            return local
        if osp.isfile(best_path):
            return best_path

    raise RuntimeError(f'best_model_path not present in ModelCheckpoint state of {source}.')


def resolve_run_dir(run: str) -> str:
    """Map a user-facing experiment id (e.g. 'fixed_swa/baseline') to its log directory."""
    if osp.isabs(run):
        return run
    run_norm = osp.normpath(run)
    if run_norm.split(os.sep, 1)[0] == 'logs':
        run_norm = run_norm.split(os.sep, 1)[1] if os.sep in run_norm else ''
    return osp.normpath(osp.join(LOG_DIR, run_norm))
