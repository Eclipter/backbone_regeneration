"""2×2 factorial matrix: prediction_target ∈ {epsilon, v} × train_loss_type ∈ {mse, rmse}.

Runs four training cells for a single chosen scenario and outputs a markdown comparison table.
Usage:
    BETA_SCHEDULE=cosine MAX_EPOCHS=3000 DEBUG_MATRIX_SCENARIOS=random_t_random_noise \
        python scripts/run_factorial_matrix.py
"""
import json
import os
import os.path as osp
import subprocess
import sys
import time
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
LOGS_ROOT = osp.join(REPO_ROOT, 'logs')

# All cells in the 2×2 factorial grid
FACTORIAL_GRID = [
    ('epsilon', 'mse'),
    ('epsilon', 'rmse'),
    ('v',       'mse'),
    ('v',       'rmse'),
]

ALL_SCENARIOS = {
    'fixed_t_fixed_noise':   {'debug_fixed_t': '100',  'debug_fixed_noise': '1'},
    'random_t_fixed_noise':  {'debug_fixed_t': 'none', 'debug_fixed_noise': '1'},
    'fixed_t_random_noise':  {'debug_fixed_t': '100',  'debug_fixed_noise': '0'},
    'random_t_random_noise': {'debug_fixed_t': 'none', 'debug_fixed_noise': '0'},
}


def read_scalar_stats(run_dir, tag):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    values = [item.value for item in ea.Scalars(tag)]
    return {'last': values[-1], 'min': min(values), 'count': len(values)}


def collect_run_summary(run_dir):
    metrics = {}
    for tag in ['train_rmse', 'train_x0_rmse', 'val_eps_rmse', 'val_rmse', 'test_eps_rmse', 'test_rmse']:
        metrics[tag] = read_scalar_stats(run_dir, tag)
    try:
        metrics['test_gen_rmse'] = read_scalar_stats(run_dir, 'test_gen_rmse')
    except Exception:
        pass
    return metrics


def run_cell(session_name, prediction_target, train_loss_type, scenario_params):
    cell_name = f'{prediction_target}__{train_loss_type}'
    run_root = os.getenv('DEBUG_MATRIX_RUN_ROOT', 'factorial_matrix')
    run_name = f'{run_root}/{session_name}/{cell_name}'
    beta_schedule = os.getenv('BETA_SCHEDULE', 'linear')
    env = os.environ.copy()
    env.update({
        'RUN_NAME':              run_name,
        'START_FROM_LAST_CKPT': '0',
        'OVERFIT_SINGLE_SAMPLE': '1',
        'OVERFIT_SAMPLE_INDEX':  '0',
        'DEBUG_SINGLE_DEVICE':   '1',
        'MAX_EPOCHS':            os.getenv('MAX_EPOCHS', '3000'),
        'BETA_SCHEDULE':         beta_schedule,
        'PREDICTION_TARGET':     prediction_target,
        'TRAIN_LOSS_TYPE':       train_loss_type,
        'DEBUG_EVAL_T':          '100',
        'DEBUG_EVAL_SNR':        os.getenv('DEBUG_EVAL_SNR', ''),
        'DEBUG_FIXED_T':         scenario_params['debug_fixed_t'],
        'DEBUG_FIXED_NOISE':     scenario_params['debug_fixed_noise'],
        'EVAL_FULL_SAMPLING':    os.getenv('EVAL_FULL_SAMPLING', '0'),
    })

    print(
        f'[start] {cell_name}  schedule={beta_schedule}  '
        f'scenario={scenario_params}',
        flush=True,
    )
    started = time.time()
    subprocess.run([sys.executable, 'src/train.py'], cwd=REPO_ROOT, env=env, check=True)
    elapsed = time.time() - started
    run_dir = osp.join(LOGS_ROOT, run_name)
    metrics = collect_run_summary(run_dir)
    print(f'[done]  {cell_name}  in {elapsed:.1f}s', flush=True)
    return {
        'cell':      cell_name,
        'run_name':  run_name,
        'elapsed_s': elapsed,
        'metrics':   metrics,
    }


def write_summary(path, session_name, scenario_name, results):
    with open(path, 'w') as f:
        json.dump({'session_name': session_name, 'scenario': scenario_name, 'results': results}, f, indent=2)


def write_markdown(path, session_name, scenario_name, results):
    has_gen = any('test_gen_rmse' in r['metrics'] for r in results)
    header = '| cell (target/loss) | train_rmse min | train_x0_rmse min | val_eps_rmse min | val_rmse min | test_rmse last'
    if has_gen:
        header += ' | test_gen_rmse last'
    header += ' | elapsed_s |'
    sep = '| --- | ---: | ---: | ---: | ---: | ---:'
    if has_gen:
        sep += ' | ---:'
    sep += ' | ---: |'
    lines = [
        f'# Factorial 2×2 matrix: {session_name}',
        f'',
        f'Scenario: **{scenario_name}**',
        f'',
        header,
        sep,
    ]
    for r in results:
        m = r['metrics']
        row = (
            '| {cell} | {a:.6f} | {b:.6f} | {c:.6f} | {d:.6f} | {e:.6f}'.format(
                cell=r['cell'],
                a=m['train_rmse']['min'],
                b=m['train_x0_rmse']['min'],
                c=m['val_eps_rmse']['min'],
                d=m['val_rmse']['min'],
                e=m['test_rmse']['last'],
            )
        )
        if has_gen:
            gen_val = m.get('test_gen_rmse', {}).get('last', float('nan'))
            row += f' | {gen_val:.6f}'
        row += f' | {r["elapsed_s"]:.1f} |'
        lines.append(row)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    session_name = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')

    # Choose scenario (single, default: random_t_random_noise)
    scenario_name = os.getenv('DEBUG_MATRIX_SCENARIOS', 'random_t_random_noise').split(',')[0].strip()
    scenario_params = ALL_SCENARIOS[scenario_name]

    # Optionally run only a subset of cells, e.g. FACTORIAL_CELLS=epsilon__mse,v__mse
    cell_filter = os.getenv('FACTORIAL_CELLS', '').strip()
    if cell_filter:
        allowed = set(cell_filter.split(','))
        grid = [(pt, lt) for pt, lt in FACTORIAL_GRID if f'{pt}__{lt}' in allowed]
    else:
        grid = FACTORIAL_GRID

    results = []
    for prediction_target, train_loss_type in grid:
        results.append(run_cell(session_name, prediction_target, train_loss_type, scenario_params))

    run_root = os.getenv('DEBUG_MATRIX_RUN_ROOT', 'factorial_matrix')
    summary_dir = osp.join(LOGS_ROOT, run_root, session_name)
    os.makedirs(summary_dir, exist_ok=True)
    write_summary(osp.join(summary_dir, 'summary.json'), session_name, scenario_name, results)
    write_markdown(osp.join(summary_dir, 'summary.md'), session_name, scenario_name, results)
    print(f'[summary] {osp.join(summary_dir, "summary.md")}', flush=True)


if __name__ == '__main__':
    main()
