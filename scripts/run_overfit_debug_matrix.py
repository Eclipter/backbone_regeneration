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


def read_scalar_stats(run_dir, tag):
    event_accumulator = EventAccumulator(run_dir)
    event_accumulator.Reload()
    values = [item.value for item in event_accumulator.Scalars(tag)]
    return {
        'last': values[-1],
        'min': min(values),
        'count': len(values),
    }


def collect_run_summary(run_dir):
    metrics = {}
    for tag in ['train_rmse', 'train_x0_rmse', 'val_eps_rmse', 'val_rmse', 'test_eps_rmse', 'test_rmse']:
        metrics[tag] = read_scalar_stats(run_dir, tag)
    return metrics


def parse_scenarios():
    all_scenarios = {
        'fixed_t_fixed_noise': {'name': 'fixed_t_fixed_noise', 'debug_fixed_t': '100', 'debug_fixed_noise': '1'},
        'random_t_fixed_noise': {'name': 'random_t_fixed_noise', 'debug_fixed_t': 'none', 'debug_fixed_noise': '1'},
        'fixed_t_random_noise': {'name': 'fixed_t_random_noise', 'debug_fixed_t': '100', 'debug_fixed_noise': '0'},
        'random_t_random_noise': {'name': 'random_t_random_noise', 'debug_fixed_t': 'none', 'debug_fixed_noise': '0'},
    }
    scenario_names = os.getenv('DEBUG_MATRIX_SCENARIOS')
    if not scenario_names:
        return list(all_scenarios.values())
    selected = []
    for scenario_name in scenario_names.split(','):
        normalized_name = scenario_name.strip()
        if normalized_name:
            selected.append(all_scenarios[normalized_name])
    return selected


def write_summary(summary_path, session_name, results):
    payload = {
        'session_name': session_name,
        'results': results,
    }
    with open(summary_path, 'w') as file_obj:
        json.dump(payload, file_obj, indent=2)


def write_markdown(summary_path, session_name, results):
    lines = [
        f'# Overfit debug matrix: {session_name}',
        '',
        '| scenario | train_rmse min | train_x0_rmse min | val_eps_rmse min | val_rmse min | test_rmse last | elapsed_s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for result in results:
        metrics = result['metrics']
        lines.append(
            '| {scenario} | {train_rmse:.6f} | {train_x0_rmse:.6f} | {val_eps_rmse:.6f} | {val_rmse:.6f} | {test_rmse:.6f} | {elapsed_s:.1f} |'.format(
                scenario=result['scenario'],
                train_rmse=metrics['train_rmse']['min'],
                train_x0_rmse=metrics['train_x0_rmse']['min'],
                val_eps_rmse=metrics['val_eps_rmse']['min'],
                val_rmse=metrics['val_rmse']['min'],
                test_rmse=metrics['test_rmse']['last'],
                elapsed_s=result['elapsed_s'],
            )
        )
    with open(summary_path, 'w') as file_obj:
        file_obj.write('\n'.join(lines) + '\n')


def run_scenario(session_name, scenario):
    run_root = os.getenv('DEBUG_MATRIX_RUN_ROOT', 'debug_matrix')
    run_name = f'{run_root}/{session_name}/{scenario["name"]}'
    beta_schedule = os.getenv('BETA_SCHEDULE', 'linear')
    train_loss_type = os.getenv('TRAIN_LOSS_TYPE', 'mse')
    prediction_target = os.getenv('PREDICTION_TARGET', 'epsilon')
    eps_directional_head = os.getenv('EPS_DIRECTIONAL_HEAD', '0')
    eps_use_local_head = os.getenv('EPS_USE_LOCAL_HEAD', '1')
    eps_normalize_agg = os.getenv('EPS_NORMALIZE_AGG', '0')
    env = os.environ.copy()
    env.update(
        {
            'RUN_NAME': run_name,
            'START_FROM_LAST_CKPT': '0',
            'OVERFIT_SINGLE_SAMPLE': '1',
            'OVERFIT_SAMPLE_INDEX': '0',
            'DEBUG_SINGLE_DEVICE': '1',
            'MAX_EPOCHS': os.getenv('MAX_EPOCHS', '3000'),
            'BETA_SCHEDULE': beta_schedule,
            'TRAIN_LOSS_TYPE': train_loss_type,
            'PREDICTION_TARGET': prediction_target,
            'DEBUG_EVAL_T': '100',
            'DEBUG_FIXED_T': scenario['debug_fixed_t'],
            'DEBUG_FIXED_NOISE': scenario['debug_fixed_noise'],
        }
    )

    print(
        f'[start] {scenario["name"]} schedule={beta_schedule} loss={train_loss_type} '
        f'target={prediction_target} directional={eps_directional_head} '
        f'local_head={eps_use_local_head} normalize_agg={eps_normalize_agg}',
        flush=True
    )
    started_at = time.time()
    subprocess.run(
        [sys.executable, 'src/train.py'],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    elapsed_s = time.time() - started_at
    run_dir = osp.join(LOGS_ROOT, run_name)
    metrics = collect_run_summary(run_dir)
    print(f'[done] {scenario["name"]} in {elapsed_s:.1f}s', flush=True)
    return {
        'scenario': scenario['name'],
        'run_name': run_name,
        'elapsed_s': elapsed_s,
        'metrics': metrics,
    }


def main():
    session_name = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    scenarios = parse_scenarios()

    results = []
    for scenario in scenarios:
        results.append(run_scenario(session_name, scenario))

    run_root = os.getenv('DEBUG_MATRIX_RUN_ROOT', 'debug_matrix')
    summary_dir = osp.join(LOGS_ROOT, run_root, session_name)
    os.makedirs(summary_dir, exist_ok=True)
    write_summary(osp.join(summary_dir, 'summary.json'), session_name, results)
    write_markdown(osp.join(summary_dir, 'summary.md'), session_name, results)
    print(f'[summary] {osp.join(summary_dir, "summary.md")}', flush=True)


if __name__ == '__main__':
    main()
