"""Run MolProbity validation (via Phenix) on structure files."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from base2backbone.runtime import PROGRESS_BAR_COLOR

DEFAULT_PHENIX_MODULE = 'phenix/2.0'
DEFAULT_MOLPROBITY_BIN_DIR = Path('/opt/shared_soft/phenix2.0/bin')

MOLPROBITY_METRIC_KEYS = (
    'clashscore',
    'num_clashes',
    'rna_bond_num_outliers',
    'rna_bond_num_total',
)


@dataclass(frozen=True)
class MolprobityRunResult:
    success: bool
    wall_time_s: float
    returncode: int
    clashscore: float | None
    num_clashes: int | None
    rna_bond_num_outliers: int | None
    rna_bond_num_total: int | None
    stdout: str
    stderr: str
    error: str | None = None

    def as_row_fields(self, prefix: str = 'molprobity') -> dict[str, Any]:
        return {
            f'{prefix}_success': self.success,
            f'{prefix}_wall_time_s': self.wall_time_s,
            f'{prefix}_clashscore': self.clashscore,
            f'{prefix}_num_clashes': self.num_clashes,
            f'{prefix}_rna_bond_num_outliers': self.rna_bond_num_outliers,
            f'{prefix}_rna_bond_num_total': self.rna_bond_num_total,
            f'{prefix}_error': self.error,
        }


def molprobity_tool_path(tool: str, bin_dir: Path | None = None) -> Path:
    root = bin_dir or DEFAULT_MOLPROBITY_BIN_DIR
    return root / f'molprobity.{tool}'


def _molprobity_shell_prefix(phenix_module: str) -> str:
    return (
        'source /etc/profile.d/modules.sh'
        f' && module load {shlex.quote(phenix_module)}'
    )


def _run_molprobity_tool(
    tool: str,
    structure_path: str | Path,
    *,
    extra_args: tuple[str, ...] = (),
    phenix_module: str = DEFAULT_PHENIX_MODULE,
    bin_dir: Path | None = None,
    timeout_s: int = 600,
) -> subprocess.CompletedProcess[str]:
    structure_path = str(Path(structure_path).resolve())
    tool_path = molprobity_tool_path(tool, bin_dir=bin_dir)
    phil_args = [
        f'model={structure_path}',
        'json=True',
        'verbose=False',
        *extra_args,
    ]
    cmd = (
        f'{_molprobity_shell_prefix(phenix_module)}'
        f' && {shlex.quote(str(tool_path))}'
        f' {" ".join(shlex.quote(arg) for arg in phil_args)}'
    )
    return subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def extract_json_object(stdout: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(stdout):
        start = stdout.find('{', idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(stdout, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and (
            'summary_results' in obj
            or 'validation_type' in obj
            or 'rna_bonds' in obj
        ):
            return obj
        idx = end
    return None


def _summary_bucket(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get('summary_results') or {}
    if not isinstance(summary, dict):
        return {}
    if '' in summary:
        return summary['']
    if len(summary) == 1:
        return next(iter(summary.values()))
    return {}


def metrics_from_clashscore_json(payload: dict[str, Any]) -> dict[str, float | int | None]:
    summary = _summary_bucket(payload)
    clashscore = summary.get('clashscore')
    num_clashes = summary.get('num_clashes')
    return {
        'clashscore': float(clashscore) if clashscore is not None else None,
        'num_clashes': int(num_clashes) if num_clashes is not None else None,
    }


def metrics_from_rna_validate_json(payload: dict[str, Any]) -> dict[str, int | None]:
    bonds = payload.get('rna_bonds') or {}
    summary = _summary_bucket(bonds if isinstance(bonds, dict) else {})
    num_outliers = summary.get('num_outliers')
    num_total = summary.get('num_total')
    return {
        'rna_bond_num_outliers': int(num_outliers) if num_outliers is not None else None,
        'rna_bond_num_total': int(num_total) if num_total is not None else None,
    }


def run_structure_molprobity(
    structure_path: str | Path,
    *,
    phenix_module: str = DEFAULT_PHENIX_MODULE,
    bin_dir: Path | None = None,
    timeout_s: int = 600,
    run_rna_validate: bool = True,
) -> MolprobityRunResult:
    """Run clashscore (+ optional rna_validate) and return flat MolProbity metrics."""
    structure_path = Path(structure_path)
    t0 = time.perf_counter()
    metrics: dict[str, float | int | None] = {
        'clashscore': None,
        'num_clashes': None,
        'rna_bond_num_outliers': None,
        'rna_bond_num_total': None,
    }
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    returncode = 0
    error = None

    try:
        clash_proc = _run_molprobity_tool(
            'clashscore',
            structure_path,
            phenix_module=phenix_module,
            bin_dir=bin_dir,
            timeout_s=timeout_s,
        )
        stdout_parts.append(clash_proc.stdout)
        stderr_parts.append(clash_proc.stderr)
        returncode = clash_proc.returncode
        if clash_proc.returncode != 0:
            error = (clash_proc.stderr or clash_proc.stdout)[-2000:]
        else:
            clash_json = extract_json_object(clash_proc.stdout)
            if clash_json is None:
                error = 'clashscore produced no parseable JSON'
            else:
                metrics.update(metrics_from_clashscore_json(clash_json))

        if run_rna_validate and error is None:
            rna_proc = _run_molprobity_tool(
                'rna_validate',
                structure_path,
                phenix_module=phenix_module,
                bin_dir=bin_dir,
                timeout_s=timeout_s,
            )
            stdout_parts.append(rna_proc.stdout)
            stderr_parts.append(rna_proc.stderr)
            if rna_proc.returncode != 0:
                error = (rna_proc.stderr or rna_proc.stdout)[-2000:]
                returncode = rna_proc.returncode
            else:
                rna_json = extract_json_object(rna_proc.stdout)
                if rna_json is None:
                    error = 'rna_validate produced no parseable JSON'
                else:
                    metrics.update(metrics_from_rna_validate_json(rna_json))
    except subprocess.TimeoutExpired as exc:
        error = f'timeout after {timeout_s}s'
        returncode = -1
        stderr_parts.append(str(exc))

    success = error is None and metrics['clashscore'] is not None
    return MolprobityRunResult(
        success=success,
        wall_time_s=time.perf_counter() - t0,
        returncode=returncode,
        clashscore=metrics['clashscore'],  # type: ignore[arg-type]
        num_clashes=metrics['num_clashes'],  # type: ignore[arg-type]
        rna_bond_num_outliers=metrics['rna_bond_num_outliers'],  # type: ignore[arg-type]
        rna_bond_num_total=metrics['rna_bond_num_total'],  # type: ignore[arg-type]
        stdout='\n'.join(stdout_parts),
        stderr='\n'.join(stderr_parts),
        error=error,
    )


def _molprobity_row_fields(
    row: dict[str, Any],
    *,
    prefix: str,
    phenix_module: str,
    bin_dir: Path | None,
    timeout_s: int,
    run_rna_validate: bool,
) -> dict[str, Any]:
    if not row.get('success') or not row.get('output_pdb'):
        return {
            f'{prefix}_success': False,
            f'{prefix}_error': 'structure benchmark failed',
        }
    mp = run_structure_molprobity(
        row['output_pdb'],
        phenix_module=phenix_module,
        bin_dir=bin_dir,
        timeout_s=timeout_s,
        run_rna_validate=run_rna_validate,
    )
    return mp.as_row_fields(prefix=prefix)


def _molprobity_row_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        row,
        prefix,
        phenix_module,
        bin_dir_str,
        timeout_s,
        run_rna_validate,
    ) = task
    bin_dir = Path(bin_dir_str) if bin_dir_str else None
    return _molprobity_row_fields(
        row,
        prefix=prefix,
        phenix_module=phenix_module,
        bin_dir=bin_dir,
        timeout_s=timeout_s,
        run_rna_validate=run_rna_validate,
    )


def annotate_benchmark_rows_with_molprobity(
    rows: list[dict[str, Any]],
    *,
    prefix: str = 'molprobity',
    phenix_module: str = DEFAULT_PHENIX_MODULE,
    bin_dir: Path | None = None,
    timeout_s: int = 600,
    run_rna_validate: bool = True,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Add MolProbity metrics to successful benchmark rows in place."""
    if not rows:
        return rows

    bin_dir_str = str(bin_dir) if bin_dir is not None else None
    task_kwargs = (
        prefix,
        phenix_module,
        bin_dir_str,
        timeout_s,
        run_rna_validate,
    )

    if max_workers == 1:
        for row in rows:
            row.update(_molprobity_row_fields(
                row,
                prefix=prefix,
                phenix_module=phenix_module,
                bin_dir=bin_dir,
                timeout_s=timeout_s,
                run_rna_validate=run_rna_validate,
            ))
        return rows

    tasks = [(row, *task_kwargs) for row in rows]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(
            ex.map(_molprobity_row_task, tasks),
            total=len(tasks),
            desc='MolProbity',
            leave=False,
            colour=PROGRESS_BAR_COLOR,
        ))
    for row, fields in zip(rows, results):
        row.update(fields)
    return rows


def summarize_molprobity_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str = 'molprobity',
) -> dict[str, float | int | None]:
    """Aggregate MolProbity metrics across benchmark rows."""
    summary: dict[str, float | int | None] = {
        'n_validated': 0,
        'n_total': len(rows),
    }
    for key in MOLPROBITY_METRIC_KEYS:
        field = f'{prefix}_{key}'
        vals = [
            row[field]
            for row in rows
            if row.get(f'{prefix}_success') and row.get(field) is not None
        ]
        summary[f'median_{key}'] = (
            float(np.median(np.asarray(vals, dtype=np.float64)))
            if vals else None
        )
        summary[f'mean_{key}'] = (
            float(np.mean(np.asarray(vals, dtype=np.float64)))
            if vals else None
        )
    summary['n_validated'] = sum(
        1 for row in rows if row.get(f'{prefix}_success')
    )
    return summary


def format_molprobity_summary(summary: dict[str, float | int | None]) -> str:
    parts = []
    for key in MOLPROBITY_METRIC_KEYS:
        median_val = summary.get(f'median_{key}')
        if median_val is not None:
            label = key.replace('_', ' ')
            if isinstance(median_val, float):
                parts.append(f'{label}={median_val:.3f}')
            else:
                parts.append(f'{label}={median_val}')
    parts.append(
        f'n={summary.get("n_validated", 0)}/{summary.get("n_total", 0)}',
    )
    return '  '.join(parts)


def print_molprobity_method_summaries(
    method_rows: dict[str, list[dict[str, Any]]],
    *,
    prefix: str = 'molprobity',
) -> dict[str, dict[str, float | int | None]]:
    summaries = {}
    print('\nMolProbity summaries:')
    for method_name, rows in method_rows.items():
        summary = summarize_molprobity_rows(rows, prefix=prefix)
        summaries[method_name] = summary
        print(f'  {method_name}: {format_molprobity_summary(summary)}')
    return summaries
