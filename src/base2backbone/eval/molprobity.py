"""Run MolProbity validation (via Phenix) on structure files."""

from __future__ import annotations

import json
import re
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
    'bond_rmsd',
    'angle_rmsd',
)


@dataclass(frozen=True)
class MolprobityRunResult:
    success: bool
    wall_time_s: float
    returncode: int
    clashscore: float | None
    num_clashes: int | None
    bond_rmsd: float | None
    angle_rmsd: float | None
    stdout: str
    stderr: str
    error: str | None = None

    def as_row_fields(self, prefix: str = 'molprobity') -> dict[str, Any]:
        return {
            f'{prefix}_success': self.success,
            f'{prefix}_wall_time_s': self.wall_time_s,
            f'{prefix}_clashscore': self.clashscore,
            f'{prefix}_num_clashes': self.num_clashes,
            f'{prefix}_bond_rmsd': self.bond_rmsd,
            f'{prefix}_angle_rmsd': self.angle_rmsd,
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
    if tool == 'molprobity':
        tool_cmd = 'phenix.molprobity'
        phil_args = [structure_path, *extra_args]
    else:
        tool_cmd = shlex.quote(str(molprobity_tool_path(tool, bin_dir=bin_dir)))
        phil_args = [
            f'model={structure_path}',
            'json=True',
            'verbose=False',
            *extra_args,
        ]
    cmd = (
        f'{_molprobity_shell_prefix(phenix_module)}'
        f' && {tool_cmd}'
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


def _search_metric(
    stdout: str,
    patterns: tuple[str, ...],
) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, stdout, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            return match.group(1)
    return None


def metrics_from_molprobity_stdout(stdout: str, stderr: str = '') -> dict[str, float | int | None]:
    # phenix.molprobity summary uses "Clashscore =", "RMS(bonds)", "RMS(angles)".
    text = f'{stdout}\n{stderr}'
    clashscore = _search_metric(
        text,
        (
            r'Clashscore\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            r'All-atom Clashscore\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        ),
    )
    num_clashes = _search_metric(
        text,
        (
            r'Num(?:ber of)? clashes\s*[:=]\s*([0-9]+)',
            r'Clashes\s*[:=]\s*([0-9]+)',
        ),
    )
    bond_rmsd = _search_metric(
        text,
        (
            r'RMS\(bonds\)\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            r'covalent geometry\s*:\s*bond\s+([0-9]+(?:\.[0-9]+)?)',
            r'^\s*Bond\s*:\s*([0-9]+(?:\.[0-9]+)?)\s',
        ),
    )
    angle_rmsd = _search_metric(
        text,
        (
            r'RMS\(angles\)\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            r'covalent geometry\s*:\s*angle\s+([0-9]+(?:\.[0-9]+)?)',
            r'^\s*Angle\s*:\s*([0-9]+(?:\.[0-9]+)?)\s',
        ),
    )
    return {
        'clashscore': float(clashscore) if clashscore is not None else None,
        'num_clashes': int(num_clashes) if num_clashes is not None else None,
        'bond_rmsd': float(bond_rmsd) if bond_rmsd is not None else None,
        'angle_rmsd': float(angle_rmsd) if angle_rmsd is not None else None,
    }


def run_structure_molprobity(
    structure_path: str | Path,
    *,
    phenix_module: str = DEFAULT_PHENIX_MODULE,
    bin_dir: Path | None = None,
    timeout_s: int = 600,
    run_rna_validate: bool = True,
) -> MolprobityRunResult:
    """Run phenix.molprobity and return flat MolProbity metrics."""
    structure_path = Path(structure_path)
    t0 = time.perf_counter()
    metrics: dict[str, float | int | None] = {
        'clashscore': None,
        'num_clashes': None,
        'bond_rmsd': None,
        'angle_rmsd': None,
    }
    returncode = 0
    error = None

    try:
        molprobity_proc = _run_molprobity_tool(
            'molprobity',
            structure_path,
            phenix_module=phenix_module,
            bin_dir=bin_dir,
            timeout_s=timeout_s,
        )
        returncode = molprobity_proc.returncode
        if molprobity_proc.returncode != 0:
            error = (molprobity_proc.stderr or molprobity_proc.stdout)[-2000:]
        else:
            metrics.update(metrics_from_molprobity_stdout(
                molprobity_proc.stdout,
                molprobity_proc.stderr,
            ))
    except subprocess.TimeoutExpired as exc:
        error = f'timeout after {timeout_s}s'
        returncode = -1
        molprobity_stdout = ''
        molprobity_stderr = str(exc)
    else:
        molprobity_stdout = molprobity_proc.stdout
        molprobity_stderr = molprobity_proc.stderr

    success = error is None and any(metrics[key] is not None for key in MOLPROBITY_METRIC_KEYS)
    return MolprobityRunResult(
        success=success,
        wall_time_s=time.perf_counter() - t0,
        returncode=returncode,
        clashscore=metrics['clashscore'],  # type: ignore[arg-type]
        num_clashes=metrics['num_clashes'],  # type: ignore[arg-type]
        bond_rmsd=metrics['bond_rmsd'],  # type: ignore[arg-type]
        angle_rmsd=metrics['angle_rmsd'],  # type: ignore[arg-type]
        stdout=molprobity_stdout,
        stderr=molprobity_stderr,
        error=error,
    )


def _molprobity_cache_path(output_pdb: str | Path) -> Path:
    path = Path(output_pdb)
    return path.with_name(f'{path.name}.molprobity.json')


def _load_molprobity_cache(output_pdb: str | Path, *, prefix: str) -> dict[str, Any] | None:
    cache_path = _molprobity_cache_path(output_pdb)
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get('prefix') != prefix:
        return None
    fields = payload.get('fields')
    return fields if isinstance(fields, dict) else None


def _save_molprobity_cache(
    output_pdb: str | Path,
    fields: dict[str, Any],
    *,
    prefix: str,
) -> None:
    cache_path = _molprobity_cache_path(output_pdb)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({'prefix': prefix, 'fields': fields}, indent=2),
        encoding='utf-8',
    )


def _molprobity_row_fields(
    row: dict[str, Any],
    *,
    prefix: str,
    phenix_module: str,
    bin_dir: Path | None,
    timeout_s: int,
    run_rna_validate: bool,
    resume: bool = True,
) -> dict[str, Any]:
    if not row.get('success') or not row.get('output_pdb'):
        return {
            f'{prefix}_success': False,
            f'{prefix}_error': 'structure benchmark failed',
        }
    if resume:
        cached = _load_molprobity_cache(row['output_pdb'], prefix=prefix)
        if cached is not None and cached.get(f'{prefix}_success'):
            return cached
    mp = run_structure_molprobity(
        row['output_pdb'],
        phenix_module=phenix_module,
        bin_dir=bin_dir,
        timeout_s=timeout_s,
        run_rna_validate=run_rna_validate,
    )
    fields = mp.as_row_fields(prefix=prefix)
    if resume and row.get('output_pdb'):
        _save_molprobity_cache(row['output_pdb'], fields, prefix=prefix)
    return fields


def _molprobity_row_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        row,
        prefix,
        phenix_module,
        bin_dir_str,
        timeout_s,
        run_rna_validate,
        resume,
    ) = task
    bin_dir = Path(bin_dir_str) if bin_dir_str else None
    return _molprobity_row_fields(
        row,
        prefix=prefix,
        phenix_module=phenix_module,
        bin_dir=bin_dir,
        timeout_s=timeout_s,
        run_rna_validate=run_rna_validate,
        resume=resume,
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
    resume: bool = True,
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
        resume,
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
                resume=resume,
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


def print_molprobity_summary(
    title: str,
    rows: list[dict[str, Any]],
    *,
    prefix: str = 'molprobity',
) -> dict[str, float | int | None]:
    """Print one MolProbity aggregate block (same layout as per-method lines)."""
    summary = summarize_molprobity_rows(rows, prefix=prefix)
    print(f'\n{title}')
    print(f'  {format_molprobity_summary(summary)}')
    return summary


def print_molprobity_method_summaries(
    method_rows: dict[str, list[dict[str, Any]]],
    *,
    prefix: str = 'molprobity',
) -> dict[str, dict[str, float | int | None]]:
    summaries = {}
    print('\nMolProbity summaries:')
    for method_name, rows in method_rows.items():
        summaries[method_name] = print_molprobity_summary(
            method_name,
            rows,
            prefix=prefix,
        )
    return summaries
