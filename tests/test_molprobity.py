from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from base2backbone.eval.molprobity import (
    annotate_benchmark_rows_with_molprobity,
    metrics_from_molprobity_stdout,
    print_molprobity_summary,
    run_structure_molprobity,
    summarize_molprobity_rows,
)
from base2backbone.eval.local_geometry import backbone_predictions_from_matched_local
from base2backbone.eval.structure_runners import FixedTorsionSampler


def test_metrics_from_molprobity_stdout():
    stdout = (
        '  Clashscore            = 9.0\n'
        '  RMS(bonds)            =   0.021\n'
        '  RMS(angles)           =   2.345\n'
    )
    metrics = metrics_from_molprobity_stdout(stdout)
    assert metrics == {
        'clashscore': 9.0,
        'num_clashes': None,
        'bond_rmsd': 0.021,
        'angle_rmsd': 2.345,
    }


def test_fixed_torsion_sampler_expands_batch():
    sampler = FixedTorsionSampler(np.zeros(7), 0.25)
    batch = MagicMock()
    batch.torsions.device = 'cpu'
    batch.num_graphs = 3
    theta, tau = sampler.sample(batch)
    assert theta.shape == (3, 7)
    assert tau.shape == (3,)
    assert float(tau[0]) == pytest.approx(0.25)


def test_backbone_predictions_from_matched_local_uses_query_frame():
    from base2backbone.data import BACKBONE_ATOMS

    ref_local = np.zeros((len(BACKBONE_ATOMS), 3), dtype=np.float64)
    ref_local[0] = [1.0, 0.0, 0.0]
    origin = np.array([10.0, 0.0, 0.0])
    frame = np.eye(3)
    preds = backbone_predictions_from_matched_local(
        ref_local,
        origin,
        frame,
        'A',
        5,
    )
    assert preds[('A', 5, BACKBONE_ATOMS[0])][0] == pytest.approx(11.0)


@patch('base2backbone.eval.molprobity._run_molprobity_tool')
def test_run_structure_molprobity_parses_molprobity_summary(mock_tool, tmp_path):
    pdb_path = tmp_path / 'x.pdb'
    pdb_path.write_text('END\n', encoding='utf-8')
    stdout = (
        '  Clashscore            = 5.5\n'
        '  RMS(bonds)            =   0.014\n'
        '  RMS(angles)           =   1.234\n'
    )
    mock_tool.return_value = MagicMock(returncode=0, stdout=stdout, stderr='')

    result = run_structure_molprobity(pdb_path)

    assert result.success
    assert result.clashscore == pytest.approx(5.5)
    assert result.bond_rmsd == pytest.approx(0.014)
    assert result.angle_rmsd == pytest.approx(1.234)


@patch('base2backbone.eval.molprobity.run_structure_molprobity')
def test_annotate_benchmark_rows_with_molprobity(mock_run):
    from base2backbone.eval.molprobity import MolprobityRunResult

    mock_run.return_value = MolprobityRunResult(
        success=True,
        wall_time_s=1.0,
        returncode=0,
        clashscore=3.0,
        num_clashes=1,
        bond_rmsd=0.02,
        angle_rmsd=1.5,
        stdout='',
        stderr='',
    )
    rows = [
        {'success': True, 'output_pdb': '/tmp/a.pdb'},
        {'success': False, 'output_pdb': ''},
    ]
    annotate_benchmark_rows_with_molprobity(rows, max_workers=1)
    assert rows[0]['molprobity_clashscore'] == 3.0
    assert rows[1]['molprobity_success'] is False
    assert mock_run.call_count == 1


def test_print_molprobity_summary(capsys):
    rows = [
        {
            'molprobity_success': True,
            'molprobity_clashscore': 8.0,
            'molprobity_num_clashes': 1,
            'molprobity_bond_rmsd': 0.02,
            'molprobity_angle_rmsd': 1.5,
        },
    ]
    summary = print_molprobity_summary('Test MolProbity:', rows)
    captured = capsys.readouterr()
    assert 'Test MolProbity:' in captured.out
    assert 'clashscore=8.000' in captured.out
    assert summary['n_validated'] == 1


def test_summarize_molprobity_rows():
    rows = [
        {
            'molprobity_success': True,
            'molprobity_clashscore': 10.0,
            'molprobity_num_clashes': 2,
            'molprobity_bond_rmsd': 0.01,
            'molprobity_angle_rmsd': 1.0,
        },
        {
            'molprobity_success': True,
            'molprobity_clashscore': 20.0,
            'molprobity_num_clashes': 4,
            'molprobity_bond_rmsd': 0.03,
            'molprobity_angle_rmsd': 2.0,
        },
    ]
    summary = summarize_molprobity_rows(rows)
    assert summary['n_validated'] == 2
    assert summary['median_clashscore'] == pytest.approx(15.0)
