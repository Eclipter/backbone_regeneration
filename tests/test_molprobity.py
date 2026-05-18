import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from base2backbone.eval.molprobity import (
    annotate_benchmark_rows_with_molprobity,
    extract_json_object,
    metrics_from_clashscore_json,
    metrics_from_rna_validate_json,
    run_structure_molprobity,
    summarize_molprobity_rows,
)
from base2backbone.eval.local_geometry import backbone_predictions_from_matched_local
from base2backbone.eval.structure_runners import FixedTorsionSampler


def test_extract_json_object_finds_summary_payload():
    stdout = 'noise\n{"summary_results": {"": {"clashscore": 12.5, "num_clashes": 3}}}\n'
    payload = extract_json_object(stdout)
    assert payload is not None
    assert payload['summary_results']['']['clashscore'] == 12.5


def test_metrics_from_clashscore_json():
    payload = {'summary_results': {'': {'clashscore': 9.0, 'num_clashes': 2}}}
    metrics = metrics_from_clashscore_json(payload)
    assert metrics == {'clashscore': 9.0, 'num_clashes': 2}


def test_metrics_from_rna_validate_json():
    payload = {
        'rna_bonds': {
            'summary_results': {
                '': {'num_outliers': 4, 'num_total': 100},
            },
        },
    }
    metrics = metrics_from_rna_validate_json(payload)
    assert metrics == {
        'rna_bond_num_outliers': 4,
        'rna_bond_num_total': 100,
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
def test_run_structure_molprobity_parses_clashscore_and_rna(mock_tool, tmp_path):
    pdb_path = tmp_path / 'x.pdb'
    pdb_path.write_text('END\n', encoding='utf-8')
    clash_json = json.dumps({
        'summary_results': {'': {'clashscore': 5.5, 'num_clashes': 1}},
    })
    rna_json = json.dumps({
        'rna_bonds': {
            'summary_results': {'': {'num_outliers': 2, 'num_total': 20}},
        },
    })
    mock_tool.side_effect = [
        MagicMock(returncode=0, stdout=clash_json, stderr=''),
        MagicMock(returncode=0, stdout=rna_json, stderr=''),
    ]

    result = run_structure_molprobity(pdb_path, run_rna_validate=True)

    assert result.success
    assert result.clashscore == pytest.approx(5.5)
    assert result.num_clashes == 1
    assert result.rna_bond_num_outliers == 2
    assert result.rna_bond_num_total == 20


@patch('base2backbone.eval.molprobity.run_structure_molprobity')
def test_annotate_benchmark_rows_with_molprobity(mock_run):
    from base2backbone.eval.molprobity import MolprobityRunResult

    mock_run.return_value = MolprobityRunResult(
        success=True,
        wall_time_s=1.0,
        returncode=0,
        clashscore=3.0,
        num_clashes=1,
        rna_bond_num_outliers=0,
        rna_bond_num_total=10,
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


def test_summarize_molprobity_rows():
    rows = [
        {
            'molprobity_success': True,
            'molprobity_clashscore': 10.0,
            'molprobity_num_clashes': 2,
            'molprobity_rna_bond_num_outliers': 1,
            'molprobity_rna_bond_num_total': 50,
        },
        {
            'molprobity_success': True,
            'molprobity_clashscore': 20.0,
            'molprobity_num_clashes': 4,
            'molprobity_rna_bond_num_outliers': 3,
            'molprobity_rna_bond_num_total': 60,
        },
    ]
    summary = summarize_molprobity_rows(rows)
    assert summary['n_validated'] == 2
    assert summary['median_clashscore'] == pytest.approx(15.0)
