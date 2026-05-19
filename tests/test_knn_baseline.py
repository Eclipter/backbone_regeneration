import numpy as np

from base2backbone.data import BACKBONE_ATOMS
from base2backbone.eval.knn_baseline import (
    KnnBaselineState,
    knn_match_indices,
    query_indices_by_pdb_from_metas,
)


def _meta(base_type: int, pdb_id: str) -> dict:
    return {
        'sample_key': f'{base_type}:{pdb_id}',
        'pdb_id': pdb_id,
        'deposit_group_id': None,
        'dna_sequence_tokens': frozenset(),
        'base_type': base_type,
        'is_edge': False,
    }


def test_knn_match_indices_expands_neighbors_until_valid_candidate():
    query_feats = np.asarray([[0.0]], dtype=np.float32)
    query_metas = [_meta(0, 'query')]
    ref_feats = np.asarray([[float(i)] for i in range(40)], dtype=np.float32)
    ref_metas = [
        _meta(0, 'query' if i < 32 else f'ref-{i}')
        for i in range(40)
    ]

    match_indices = knn_match_indices(
        query_feats,
        query_metas,
        ref_feats,
        ref_metas,
        allow_same_sample=False,
    )

    assert match_indices == [32]


def test_knn_match_indices_matches_queries_within_each_base_type():
    query_feats = np.asarray([[0.0], [10.0]], dtype=np.float32)
    query_metas = [_meta(0, 'query-a'), _meta(1, 'query-b')]
    ref_feats = np.asarray([[0.2], [9.8], [20.0]], dtype=np.float32)
    ref_metas = [_meta(0, 'ref-a'), _meta(1, 'ref-b'), _meta(0, 'ref-c')]

    match_indices = knn_match_indices(
        query_feats,
        query_metas,
        ref_feats,
        ref_metas,
        allow_same_sample=False,
    )

    assert match_indices == [0, 1]


def test_knn_fit_cache_reuses_fitted_indices():
    query_feats = np.asarray([[0.0]], dtype=np.float32)
    query_metas = [_meta(0, 'query')]
    ref_feats = np.asarray([[1.0], [2.0]], dtype=np.float32)
    ref_metas = [_meta(0, 'ref-a'), _meta(0, 'ref-b')]

    fit_cache = {}
    cache_key = ('feat',)
    knn_match_indices(
        query_feats,
        query_metas,
        ref_feats,
        ref_metas,
        allow_same_sample=False,
        fit_cache=fit_cache,
        fit_cache_key=cache_key,
    )
    knn_match_indices(
        query_feats,
        query_metas,
        ref_feats,
        ref_metas,
        allow_same_sample=False,
        fit_cache=fit_cache,
        fit_cache_key=cache_key,
    )

    assert len(fit_cache) == 1


def test_knn_baseline_state_indexes_query_samples_by_pdb():
    metas = [_meta(0, '1abc'), _meta(1, '1abc'), _meta(0, '2xyz')]
    state = KnnBaselineState(
        query_metas=metas,
        ref_locals=np.zeros((1, 6, 3), dtype=np.float32),
        match_indices=[0, None, 0],
    )

    assert state.query_indices_by_pdb == query_indices_by_pdb_from_metas(metas)
    assert state.query_indices_by_pdb['1abc'] == [0, 1]
    assert state.query_indices_by_pdb['2xyz'] == [2]


def test_knn_baseline_state_predictions_for_pdb_uses_indexed_meta():
    metas = [
        {
            **_meta(0, '1abc'),
            'origin_world': np.zeros(3, dtype=np.float32),
            'frame_world': np.eye(3, dtype=np.float32),
            'segid': 'A',
            'resid': 1,
        },
        {
            **_meta(1, '2xyz'),
            'origin_world': np.zeros(3, dtype=np.float32),
            'frame_world': np.eye(3, dtype=np.float32),
            'segid': 'B',
            'resid': 2,
        },
    ]
    state = KnnBaselineState(
        query_metas=metas,
        ref_locals=np.zeros((1, len(BACKBONE_ATOMS), 3), dtype=np.float32),
        match_indices=[0, None],
    )

    predictions = state.predictions_for_pdb('1abc')

    assert predictions
    assert all(key[0] == 'A' and key[1] == 1 for key in predictions)
