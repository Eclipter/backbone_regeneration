import numpy as np

from base2backbone.eval.knn_baseline import knn_match_indices


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
