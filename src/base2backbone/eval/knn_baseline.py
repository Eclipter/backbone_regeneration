"""kNN backbone baseline: feature indexing, matching, and structure export."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..data import BACKBONE_ATOMS
from ..dataset import WindowTargetDataset
from .local_geometry import (
    backbone_predictions_from_matched_local,
    local_backbone_rmsd,
)

KNN_INITIAL_NEIGHBORS = 32
KNN_NEIGHBOR_GROWTH = 4


def payload_key(dataset: WindowTargetDataset, target_type: int) -> str:
    return 'central' if target_type == dataset.CENTRAL else 'edge'


def feature_blocks(base_data, payload):
    return {
        'rel_origins': payload['rel_origins'],
        'rel_frames': payload['rel_frames'],
        'pair_rel_origins': payload['pair_rel_origins'],
        'pair_rel_frames': payload['pair_rel_frames'],
        'base_types': base_data.base_types,
        'has_pair_nt': base_data.has_pair_nt.float(),
        'chain_end_class': base_data.chain_end_class,
        'is_target_nt': payload['is_target_nt'].float(),
    }


def feature_dim_and_slices(base_data, payload):
    slices = {}
    offset = 0
    for name, arr in feature_blocks(base_data, payload).items():
        size = int(arr.numel())
        slices[name] = slice(offset, offset + size)
        offset += size
    return offset, slices


def knn_feature_into(base_data, payload, out):
    offset = 0
    for arr in feature_blocks(base_data, payload).values():
        flat = arr.reshape(-1).numpy()
        out[offset:offset + flat.size] = flat
        offset += flat.size


def dataset_samples_cached(dataset, label, pdb_meta_cache, progress_bar=None):
    n_samples = len(dataset)
    n_bb = len(BACKBONE_ATOMS)
    metas: list[dict[str, Any]] = []

    first_w_idx, first_target_type = dataset.virtual_entries[0]
    first_base = dataset.base.get(first_w_idx)
    first_payloads = cast(
        dict[str, dict[str, Any]],
        getattr(first_base, '_precomputed_target_payloads'),
    )
    first_payload = first_payloads[payload_key(dataset, first_target_type)]
    feat_dim, feat_slices = feature_dim_and_slices(first_base, first_payload)

    feats = np.empty((n_samples, feat_dim), dtype=np.float32)
    locals_ = np.empty((n_samples, n_bb, 3), dtype=np.float32)
    base_cache: dict[int, Any] = {first_w_idx: first_base}

    iterator = range(n_samples)
    if progress_bar is not None:
        iterator = progress_bar(iterator)

    for i in iterator:
        w_idx, target_type = dataset.virtual_entries[i]
        if w_idx not in base_cache:
            base_cache[w_idx] = dataset.base.get(w_idx)
        base = base_cache[w_idx]
        payloads = cast(
            dict[str, dict[str, Any]],
            getattr(base, '_precomputed_target_payloads'),
        )
        payload = payloads[payload_key(dataset, target_type)]
        ti = int(payload['target_nt_idx'].item())
        path = dataset.base.data_list[w_idx]
        pdb_id = Path(path).parent.name
        bb_w = base.bb_xyz_world[ti].numpy()
        o = base.nt_origins_world[ti].numpy()
        R = base.nt_frames_world[ti].numpy()
        knn_feature_into(base, payload, feats[i])
        locals_[i] = ((bb_w - o) @ R).astype(np.float32)
        pdb_meta = pdb_meta_cache[pdb_id]
        metas.append({
            'sample_key': f'{label}:{w_idx}:{target_type}',
            'pdb_id': pdb_id,
            'segid': payload['target_segid'],
            'resid': int(payload['target_resid']),
            'origin_world': o,
            'frame_world': R,
            'deposit_group_id': pdb_meta['deposit_group_id'],
            'dna_sequence_tokens': pdb_meta['dna_sequence_tokens'],
            'base_type': int(base.base_types[ti].argmax().item()),
            'is_edge': bool(base.is_chain_edge_nt[ti].item()),
        })
    return feats, locals_, metas, feat_slices


def select_feature_columns(feats, feat_slices, feature_names):
    return np.concatenate(
        [feats[:, feat_slices[name]] for name in feature_names],
        axis=1,
    )


def fit_knn_indices(ref_feats, ref_metas):
    scaler = StandardScaler()
    ref_feats_scaled = np.asarray(scaler.fit_transform(ref_feats), dtype=np.float32)
    ref_indices_lists: defaultdict[int, list[int]] = defaultdict(list)
    for idx, meta in enumerate(ref_metas):
        ref_indices_lists[meta['base_type']].append(idx)
    ref_indices_by_base = {
        base_type: np.asarray(indices, dtype=np.int64)
        for base_type, indices in ref_indices_lists.items()
    }
    nn_by_base = {}
    for base_type, indices in ref_indices_by_base.items():
        nn = NearestNeighbors(algorithm='auto')
        nn.fit(ref_feats_scaled[indices])
        nn_by_base[base_type] = nn
    return scaler, ref_indices_by_base, nn_by_base


def candidate_allowed(query_meta, ref_meta, allow_same_sample):
    if not allow_same_sample and ref_meta['sample_key'] == query_meta['sample_key']:
        return False
    if ref_meta['pdb_id'] == query_meta['pdb_id']:
        return False
    if (
        query_meta['deposit_group_id'] is not None
        and ref_meta['deposit_group_id'] == query_meta['deposit_group_id']
    ):
        return False
    if query_meta['dna_sequence_tokens'] & ref_meta['dna_sequence_tokens']:
        return False
    return True


def _batched_matches_for_base_type(
    query_indices,
    query_feats_scaled,
    query_metas,
    ref_indices,
    ref_metas,
    nn,
    *,
    allow_same_sample: bool,
):
    match_indices: list[int | None] = [None] * len(query_indices)
    if len(query_indices) == 0 or len(ref_indices) == 0:
        return match_indices

    query_batch = query_feats_scaled[np.asarray(query_indices, dtype=np.int64)]
    pending_positions = np.arange(len(query_indices), dtype=np.int64)
    n_ref = len(ref_indices)
    n_neighbors = min(KNN_INITIAL_NEIGHBORS, n_ref)

    while len(pending_positions) > 0:
        _, neighbors = nn.kneighbors(
            query_batch[pending_positions],
            n_neighbors=n_neighbors,
        )
        next_pending: list[int] = []
        for batch_row, query_pos in enumerate(pending_positions):
            query_meta = query_metas[query_indices[int(query_pos)]]
            match_idx = None
            for local_idx in neighbors[batch_row]:
                ref_idx = int(ref_indices[int(local_idx)])
                if candidate_allowed(query_meta, ref_metas[ref_idx], allow_same_sample):
                    match_idx = ref_idx
                    break
            if match_idx is None and n_neighbors < n_ref:
                next_pending.append(int(query_pos))
                continue
            match_indices[int(query_pos)] = match_idx
        if not next_pending or n_neighbors == n_ref:
            break
        pending_positions = np.asarray(next_pending, dtype=np.int64)
        n_neighbors = min(n_ref, n_neighbors * KNN_NEIGHBOR_GROWTH)
    return match_indices


def knn_match_indices(
    query_feats,
    query_metas,
    ref_feats,
    ref_metas,
    *,
    allow_same_sample: bool,
):
    scaler, ref_indices_by_base, nn_by_base = fit_knn_indices(ref_feats, ref_metas)
    query_feats_scaled = np.asarray(scaler.transform(query_feats), dtype=np.float32)
    match_indices: list[int | None] = [None] * len(query_metas)
    query_indices_by_base: defaultdict[int, list[int]] = defaultdict(list)
    for i, meta in enumerate(query_metas):
        query_indices_by_base[meta['base_type']].append(i)
    for base_type, query_indices in query_indices_by_base.items():
        ref_indices = ref_indices_by_base.get(base_type)
        if ref_indices is None or len(ref_indices) == 0:
            continue
        base_matches = _batched_matches_for_base_type(
            query_indices,
            query_feats_scaled,
            query_metas,
            ref_indices,
            ref_metas,
            nn_by_base[base_type],
            allow_same_sample=allow_same_sample,
        )
        for query_idx, match_idx in zip(query_indices, base_matches):
            match_indices[query_idx] = match_idx
    return match_indices


def run_knn_protocol(
    name,
    query_feats,
    query_locals,
    query_metas,
    ref_feats,
    ref_locals,
    ref_metas,
    allow_same_sample,
    *,
    print_summary=None,
):
    match_indices = knn_match_indices(
        query_feats,
        query_metas,
        ref_feats,
        ref_metas,
        allow_same_sample=allow_same_sample,
    )
    rmsds: list[float] = []
    is_edge: list[bool] = []
    missing = 0
    for i, meta in enumerate(query_metas):
        match_idx = match_indices[i]
        if match_idx is None:
            rmsds.append(np.nan)
            missing += 1
        else:
            rmsds.append(local_backbone_rmsd(ref_locals[match_idx], query_locals[i]))
        is_edge.append(meta['is_edge'])
    rmsds_arr = np.asarray(rmsds, dtype=np.float64)
    is_edge_arr = np.asarray(is_edge, dtype=bool)
    if print_summary is not None:
        print_summary(
            f'kNN backbone RMSD (local frame) [{name}]',
            rmsds_arr,
            is_edge_arr,
        )
        print(
            f'  {"eligible":>10}: '
            f'{np.isfinite(rmsds_arr).sum()}/{len(rmsds_arr)} '
            f'(missing={missing})'
        )
    return rmsds_arr, is_edge_arr, match_indices


@dataclass
class KnnBaselineState:
    query_metas: list[dict[str, Any]]
    ref_locals: np.ndarray
    match_indices: list[int | None]

    def predictions_for_pdb(self, pdb_id: str) -> dict[tuple[str, int, str], np.ndarray]:
        predictions: dict[tuple[str, int, str], np.ndarray] = {}
        for sample_idx, meta in enumerate(self.query_metas):
            if meta['pdb_id'] != pdb_id:
                continue
            match_idx = self.match_indices[sample_idx]
            if match_idx is None:
                continue
            residue_preds = backbone_predictions_from_matched_local(
                self.ref_locals[match_idx],
                meta['origin_world'],
                meta['frame_world'],
                meta['segid'],
                meta['resid'],
            )
            predictions.update(residue_preds)
        return predictions


def build_knn_baseline_state(
    query_feats,
    query_metas,
    ref_feats,
    ref_locals,
    ref_metas,
    *,
    allow_same_sample: bool = False,
) -> KnnBaselineState:
    match_indices = knn_match_indices(
        query_feats,
        query_metas,
        ref_feats,
        ref_metas,
        allow_same_sample=allow_same_sample,
    )
    return KnnBaselineState(
        query_metas=query_metas,
        ref_locals=ref_locals,
        match_indices=match_indices,
    )
