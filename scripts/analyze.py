# %% Imports
import os.path as osp
import random
import shlex
import shutil
import subprocess
import tempfile
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import py3Dmol
import seaborn as sns
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from config import BASE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from base2backbone.data import BACKBONE_ATOMS, BASE_TO_INDEX, parse_dna
from base2backbone.dataset import DNADataModule
from base2backbone.eval import (backbone_local_in_target_frame,
                                backbone_segments_from_local_coords,
                                bond_segments_from_nt_graph,
                                coords_local_per_nt,
                                find_window_matching_sample,
                                local_backbone_rmsd, ordered_backbone_segments,
                                phosphodiester_segments_local,
                                world_to_local_np)
from base2backbone.geometry.backbone import build_backbone_local
from base2backbone.inference import \
    _build_output_universe as build_output_universe
from base2backbone.inference import \
    _predict_backbone_from_chain_records as predict_backbone_from_chain_records
from base2backbone.inference import write_structure
from base2backbone.io import default_atoms_provider
from base2backbone.runtime import (PROGRESS_BAR_COLOR, collect_scalar_history,
                                   load_analysis_run_artifacts)
from base2backbone.torsion_constants import N_TORSIONS, TAU_M_MAX, TAU_M_MIN

IDX_TO_BASE = {v: k for k, v in BASE_TO_INDEX.items()}

TOR_NAMES = ['α', 'β', 'γ', 'ε', 'ζ', 'χ', 'phase']

# %% Load prerequisites
run_id = 'torsions/10/CLOSURE_LOSS_WEIGHT=0.001_CLOSURE_ANGLE_WEIGHT=0.3'

artifacts = load_analysis_run_artifacts(run_id)
run_dir = artifacts.run_dir
ckpt_path = artifacts.ckpt_path
event_files = artifacts.event_files
model = artifacts.model
device = artifacts.device
analysis_hparams = dict(model.hparams)
_analysis_edge_weight_raw = analysis_hparams.get('edge_weight')
analysis_edge_weight = (
    float(_analysis_edge_weight_raw)
    if _analysis_edge_weight_raw is not None
    else float(cast(float, BASE['EDGE_WEIGHT']))
)
_analysis_seed_raw = analysis_hparams.get('seed')
analysis_seed = (
    int(_analysis_seed_raw)
    if _analysis_seed_raw is not None
    else int(cast(int, BASE['SEED']))
)
_analysis_manifest_raw = (
    analysis_hparams.get('dataset_manifest')
    if analysis_hparams.get('dataset_manifest') is not None
    else BASE['DATASET_MANIFEST']
)
analysis_manifest = (
    None
    if _analysis_manifest_raw is None
    else str(_analysis_manifest_raw)
)
dm = DNADataModule(
    batch_size=1,
    edge_weight=analysis_edge_weight,
    seed=analysis_seed,
    dataset_manifest=analysis_manifest,
)
dm.setup()
train_dataset = dm.train_dataset
val_dataset = dm.val_dataset
test_dataset = dm.test_dataset
dataset = train_dataset.base
target_modes = artifacts.target_modes
mode_colors = {
    'avg': 'indigo',
    'central': PROGRESS_BAR_COLOR,
    'edge': 'violet',
}
mode_linestyles = {'avg': '-', 'central': '--', 'edge': '--'}
print(f'device: {device}')


class CheckpointSamplerAdapter:
    """Wrap checkpoint ``sample``; ``num_timesteps=None`` uses ``model.hparams`` default."""

    def __init__(self, checkpoint_model, num_timesteps: int | None):
        self.checkpoint_model = checkpoint_model
        self.num_timesteps = num_timesteps

    def sample(self, batch):
        if self.num_timesteps is None:
            return self.checkpoint_model.sample(batch)
        return self.checkpoint_model.sample(
            batch,
            num_timesteps=self.num_timesteps,
        )


RMSD_SUMMARY_FIELDS = (
    ('avg_median', 'avg median'),
    ('avg_mean', 'avg mean'),
    ('avg_p90', 'avg p90'),
    ('central_mean', 'central mean'),
    ('edge_mean', 'edge mean'),
)


def finite_rmsd_values(rmsds, mask=None):
    values = np.asarray(rmsds, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if mask is not None:
        finite_mask &= np.asarray(mask, dtype=bool)
    return values[finite_mask]


def _rmsd_mean(values):
    vals = finite_rmsd_values(values)
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def rmsd_summary_metrics(rmsds, is_edge=None):
    values = np.asarray(rmsds, dtype=np.float64)
    avg_values = finite_rmsd_values(values)
    summary = {
        'avg_median': None,
        'avg_mean': None,
        'avg_p90': None,
        'central_mean': None,
        'edge_mean': None,
        'n_finite': int(len(avg_values)),
        'n_total': int(len(values)),
    }
    if len(avg_values) == 0:
        return summary

    summary.update({
        'avg_median': float(np.median(avg_values)),
        'avg_mean': float(np.mean(avg_values)),
        'avg_p90': float(np.percentile(avg_values, 90)),
    })
    if is_edge is None:
        return summary

    edge_mask = np.asarray(is_edge, dtype=bool)
    if len(edge_mask) != len(values):
        raise ValueError('is_edge must have the same length as rmsds')
    summary['central_mean'] = _rmsd_mean(values[~edge_mask])
    summary['edge_mean'] = _rmsd_mean(values[edge_mask])
    return summary


def format_rmsd_value(value):
    if value is None:
        return 'n/a'
    return f'{value:.3f} A'


def format_rmsd_summary_line(summary):
    metric_text = [
        f'{label}={format_rmsd_value(summary[key])}'
        for key, label in RMSD_SUMMARY_FIELDS
    ]
    metric_text.append(f'n={summary["n_finite"]}/{summary["n_total"]}')
    return '  '.join(metric_text)


def print_rmsd_metric_summary(title, summary):
    print(f'\n{title}')
    if summary['n_finite'] == 0:
        print('  no finite RMSDs')
        return summary
    for key, label in RMSD_SUMMARY_FIELDS:
        print(f'  {label:>12}: {format_rmsd_value(summary[key])}')
    print(f'  {"n":>12}: {summary["n_finite"]}/{summary["n_total"]}')
    return summary


def print_rmsd_summary(title, rmsds, is_edge=None):
    summary = rmsd_summary_metrics(rmsds, is_edge)
    return print_rmsd_metric_summary(title, summary)


def rmsd_summary_with_suffix(rmsds, is_edge=None, suffix='_rmsd'):
    summary = rmsd_summary_metrics(rmsds, is_edge)
    return {
        f'{key}{suffix}': summary[key]
        for key, _ in RMSD_SUMMARY_FIELDS
    } | {
        'n_finite': summary['n_finite'],
        'n_total': summary['n_total'],
    }


def empty_prefixed_rmsd_stats(prefix):
    return {
        f'{prefix}_{key}_rmsd': None
        for key, _ in RMSD_SUMMARY_FIELDS
    }


def prefixed_rmsd_stats(summary, prefix):
    return {
        f'{prefix}_{key}_rmsd': summary[f'{key}_rmsd']
        for key, _ in RMSD_SUMMARY_FIELDS
    }


# %% Mean train-torsion baseline on test set
def mean_target_torsions_and_tau(dataset):
    first_data = cast(Data, dataset[0])
    n_torsions = int(first_data.torsions.shape[-1])
    sin_sums = np.zeros(n_torsions, dtype=np.float64)
    cos_sums = np.zeros(n_torsions, dtype=np.float64)
    torsion_counts = np.zeros(n_torsions, dtype=np.int64)
    tau_sum = 0.0
    tau_count = 0

    for i in tqdm(
        range(len(dataset)),
        desc='mean baseline: train torsions',
        colour=PROGRESS_BAR_COLOR,
    ):
        data = cast(Data, dataset[i])
        tidx = int(data.target_nt_idx.item())
        theta = data.torsions[tidx].detach().cpu().numpy()
        theta_mask = data.torsion_mask[tidx].detach().cpu().numpy().astype(bool)
        sin_sums[theta_mask] += np.sin(theta[theta_mask])
        cos_sums[theta_mask] += np.cos(theta[theta_mask])
        torsion_counts[theta_mask] += 1

        if bool(data.tau_m_mask[tidx].item()):
            tau_sum += float(data.tau_m[tidx].item())
            tau_count += 1

    mean_theta = np.zeros(n_torsions, dtype=np.float64)
    valid_torsions = torsion_counts > 0
    mean_theta[valid_torsions] = np.arctan2(
        sin_sums[valid_torsions],
        cos_sums[valid_torsions],
    )
    mean_tau = tau_sum / tau_count
    return mean_theta, mean_tau, torsion_counts, tau_count


def evaluate_mean_torsion_baseline(dataset, mean_theta, mean_tau):
    pred_theta = torch.as_tensor(
        mean_theta,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    pred_tau = torch.tensor([mean_tau], device=device, dtype=torch.float32)
    rmsds: list[float] = []
    is_edge: list[bool] = []

    with torch.no_grad():
        for i in tqdm(
            range(len(dataset)),
            desc='mean baseline: test decode',
            colour=PROGRESS_BAR_COLOR,
        ):
            data = cast(Data, dataset[i].clone())
            batch = cast(Any, Batch.from_data_list([data])).to(device)
            rmsd = model._compute_rmsd_per_graph_local(pred_theta, pred_tau, batch)

            rmsds.append(float(rmsd.item()))
            tidx = int(data.target_nt_idx.item())
            is_edge.append(bool(data.is_chain_edge_nt[tidx].item()))

    return np.asarray(rmsds, dtype=np.float64), np.asarray(is_edge, dtype=bool)


mean_baseline_theta, mean_baseline_tau, mean_baseline_torsion_counts, mean_baseline_tau_count = (
    mean_target_torsions_and_tau(train_dataset)
)
mean_baseline_test_rmsds, mean_baseline_test_is_edge = evaluate_mean_torsion_baseline(
    test_dataset,
    mean_baseline_theta,
    mean_baseline_tau,
)
print_rmsd_summary(
    'Mean train torsion/amplitude baseline test RMSD:',
    mean_baseline_test_rmsds,
    mean_baseline_test_is_edge,
)


# %% Show samples from dataset
rng = np.random.default_rng()
raw_idx = rng.integers(len(dataset))
raw_data = cast(Data, dataset[raw_idx])
pdb_id = Path(dataset.data_list[raw_idx]).parent.name

ws = raw_data.bb_xyz_world.shape[0]
tidx_raw = int(raw_data.target_nt_idx.item())
o_t_raw = raw_data.nt_origins_world[tidx_raw].numpy()
R_t_raw = raw_data.nt_frames_world[tidx_raw].numpy()
axis_len = 2.5
axis_tip_scale = 0.25


def to_target_local(points_world):
    # Express world-space points in the target nucleotide frame.
    return world_to_local_np(points_world, o_t_raw, R_t_raw)


def format_nt_label(nucleotide_idx, target_idx):
    if nucleotide_idx == target_idx:
        return f'Нуклеотид {nucleotide_idx} (целевой)'
    return f'Нуклеотид {nucleotide_idx} (контекстный)'


def add_local_axes(
    fig,
    origin_world,
    frame_world,
    target_origin,
    target_frame,
    axis_length,
    tip_scale,
    opacity,
    label_prefix,
):
    origin_local = world_to_local_np(origin_world, target_origin, target_frame).reshape(3)
    axis_specs = [('X', 'red', 0), ('Y', 'green', 1), ('Z', 'blue', 2)]
    for axis_name, color, axis_idx in axis_specs:
        end_world = origin_world + axis_length * frame_world[:, axis_idx]
        end_local = world_to_local_np(end_world, target_origin, target_frame).reshape(3)
        direction_local = end_local - origin_local
        fig.add_trace(go.Scatter3d(
            x=[origin_local[0], end_local[0]],
            y=[origin_local[1], end_local[1]],
            z=[origin_local[2], end_local[2]],
            mode='lines',
            line=dict(color=color, width=4),
            opacity=opacity,
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Cone(
            x=[end_local[0]],
            y=[end_local[1]],
            z=[end_local[2]],
            u=[direction_local[0]],
            v=[direction_local[1]],
            w=[direction_local[2]],
            showscale=False,
            colorscale=[[0, color], [1, color]],
            sizemode='absolute',
            sizeref=tip_scale,
            anchor='tail',
            opacity=opacity,
            name=f'{label_prefix} {axis_name}',
            showlegend=False,
            hoverinfo='skip',
        ))


def segments_to_xyz(segments):
    """Convert (p1, p2) segment pairs to flat xyz lists with None separators."""
    x, y, z = [], [], []
    for p1, p2 in segments:
        x.extend([p1[0], p2[0], None])
        y.extend([p1[1], p2[1], None])
        z.extend([p1[2], p2[2], None])
    return x, y, z


def add_xyz_gizmo(fig, axis_len, tip_scale, label_z=0.15, unit_cones=False):
    """Add X/Y/Z axis lines, cones, and labels to a 3D plotly figure."""
    for ax_i, color in enumerate(['red', 'green', 'blue']):
        end = [0.0, 0.0, 0.0]
        end[ax_i] = axis_len
        tip = [0.0, 0.0, 0.0]
        tip[ax_i] = 1.0 if unit_cones else axis_len
        fig.add_trace(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode='lines', line=dict(color=color, width=5), showlegend=False,
        ))
        fig.add_trace(go.Cone(
            x=[end[0]], y=[end[1]], z=[end[2]],
            u=[tip[0]], v=[tip[1]], w=[tip[2]],
            showscale=False, colorscale=[[0, color], [1, color]],
            sizemode='absolute', sizeref=tip_scale, anchor='tail',
            showlegend=False, hoverinfo='skip',
        ))
    fig.add_trace(go.Scatter3d(
        x=[axis_len, 0.3, 0, 0],
        y=[0.3, axis_len, 0.3, 0],
        z=[0, 0, axis_len, label_z],
        mode='text', text=['X', 'Y', 'Z', '(0, 0, 0)'],
        textposition=['middle right', 'middle right', 'top center', 'top center'],
        textfont=dict(size=14, color='black'), showlegend=False,
    ))


fig = go.Figure()

# Single pass: build backbone segments, neighbor and target point sets.
all_bb_segs = []
neighbor_pts = []
target_pts = np.empty((0, 3))
for i in range(ws):
    bb_world = raw_data.bb_xyz_world[i].numpy()
    valid = ~np.any(np.isnan(bb_world), axis=1)
    bb_local = to_target_local(bb_world[valid])
    all_bb_segs.extend(backbone_segments_from_local_coords(bb_local, valid))
    if i == tidx_raw:
        target_pts = bb_local
    elif len(bb_local) > 0:
        neighbor_pts.append(bb_local)

source_lines_x, source_lines_y, source_lines_z = segments_to_xyz(all_bb_segs)
if source_lines_x:
    fig.add_trace(go.Scatter3d(
        x=source_lines_x,
        y=source_lines_y,
        z=source_lines_z,
        mode='lines',
        line=dict(color='rgba(45, 45, 45, 0.45)', width=6),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=source_lines_x,
        y=source_lines_y,
        z=source_lines_z,
        mode='lines',
        line=dict(color='rgba(210, 210, 210, 0.95)', width=3),
        showlegend=False,
        hoverinfo='skip',
    ))

# Neighbor backbone atoms
if neighbor_pts:
    neighbor_pts = np.concatenate(neighbor_pts, axis=0)
    fig.add_trace(go.Scatter3d(
        x=neighbor_pts[:, 0],
        y=neighbor_pts[:, 1],
        z=neighbor_pts[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color='#ff9896',
            opacity=0.9,
            symbol='circle',
            line=dict(width=1.0, color='rgba(20, 20, 20, 0.65)'),
        ),
        name='Исходный контекстный остов',
    ))

# Target backbone atoms
fig.add_trace(go.Scatter3d(
    x=target_pts[:, 0],
    y=target_pts[:, 1],
    z=target_pts[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color='#d62728',
        opacity=0.96,
        symbol='circle',
        line=dict(width=1.2, color='rgba(20, 20, 20, 0.65)'),
    ),
    name='Исходный целевой остов',
))

# Nucleotide origins and local axes
origin_worlds = raw_data.nt_origins_world.numpy()
origin_locals = to_target_local(origin_worlds)
origin_labels = [
    format_nt_label(i, tidx_raw)
    for i in range(ws)
]
fig.add_trace(go.Scatter3d(
    x=origin_locals[:, 0],
    y=origin_locals[:, 1],
    z=origin_locals[:, 2],
    mode='text',
    text=origin_labels,
    textposition='top center',
    name='Центры локальных систем координат',
    showlegend=False,
))

for i in range(ws):
    frame_world = raw_data.nt_frames_world[i].numpy()
    add_local_axes(
        fig,
        origin_worlds[i],
        frame_world,
        o_t_raw,
        R_t_raw,
        axis_len,
        axis_tip_scale,
        opacity=1.0 if i == tidx_raw else 0.55,
        label_prefix=f'Нуклеотид {i}',
    )

add_xyz_gizmo(fig, axis_len, axis_tip_scale)

fig.update_layout(
    title=f'PDB ID: {pdb_id}',
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
    ),
    width=900, height=760,
    margin=dict(r=10, l=10, b=10, t=45),
    legend=dict(itemsizing='constant'),
)
fig.show()

# %% Training curves and logged metrics
plt.rcParams['font.family'] = 'Nunito'


metric_tags = {
    'train/loss':                          ('avg',     'train/loss'),
    'diagnostics/train/score_loss':        ('avg',     'diagnostics/train/score_loss'),
    'val/rmsd/avg':                        ('avg',     'val/rmsd'),
    'val/rmsd/central':                    ('central', 'val/rmsd'),
    'val/rmsd/edge':                       ('edge',    'val/rmsd'),
    'test/rmsd/avg':                       ('avg',     'test/rmsd'),
    'test/rmsd/avg_median':                ('avg',     'test/rmsd/avg_median'),
    'test/rmsd/avg_p90':                   ('avg',     'test/rmsd/avg_p90'),
    'test/rmsd/central':                   ('central', 'test/rmsd'),
    'test/rmsd/edge':                      ('edge',    'test/rmsd'),
}
scalars = collect_scalar_history(event_files, metric_tags)
wide_per_mode = {
    mode: scalars.loc[scalars['mode'] == mode].pivot_table(
        index='epoch', columns='metric', values='value', aggfunc='last'
    )
    for mode in target_modes
}
wide = wide_per_mode['avg']
train_score_loss_col = 'diagnostics/train/score_loss' if 'diagnostics/train/score_loss' in wide.columns else 'train/loss'
if train_score_loss_col in wide.columns:
    wide['train_noise_rmse'] = np.sqrt(wide[train_score_loss_col].clip(lower=0))


def plot_metric(ax, table, column, color, label, linestyle='-'):
    values = table[column].dropna()
    return ax.plot(
        values.index.to_numpy(),
        values.to_numpy(),
        color=color,
        linewidth=2,
        label=label,
        linestyle=linestyle,
    )[0]


validation_labels = {
    'avg': 'все нуклеотиды',
    'central': 'внутренние нуклеотиды',
    'edge': 'крайние нуклеотиды',
}


fig, ax = plt.subplots(figsize=(7, 4))
ax.tick_params(axis='both', labelsize=15)
if 'train_noise_rmse' in wide.columns:
    plot_metric(ax, wide, 'train_noise_rmse', 'indigo', 'train_rmse')
swa_epoch = 100
ax.axvline(swa_epoch, color='red', linewidth=3, linestyle='--')
ax.text(
    swa_epoch + 0.5,
    0.95,
    'Старт стохастического\nусреднения весов',
    transform=ax.get_xaxis_transform(),
    color='red',
    fontsize=14,
    va='top',
    ha='left',
)
ax.set_xlabel('Эпоха', fontsize=18)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'train.png'), bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(7, 4))
ax.tick_params(axis='both', labelsize=15)
for mode in target_modes:
    w = wide_per_mode[mode]
    if 'val/rmsd' in w.columns:
        plot_metric(
            ax,
            w,
            'val/rmsd',
            mode_colors[mode],
            validation_labels[mode],
            mode_linestyles[mode],
        )
ax.set_xlabel('Эпоха', fontsize=18)
ax.set_ylabel('RMSD остова (Å)', fontsize=18)
ax.legend(fontsize=14)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'val.png'), bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(6, 3))
ax.tick_params(axis='both', labelsize=15)
test_values = [
    float(wide_per_mode[mode]['test/rmsd'].dropna().iloc[-1])
    for mode in target_modes
]
bars = ax.barh(
    [validation_labels[mode] for mode in target_modes],
    test_values,
    color=[mode_colors[mode] for mode in target_modes],
)
ax.bar_label(bars, labels=[f'{value:.2f}' for value in test_values], fontsize=14, padding=4)
ax.set_xlabel('RMSD остова (Å)', fontsize=18)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'test.png'), bbox_inches='tight', dpi=300)
plt.show()

# %% Visualize one test prediction
test_pdb_to_local: dict[str, list[int]] = defaultdict(list)
test_paths = []
for local_i, (w_idx, _) in enumerate(test_dataset.virtual_entries):
    p = test_dataset.base.data_list[w_idx]
    test_paths.append(p)
    test_pdb_to_local[Path(p).parent.name].append(local_i)

sample_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
sample_idx = random.choice(test_pdb_to_local[sample_pdb_id])
sample_window = int(Path(test_paths[sample_idx]).stem)
data = cast(Data, test_dataset[sample_idx].clone())
tidx = int(data.target_nt_idx.item())
tidx_origin = data.nt_origins_world[tidx].numpy()
tidx_frame = data.nt_frames_world[tidx].numpy()
# Fall back to identity frame if pynamod returned NaN for this (edge) nucleotide
if np.isnan(tidx_origin).any() or np.isnan(tidx_frame).any():
    tidx_origin = np.zeros(3, dtype=np.float64)
    tidx_frame = np.eye(3, dtype=np.float64)


def collect_frame_geometry(origins_world, frames_world, target_origin, target_frame, target_idx):
    labels = []
    axis_entries = []
    local_points = []
    for i, (origin_world, frame_world) in enumerate(zip(origins_world, frames_world)):
        if np.isnan(origin_world).any():
            continue
        origin_local = world_to_local_np(origin_world, target_origin, target_frame).reshape(3)
        if np.isnan(origin_local).any():
            continue
        local_points.append(origin_local)
        labels.append(format_nt_label(i, target_idx))
        if not np.isnan(frame_world).any():
            axis_entries.append((i, origin_world, frame_world))
    return np.asarray(local_points, dtype=np.float64), labels, axis_entries


pos_full = []
target_mask = []
for i in range(data.bb_xyz_world.shape[0]):
    names_i, local_i = backbone_local_in_target_frame(data, i, tidx_origin, tidx_frame)
    pos_full.extend(local_i.tolist())
    target_mask.extend([i == tidx] * len(names_i))
pos_full = np.asarray(pos_full, dtype=np.float64)
target_mask = np.asarray(target_mask, dtype=bool)
side_mask = ~target_mask
true_bb = pos_full[target_mask]

with torch.no_grad():
    batch = cast(Any, Batch.from_data_list([data])).to(device)
    pred_theta, pred_tau_m = model.sample(batch)

restype = IDX_TO_BASE[int(data.base_types[tidx].argmax().item())]
o3_prev_vis = None
if bool(data.o3_prev_valid[tidx].item()):
    o3_prev_vis = data.o3_prev_local[tidx].numpy()
pred_local = build_backbone_local(
    pred_theta[0].cpu().numpy(),
    restype,
    o3_prev_local=o3_prev_vis,
    tau_m=float(pred_tau_m[0].clamp(min=1e-3).item()),
)

bb_tidx = data.bb_xyz_world[tidx].numpy()
valid_mask = ~np.any(np.isnan(bb_tidx), axis=1)
true_local = {
    BACKBONE_ATOMS[j]: true_bb[k]
    for k, j in enumerate(np.where(valid_mask)[0])
}
pred_bb = np.array(
    [pred_local[name] for name in BACKBONE_ATOMS if name in pred_local],
    dtype=np.float64,
)

matched_window = find_window_matching_sample(test_dataset.base, sample_pdb_id, data)
full_source_segments: list[tuple[np.ndarray, np.ndarray]] = []
target_base_pts: list[np.ndarray] = []
side_base_pts: list[np.ndarray] = []
if matched_window is not None:
    per_nt_coords = coords_local_per_nt(matched_window, tidx_origin, tidx_frame)
    bb_atom_names = set(BACKBONE_ATOMS)
    for i, (nucleotide, cmap) in enumerate(zip(matched_window, per_nt_coords)):
        full_source_segments.extend(bond_segments_from_nt_graph(cmap, nucleotide.restype))
        for aname, xyz in cmap.items():
            if aname not in bb_atom_names:
                (target_base_pts if i == tidx else side_base_pts).append(xyz)
    full_source_segments.extend(phosphodiester_segments_local(data, tidx_origin, tidx_frame))
else:
    print(
        'Results: не удалось сопоставить окно с mmCIF; '
        'рисуем только связи остова цели.'
    )

fig = go.Figure()

if full_source_segments:
    lsx, lsy, lsz = segments_to_xyz(full_source_segments)
    fig.add_trace(go.Scatter3d(
        x=lsx, y=lsy, z=lsz,
        mode='lines',
        line=dict(color='rgba(35, 35, 35, 0.55)', width=6),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=lsx, y=lsy, z=lsz,
        mode='lines',
        line=dict(color='rgba(210, 210, 210, 0.95)', width=3),
        showlegend=False,
        hoverinfo='skip',
    ))
elif matched_window is None:
    fb_only = ordered_backbone_segments(true_local)
    if fb_only:
        osx, osy, osz = segments_to_xyz(fb_only)
        fig.add_trace(go.Scatter3d(
            x=osx, y=osy, z=osz,
            mode='lines',
            line=dict(color='rgba(210, 210, 210, 0.95)', width=3),
            showlegend=False,
            hoverinfo='skip',
        ))

if np.any(side_mask):
    side_pts = pos_full[side_mask]
    fig.add_trace(go.Scatter3d(
        x=side_pts[:, 0],
        y=side_pts[:, 1],
        z=side_pts[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color='#ff9896',
            opacity=0.9,
            symbol='circle',
            line=dict(width=1.0, color='rgba(20, 20, 20, 0.65)'),
        ),
        name='Исходный контекстный остов',
    ))

fig.add_trace(go.Scatter3d(
    x=true_bb[:, 0],
    y=true_bb[:, 1],
    z=true_bb[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color='#d62728',
        opacity=0.96,
        symbol='circle',
        line=dict(width=1.2, color='rgba(20, 20, 20, 0.65)'),
    ),
    name='Исходный целевой остов',
))

if target_base_pts:
    tbp = np.stack(target_base_pts, axis=0)
    fig.add_trace(go.Scatter3d(
        x=tbp[:, 0],
        y=tbp[:, 1],
        z=tbp[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color='#2ca02c',
            opacity=0.94,
            symbol='circle',
            line=dict(width=1.0, color='rgba(20, 20, 20, 0.65)'),
        ),
        name='Исходное основание целевого нуклеотида, ',
    ))

fig.add_trace(go.Scatter3d(
    x=pred_bb[:, 0],
    y=pred_bb[:, 1],
    z=pred_bb[:, 2],
    mode='markers',
    marker=dict(
        size=9,
        color='#1f77b4',
        opacity=0.97,
        symbol='circle',
        line=dict(width=2.4, color='rgba(10, 10, 10, 0.9)'),
    ),
    name='Сгенерированный остов',
))

if side_base_pts:
    sbp = np.stack(side_base_pts, axis=0)
    fig.add_trace(go.Scatter3d(
        x=sbp[:, 0],
        y=sbp[:, 1],
        z=sbp[:, 2],
        mode='markers',
        marker=dict(
            size=9,
            color='#6acb3d',
            opacity=0.98,
            symbol='circle',
            line=dict(width=1.4, color='rgba(15, 55, 10, 0.95)'),
        ),
        name='Исходное основание контекстного нуклеотида',
    ))

pred_bb_segments = ordered_backbone_segments(pred_local)
if pred_bb_segments:
    lines_x, lines_y, lines_z = segments_to_xyz(pred_bb_segments)
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='rgba(8, 65, 140, 0.95)', width=6),
        showlegend=False,
        hoverinfo='skip',
    ))

shared_names = [name for name in BACKBONE_ATOMS if name in true_local and name in pred_local]
if shared_names:
    corr_x, corr_y, corr_z = segments_to_xyz(
        [(true_local[n], pred_local[n]) for n in shared_names]
    )
    fig.add_trace(go.Scatter3d(
        x=corr_x,
        y=corr_y,
        z=corr_z,
        mode='lines',
        line=dict(color='rgba(40, 170, 255, 0.98)', width=3, dash='dot'),
        name='Пары исходный-сгенерированный',
    ))

results_origin_worlds = data.nt_origins_world.numpy()
results_frame_worlds = data.nt_frames_world.numpy()
results_origin_locals, results_origin_labels, results_axis_entries = collect_frame_geometry(
    results_origin_worlds,
    results_frame_worlds,
    tidx_origin,
    tidx_frame,
    tidx,
)
if len(results_origin_locals) > 0:
    fig.add_trace(go.Scatter3d(
        x=results_origin_locals[:, 0],
        y=results_origin_locals[:, 1],
        z=results_origin_locals[:, 2],
        mode='text',
        text=results_origin_labels,
        textposition='top center',
        showlegend=False,
    ))
for i, origin_world, frame_world in results_axis_entries:
    add_local_axes(
        fig,
        origin_world,
        frame_world,
        tidx_origin,
        tidx_frame,
        2.5,
        axis_tip_scale,
        opacity=1.0 if i == tidx else 0.55,
        label_prefix=f'Нуклеотид {i}',
    )

all_points = np.vstack([pos_full, pred_bb])
radius = float(np.max(np.linalg.norm(all_points, axis=1)))
axis_len = max(4.0, radius * 1.15) / 5.0
add_xyz_gizmo(fig, axis_len, 0.45, label_z=0.25, unit_cones=True)

fig.update_layout(
    title=f'Эксперимент {run_dir}: PDB {sample_pdb_id}, окно {sample_window}',
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
    ),
    width=800,
    height=600,
    margin=dict(r=10, l=10, b=10, t=50),
    legend=dict(itemsizing='constant'),
)
fig.show()

# %% Run whole-structure inference
inference_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
raw_inference_path = osp.join('..', 'data', 'raw', f'{inference_pdb_id}.cif')

generated_pdb_path = osp.join(run_dir, f'generated_backbone_{inference_pdb_id}.pdb')
_window_sz = int(test_dataset.base.window_size)
_inference_sampler = CheckpointSamplerAdapter(model, None)
_, chain_records_inference = parse_dna(
    raw_inference_path,
    use_full_nucleotide=False,
    window_size=_window_sz,
)
_predictions_inference = predict_backbone_from_chain_records(
    [chain_records_inference],
    _inference_sampler,
    device
)[0]
with warnings.catch_warnings():
    warnings.simplefilter('ignore', PDBConstructionWarning)
    _, full_chain_records = parse_dna(
        raw_inference_path,
        use_full_nucleotide=True,
        window_size=_window_sz,
    )
generated_universe = build_output_universe(full_chain_records, _predictions_inference)

original_preds = {}
for _, chain, _ in full_chain_records:
    for nucleotide in chain:
        exp_positions = dict(default_atoms_provider(nucleotide))
        for atom_name in BACKBONE_ATOMS:
            xyz = exp_positions.get(atom_name)
            if xyz is not None:
                original_preds[(nucleotide.segid, int(nucleotide.resid), atom_name)] = xyz

write_structure(generated_universe, generated_pdb_path)
original_pdb_path = osp.join(run_dir, f'original_backbone_{inference_pdb_id}.pdb')
write_structure(
    build_output_universe(full_chain_records, original_preds),
    original_pdb_path,
)

with open(generated_pdb_path) as f:
    pdb_str_generated = f.read()

with open(original_pdb_path) as f:
    pdb_str_original = f.read()

view = py3Dmol.view(width=800, height=400, linked=False, viewergrid=(1, 2))
view.addModel(pdb_str_generated, 'pdb', viewer=(0, 0))
view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}}, viewer=(0, 0))
view.addLabel(
    f'Сгенерированный остов ({inference_pdb_id})',
    {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
    viewer=(0, 0),
)

view.addModel(pdb_str_original, 'pdb', viewer=(0, 1))
view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}}, viewer=(0, 1))
view.addLabel(
    f'Исходный остов ({inference_pdb_id})',
    {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
    viewer=(0, 1),
)

view.zoomTo()
view.show()

# %% Measure stochastic spread across independent diffusion runs on the same input
K = 10
N_SAMPLES = 60

skip_atoms = {'OP1', 'OP2', 'P'}
valid_bb_atoms = [a for a in BACKBONE_ATOMS if a not in skip_atoms]
idx_to_base = IDX_TO_BASE

pool_size = len(test_dataset)
inter_run_indices = rng.choice(
    pool_size, size=min(N_SAMPLES, pool_size), replace=False
).tolist()

inter_run_rmsds: list[float] = []

with torch.no_grad():
    for wi in tqdm(
            inter_run_indices,
            desc='Inter-run RMSD',
            colour=PROGRESS_BAR_COLOR,
    ):
        data = cast(Any, test_dataset[wi].clone())
        tidx = int(data.target_nt_idx.item())
        restype = idx_to_base[int(data.base_types[tidx].argmax().item())]
        o3_ir = None
        if bool(data.o3_prev_valid[tidx].item()):
            o3_ir = data.o3_prev_local[tidx].numpy()

        batch = cast(Any, Batch.from_data_list([data])).to(device)

        run_coords: list[dict] = []
        for _ in range(K):
            pred_theta, pred_tau_m = model.sample(batch)
            torsions_np = pred_theta[0].cpu().numpy()
            tau_m_val = float(pred_tau_m[0].clamp(min=1e-3).item())
            bb = build_backbone_local(
                torsions_np, restype, o3_prev_local=o3_ir, tau_m=tau_m_val,
            )
            run_coords.append({k: v for k, v in bb.items() if k not in skip_atoms})

        # intersection of atoms present in every run, ordered by backbone_atoms
        shared = set(run_coords[0].keys())
        for rc in run_coords[1:]:
            shared &= set(rc.keys())
        shared_atoms = [a for a in valid_bb_atoms if a in shared]

        if len(shared_atoms) < 3:
            continue

        coords = np.array(
            [[rc[a] for a in shared_atoms] for rc in run_coords]
        )  # [K, n_atoms, 3]

        # vectorized pairwise RMSD over all K*(K-1)/2 pairs
        diffs = coords[:, None] - coords[None]  # [K, K, n_atoms, 3]
        rmsds_mat = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1), axis=-1))  # [K, K]
        r1, r2 = np.triu_indices(K, k=1)
        pair_rmsds = rmsds_mat[r1, r2].tolist()

        inter_run_rmsds.extend(pair_rmsds)

inter_run_rmsd_mean = _rmsd_mean(inter_run_rmsds)
print(f'\nInter-run RMSD (mean): {format_rmsd_value(inter_run_rmsd_mean)}')

# %% Prepare kNN baseline

knn_base_dataset = train_dataset.base


@lru_cache
def deposit_group_id(pdb_id):
    return dm._read_deposit_group_id(knn_base_dataset, pdb_id)


def normalize_poly_seq(seq):
    return ''.join(str(seq).split()).upper()


@lru_cache
def dna_sequence_tokens(pdb_id):
    raw_path = osp.join(knn_base_dataset.raw_dir, f'{pdb_id}.cif')
    if not osp.exists(raw_path):
        return frozenset()
    mmcif = MMCIF2Dict(raw_path)
    poly_types = mmcif.get('_entity_poly.type', [])
    poly_seqs = mmcif.get('_entity_poly.pdbx_seq_one_letter_code_can', [])
    if isinstance(poly_types, str):
        poly_types = [poly_types]
    if isinstance(poly_seqs, str):
        poly_seqs = [poly_seqs]
    seqs = {
        normalize_poly_seq(seq)
        for poly_type, seq in zip(poly_types, poly_seqs)
        if 'deoxyribonucleotide' in str(poly_type).lower() and normalize_poly_seq(seq)
    }
    return frozenset(seqs)


def collect_pdb_ids(*datasets):
    ids = set()
    for dataset in datasets:
        for w_idx, _ in dataset.virtual_entries:
            path = dataset.base.data_list[w_idx]
            ids.add(Path(path).parent.name)
    return sorted(ids)


def build_pdb_meta_cache(pdb_ids):
    cache = {}
    for pdb_id in tqdm(
            pdb_ids,
            desc='kNN: reading PDB metadata',
            colour=PROGRESS_BAR_COLOR,
    ):
        cache[pdb_id] = {
            'deposit_group_id': deposit_group_id(pdb_id),
            'dna_sequence_tokens': dna_sequence_tokens(pdb_id),
        }
    return cache


def payload_key(dataset, target_type):
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
    """Return (total_dim, name→slice) in one pass over feature_blocks."""
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


def dataset_samples_cached(dataset, label, pdb_meta_cache):
    n_samples = len(dataset)
    n_bb = len(BACKBONE_ATOMS)
    metas: list[dict[str, Any]] = []

    first_w_idx, first_target_type = dataset.virtual_entries[0]
    first_base = dataset.base.get(first_w_idx)
    first_payloads = cast(
        dict[str, dict[str, torch.Tensor]],
        getattr(first_base, '_precomputed_target_payloads'),
    )
    first_payload = first_payloads[payload_key(dataset, first_target_type)]
    feat_dim, feat_slices = feature_dim_and_slices(first_base, first_payload)

    feats = np.empty((n_samples, feat_dim), dtype=np.float32)
    locals_ = np.empty((n_samples, n_bb, 3), dtype=np.float32)
    base_cache: dict[int, Any] = {first_w_idx: first_base}

    for i in tqdm(
            range(n_samples),
            desc=f'kNN: indexing {label}',
            colour=PROGRESS_BAR_COLOR,
    ):
        w_idx, target_type = dataset.virtual_entries[i]
        if w_idx not in base_cache:
            base_cache[w_idx] = dataset.base.get(w_idx)
        base = base_cache[w_idx]
        payloads = cast(
            dict[str, dict[str, torch.Tensor]],
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
            'deposit_group_id': pdb_meta['deposit_group_id'],
            'dna_sequence_tokens': pdb_meta['dna_sequence_tokens'],
            'base_type': int(base.base_types[ti].argmax().item()),
            'is_edge': bool(base.is_chain_edge_nt[ti].item()),
        })
    return feats, locals_, metas, feat_slices


def fit_knn_indices(ref_feats, ref_metas) -> tuple[
    StandardScaler,
    dict[int, np.ndarray],
    dict[int, NearestNeighbors],
]:
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
        nn = NearestNeighbors(n_neighbors=len(indices), algorithm='auto')
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


def run_knn_protocol(name, query_feats, query_locals, query_metas, ref_feats, ref_locals, ref_metas,
                     allow_same_sample):
    scaler, ref_indices_by_base, nn_by_base = fit_knn_indices(ref_feats, ref_metas)
    query_feats_scaled = np.asarray(scaler.transform(query_feats), dtype=np.float32)
    rmsds: list[float] = []
    is_edge: list[bool] = []
    missing = 0
    for i, meta in enumerate(tqdm(
            query_metas,
            desc=f'kNN: {name}',
            colour=PROGRESS_BAR_COLOR,
    )):
        ref_indices = ref_indices_by_base.get(meta['base_type'])
        if ref_indices is None or len(ref_indices) == 0:
            rmsds.append(np.nan)
            is_edge.append(meta['is_edge'])
            missing += 1
            continue
        _, neighbors = nn_by_base[meta['base_type']].kneighbors(
            query_feats_scaled[i:i + 1],
            n_neighbors=len(ref_indices),
        )
        match_idx = None
        for local_idx in neighbors[0]:
            ref_idx = int(ref_indices[int(local_idx)])
            if candidate_allowed(meta, ref_metas[ref_idx], allow_same_sample):
                match_idx = ref_idx
                break
        if match_idx is None:
            rmsds.append(np.nan)
            missing += 1
        else:
            rmsds.append(local_backbone_rmsd(ref_locals[match_idx], query_locals[i]))
        is_edge.append(meta['is_edge'])
    rmsds_arr = np.asarray(rmsds, dtype=np.float64)
    is_edge_arr = np.asarray(is_edge, dtype=bool)
    print_rmsd_summary(
        f'kNN backbone RMSD (local frame) [{name}]',
        rmsds_arr,
        is_edge_arr,
    )
    print(
        f'  {"eligible":>10}: '
        f'{np.isfinite(rmsds_arr).sum()}/{len(rmsds_arr)} '
        f'(missing={missing})'
    )
    return rmsds_arr, is_edge_arr


pdb_meta_cache = build_pdb_meta_cache(
    collect_pdb_ids(train_dataset, val_dataset, test_dataset)
)

train_feats, train_locals, train_metas, train_feat_slices = dataset_samples_cached(
    train_dataset,
    'train',
    pdb_meta_cache,
)
val_feats, val_locals, val_metas, val_feat_slices = dataset_samples_cached(
    val_dataset,
    'val',
    pdb_meta_cache,
)
test_feats, test_locals, test_metas, test_feat_slices = dataset_samples_cached(
    test_dataset,
    'test',
    pdb_meta_cache,
)

feature_sets = [
    ('base_type only', ['base_types']),
    ('rel_origins only', ['rel_origins']),
    ('rel_frames only', ['rel_frames']),
    ('rel_origins + rel_frames', ['rel_origins', 'rel_frames']),
    ('pair_rel_origins + pair_rel_frames', ['pair_rel_origins', 'pair_rel_frames']),
    ('rel_origins + rel_frames + base_type', ['rel_origins', 'rel_frames', 'base_types']),
    ('full static feature', list(train_feat_slices.keys())),
]


def select_feature_columns(feats, feat_slices, feature_names):
    return np.concatenate(
        [feats[:, feat_slices[name]] for name in feature_names],
        axis=1,
    )


# %% Run kNN feature-set comparison
for feature_set_name, feature_names in feature_sets:
    test_subset_feats = select_feature_columns(test_feats, test_feat_slices, feature_names)
    train_subset_feats = select_feature_columns(train_feats, train_feat_slices, feature_names)
    run_knn_protocol(
        f'test-to-train, {feature_set_name}, same base type, no same PDB/deposit/identical DNA seq',
        test_subset_feats,
        test_locals,
        test_metas,
        train_subset_feats,
        train_locals,
        train_metas,
        allow_same_sample=False,
    )
# %% Compare logged model RMSD and oracle decoder RMSD
wide_per_mode = globals().get('wide_per_mode')
if wide_per_mode is None:
    print('  unavailable: run the training-metrics cell first')
else:
    logged_model_test_summary: dict[str, float | int | None] = {
        key: None
        for key, _ in RMSD_SUMMARY_FIELDS
    }
    logged_model_test_n_finite = 0
    for label, metric_name, summary_key in [
        ('avg', 'test/rmsd/avg_median', 'avg_median'),
        ('avg', 'test/rmsd', 'avg_mean'),
        ('avg', 'test/rmsd/avg_p90', 'avg_p90'),
        ('central', 'test/rmsd', 'central_mean'),
        ('edge', 'test/rmsd', 'edge_mean'),
    ]:
        w = wide_per_mode.get(label)
        if w is None:
            continue
        if metric_name in w.columns:
            values = w[metric_name].dropna()
            if len(values):
                logged_model_test_summary[summary_key] = float(values.iloc[-1])
                logged_model_test_n_finite += 1
    logged_model_test_summary['n_finite'] = logged_model_test_n_finite
    logged_model_test_summary['n_total'] = len(RMSD_SUMMARY_FIELDS)
    print_rmsd_metric_summary('Logged model test RMSD:', logged_model_test_summary)


def accumulate_per_atom_errors(
    pred_local,
    gt_local,
    atom_sq_sums,
    atom_counts,
):
    bb_atom_idx = {name: i for i, name in enumerate(BACKBONE_ATOMS)}
    op_match = {'OP1': 'OP1', 'OP2': 'OP2'}
    j_op1 = bb_atom_idx['OP1']
    j_op2 = bb_atom_idx['OP2']
    pred_op1 = pred_local[j_op1]
    pred_op2 = pred_local[j_op2]
    gt_op1 = gt_local[j_op1]
    gt_op2 = gt_local[j_op2]
    if not (
        np.isnan(pred_op1).any()
        or np.isnan(pred_op2).any()
        or np.isnan(gt_op1).any()
        or np.isnan(gt_op2).any()
    ):
        direct_err = np.sum((pred_op1 - gt_op1) ** 2) + np.sum((pred_op2 - gt_op2) ** 2)
        swapped_err = np.sum((pred_op1 - gt_op2) ** 2) + np.sum((pred_op2 - gt_op1) ** 2)
        if swapped_err < direct_err:
            op_match = {'OP1': 'OP2', 'OP2': 'OP1'}

    for atom_name in BACKBONE_ATOMS:
        pred_idx = bb_atom_idx[atom_name]
        gt_idx = bb_atom_idx[op_match.get(atom_name, atom_name)]
        pred_xyz = pred_local[pred_idx]
        gt_xyz = gt_local[gt_idx]
        if np.isnan(pred_xyz).any() or np.isnan(gt_xyz).any():
            continue
        sq_err = float(np.sum((pred_xyz - gt_xyz) ** 2))
        atom_sq_sums[atom_name] += sq_err
        atom_counts[atom_name] += 1


def collect_per_atom_errors(atom_sq_sums, atom_counts):
    total_sq = float(sum(atom_sq_sums.values()))
    atom_rows = []
    for atom_name in BACKBONE_ATOMS:
        count = atom_counts[atom_name]
        if not count:
            continue
        mean_sq = atom_sq_sums[atom_name] / count
        atom_rows.append((
            atom_name,
            float(np.sqrt(mean_sq)),
            100.0 * atom_sq_sums[atom_name] / max(total_sq, 1e-12),
        ))
    return atom_rows


def plot_per_atom_errors(atom_rows):
    if not atom_rows:
        return

    atom_names = [row[0] for row in atom_rows]
    atom_rmsds = [row[1] for row in atom_rows]
    atom_contribs = [row[2] for row in atom_rows]

    fig_rmsd, ax_atom_rmsd = plt.subplots(figsize=(6, 4))
    ax_atom_rmsd.bar(atom_names, atom_rmsds, color='indigo')
    ax_atom_rmsd.set_ylabel('RMSD (Å)')
    ax_atom_rmsd.tick_params(axis='x', rotation=45)
    ax_atom_rmsd.spines['top'].set_visible(False)
    ax_atom_rmsd.spines['right'].set_visible(False)
    fig_rmsd.tight_layout()
    plt.savefig(osp.join(run_dir, 'per_atom_rmsd.png'), bbox_inches='tight', dpi=300)
    plt.show()

    fig_contrib, ax_atom_contrib = plt.subplots(figsize=(6, 4))
    ax_atom_contrib.bar(atom_names, atom_contribs, color='violet')
    ax_atom_contrib.set_ylabel('Вклад (%)')
    ax_atom_contrib.tick_params(axis='x', rotation=45)
    ax_atom_contrib.spines['top'].set_visible(False)
    ax_atom_contrib.spines['right'].set_visible(False)
    fig_contrib.tight_layout()
    plt.savefig(osp.join(run_dir, 'per_atom_contrib.png'), bbox_inches='tight', dpi=300)
    plt.show()


# %% Oracle decoder RMSD + decoder(pred torsions) vs decoder(true torsions)
RMSD_EVAL_BATCH_SIZE = 128

oracle_rmsds_list: list[float] = []
oracle_is_edge_list: list[bool] = []
oracle_atom_sq_sums: defaultdict[str, float] = defaultdict(float)
oracle_atom_counts: defaultdict[str, int] = defaultdict(int)
decoded_torsion_rmsds_list: list[float] = []
decoded_torsion_is_edge_list: list[bool] = []
sample_total_rmsds_list: list[float] = []
sample_total_is_edge_list: list[bool] = []

eval_loader = DataLoader(
    cast(Any, test_dataset),
    batch_size=RMSD_EVAL_BATCH_SIZE,
    shuffle=False,
)

with torch.no_grad():
    for batch in tqdm(
        eval_loader,
        total=len(eval_loader),
        desc='oracle + decoded torsion RMSD',
        colour=PROGRESS_BAR_COLOR,
    ):
        batch = cast(Any, batch).to(device)

        b, ws = model._b_ws(batch)
        ti = batch.target_nt_idx.long()
        bi = torch.arange(b, device=device)

        theta_w = batch.torsions.view(b, ws, N_TORSIONS).float()
        tau_w = (
            batch.tau_m
            .view(b, ws)
            .clamp(min=TAU_M_MIN, max=TAU_M_MAX)
            .float()
        )
        torsion_mask_w = batch.torsion_mask.view(b, ws, N_TORSIONS)

        coords_oracle_w = model._build_window_backbone(
            theta_w,
            tau_w,
            batch,
            torsion_mask_w,
        )

        pred_theta, pred_tau_m = model.sample(batch, num_timesteps=15)
        theta_pred_w = theta_w.clone()
        tau_pred_w = tau_w.clone()
        theta_pred_w[bi, ti] = pred_theta.float()
        tau_pred_w[bi, ti] = pred_tau_m.clamp(min=TAU_M_MIN, max=TAU_M_MAX).float()

        coords_pred_w = model._build_window_backbone(
            theta_pred_w,
            tau_pred_w,
            batch,
            torsion_mask_w,
        )

        n_bb = len(BACKBONE_ATOMS)
        gt_bb_world = batch.bb_xyz_world.view(b, ws, n_bb, 3)[bi, ti]
        oracle_bb_world = coords_oracle_w[bi, ti]
        pred_bb_world = coords_pred_w[bi, ti]
        origin = batch.nt_origins_world.view(b, ws, 3)[bi, ti]
        frame = batch.nt_frames_world.view(b, ws, 3, 3)[bi, ti]

        gt_local = ((gt_bb_world - origin[:, None]) @ frame).detach().cpu().numpy()
        oracle_local = ((oracle_bb_world - origin[:, None]) @ frame).detach().cpu().numpy()
        pred_local = ((pred_bb_world - origin[:, None]) @ frame).detach().cpu().numpy()
        is_edge_batch = model._is_edge_target(batch).detach().cpu().numpy().astype(bool)

        for sample_idx in range(b):
            oracle_rmsds_list.append(
                local_backbone_rmsd(oracle_local[sample_idx], gt_local[sample_idx])
            )
            oracle_is_edge_list.append(bool(is_edge_batch[sample_idx]))
            accumulate_per_atom_errors(
                oracle_local[sample_idx],
                gt_local[sample_idx],
                oracle_atom_sq_sums,
                oracle_atom_counts,
            )
            decoded_torsion_rmsds_list.append(
                local_backbone_rmsd(pred_local[sample_idx], oracle_local[sample_idx])
            )
            decoded_torsion_is_edge_list.append(bool(is_edge_batch[sample_idx]))
            sample_total_rmsds_list.append(
                local_backbone_rmsd(pred_local[sample_idx], gt_local[sample_idx])
            )
            sample_total_is_edge_list.append(bool(is_edge_batch[sample_idx]))

# %% Plot decoder RMSD summaries
oracle_rmsds = np.asarray(oracle_rmsds_list, dtype=np.float64)
oracle_is_edge = np.asarray(oracle_is_edge_list, dtype=bool)
print_rmsd_summary('Oracle decoder RMSD:', oracle_rmsds, oracle_is_edge)
oracle_per_atom_rows = collect_per_atom_errors(
    oracle_atom_sq_sums,
    oracle_atom_counts,
)
plot_per_atom_errors(oracle_per_atom_rows)

decoded_torsion_rmsds = np.asarray(decoded_torsion_rmsds_list, dtype=np.float64)
decoded_torsion_is_edge = np.asarray(decoded_torsion_is_edge_list, dtype=bool)
print_rmsd_summary(
    'Decoder(pred torsions) vs decoder(true torsions):',
    decoded_torsion_rmsds,
    decoded_torsion_is_edge,
)

sample_total_rmsds = np.asarray(sample_total_rmsds_list, dtype=np.float64)
sample_total_is_edge = np.asarray(sample_total_is_edge_list, dtype=bool)
print_rmsd_summary(
    'Decoder(pred torsions) vs true coords:',
    sample_total_rmsds,
    sample_total_is_edge,
)

# %% Compare PHENIX run time with Base2Backbone

NUM_TIMESTEPS_LIST = [5, 10, 15, 20, 30, 50, 100, 200, 300, 500, 1000]


def pick_phenix_minimized_structure(tmpdir: Path, local_input: Path) -> Path | None:
    """Find geometry_minimization coordinates under ``tmpdir`` (including subdirs)."""

    local_resolved = local_input.resolve()

    def candidates(paths):
        out = []
        for p in paths:
            if p.is_file() and p.resolve() != local_resolved:
                out.append(p)
        return out

    geo_patterns = (
        '**/*geo_minimized*.pdb',
        '**/*geo_minimized*.cif',
        '**/*geo_minimized*.mmcif',
    )
    minimized_patterns = (
        '**/*minimized*.pdb',
        '**/*minimized*.cif',
        '**/*minimized*.mmcif',
    )

    tier1 = []
    for pattern in geo_patterns:
        tier1.extend(tmpdir.glob(pattern))
    tier1 = candidates(tier1)
    if tier1:
        return max(tier1, key=lambda p: p.stat().st_mtime)

    tier2 = []
    for pattern in minimized_patterns:
        tier2.extend(tmpdir.glob(pattern))
    tier2 = candidates(tier2)
    if tier2:
        return max(tier2, key=lambda p: p.stat().st_mtime)

    fallback = []
    for path in tmpdir.rglob('*'):
        if path.suffix.lower() not in ('.pdb', '.cif', '.mmcif'):
            continue
        if path.is_file() and path.resolve() != local_resolved:
            fallback.append(path)
    return max(fallback, key=lambda p: p.stat().st_mtime) if fallback else None


def run_phenix_geometry_minimization(
    input_pdb: str | Path,
    output_dir: str | Path,
    selection: str = 'dna backbone',
    max_iterations: int = 500,
    macro_cycles: int = 5,
    timeout_s: int = 300,
) -> dict:
    """
    Run Phenix geometry minimization on a DNA structure.

    Returns:
        {
            "success": bool,
            "wall_time_s": float,
            "returncode": int,
            "output_pdb": Path | None,
            "stdout": str,
            "stderr": str,
        }
    """

    input_pdb = Path(input_pdb).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = input_pdb.stem

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
        tmpdir = Path(tmpdir)

        local_pdb = tmpdir / input_pdb.name
        shutil.copy2(input_pdb, local_pdb)

        out_prefix = tmpdir / prefix

        # geometry_minimization rejects ``output.prefix=`` on CLI for some builds; prefix is standardized here.
        phenix_args = [
            shlex.quote(str(local_pdb)),
            shlex.quote(f'selection={selection}'),
            shlex.quote(f'minimization.max_iterations={max_iterations}'),
            shlex.quote(f'minimization.macro_cycles={macro_cycles}'),
            shlex.quote(f'output_file_name_prefix={out_prefix}'),
        ]
        bash_script = (
            'source /etc/profile.d/modules.sh'
            ' && module load phenix'
            f" && phenix.geometry_minimization {' '.join(phenix_args)}"
        )

        try:
            result = subprocess.run(
                ['bash', '-lc', bash_script],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                cwd=str(tmpdir),
            )

            wall_time_s = time.perf_counter() - t0

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "wall_time_s": time.perf_counter() - t0,
                "returncode": -1,
                "output_pdb": None,
                "stdout": e.stdout or '',
                "stderr": f"Timeout after {timeout_s}s\n{e.stderr or ''}",
            }

        src = pick_phenix_minimized_structure(tmpdir, local_pdb)

        output_pdb = None
        if src is not None:
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            output_pdb = dst

        stderr_out = result.stderr or ''
        if src is None and result.returncode == 0:
            stdout_tail = (result.stdout or '').strip()
            if stdout_tail:
                stderr_out = (
                    f'{stderr_out}\n--- phenix stdout (tail) ---\n{stdout_tail[-6000:]}'
                ).strip()

        return {
            "success": result.returncode == 0 and output_pdb is not None,
            "wall_time_s": wall_time_s,
            "returncode": result.returncode,
            "output_pdb": output_pdb,
            "stdout": result.stdout,
            "stderr": stderr_out,
        }


def run_base2backbone_inference(
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_model,
    device: str | None = None,
    window_size: int | None = None,
    num_timesteps: int | None = None,
) -> dict:
    """
    Run end-to-end inference from a Lightning checkpoint and write a backbone PDB.

    Timing includes inference and output serialization to mirror PHENIX latency.
    """

    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if window_size is None:
        window_size_i = int(getattr(test_dataset.base, 'window_size', 3))
    else:
        window_size_i = int(window_size)

    output_pdb = output_dir / f'{input_path.stem}_base2backbone.pdb'
    t0 = time.perf_counter()
    input_path_str = str(input_path)

    try:
        adapter = CheckpointSamplerAdapter(checkpoint_model, num_timesteps)
        _, chain_records = parse_dna(
            input_path_str,
            use_full_nucleotide=False,
            window_size=window_size_i,
        )
        predictions = predict_backbone_from_chain_records(
            [chain_records],
            adapter,
            device
        )[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                input_path_str,
                use_full_nucleotide=True,
                window_size=window_size_i,
            )
        write_structure(build_output_universe(full_chain_records, predictions), output_pdb)
    except Exception as e:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(e),
        }

    return {
        'success': True,
        'wall_time_s': time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_pdb,
        'stdout': '',
        'stderr': '',
    }


def run_base2backbone_checkpoint_inference(
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_model,
    num_timesteps: int,
    device: str | None = None,
    window_size: int | None = None,
) -> dict:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if window_size is None:
        window_size_i = int(getattr(test_dataset.base, 'window_size', 3))
    else:
        window_size_i = int(window_size)

    output_pdb = output_dir / f'{input_path.stem}_steps_{num_timesteps}.pdb'
    t0 = time.perf_counter()
    input_path_str = str(input_path)

    try:
        _, chain_records = parse_dna(
            input_path_str,
            use_full_nucleotide=False,
            window_size=window_size_i,
        )
        adapter = CheckpointSamplerAdapter(checkpoint_model, num_timesteps)
        predictions = predict_backbone_from_chain_records(
            [chain_records],
            adapter,
            device
        )[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                input_path_str,
                use_full_nucleotide=True,
                window_size=window_size_i,
            )
        write_structure(build_output_universe(full_chain_records, predictions), output_pdb)
    except Exception as e:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(e),
        }

    return {
        'success': True,
        'wall_time_s': time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_pdb,
        'stdout': '',
        'stderr': '',
    }


def run_structure_benchmark(
    input_paths,
    output_dir,
    runner,
    label,
    max_workers=1,
    **runner_kwargs,
):
    output_dir = Path(output_dir)
    input_paths = [Path(path) for path in input_paths]

    rows = []
    batch_t0 = time.perf_counter()
    output_pdb_dir = output_dir / 'pdb'
    output_pdb_dir.mkdir(parents=True, exist_ok=True)

    def append_result(input_path, res):
        rows.append({
            'id': input_path.stem,
            'input_path': str(input_path),
            'success': res['success'],
            'wall_time_s': res['wall_time_s'],
            'returncode': res['returncode'],
            'output_pdb': str(res['output_pdb']) if res['output_pdb'] else '',
            'stderr': res['stderr'][:1000],
        })

    if max_workers == 1:
        for input_path in tqdm(
            input_paths,
            desc=f'{label} benchmark',
            leave=False,
            colour=PROGRESS_BAR_COLOR,
        ):
            try:
                res = runner(input_path, output_pdb_dir, **runner_kwargs)
            except Exception as e:
                res = {
                    'success': False,
                    'wall_time_s': None,
                    'returncode': -999,
                    'output_pdb': None,
                    'stderr': repr(e),
                }
            append_result(input_path, res)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    runner,
                    input_path,
                    output_pdb_dir,
                    **runner_kwargs,
                ): input_path
                for input_path in input_paths
            }

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f'{label} benchmark',
                leave=False,
                colour=PROGRESS_BAR_COLOR,
            ):
                input_path = futures[fut]

                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        'success': False,
                        'wall_time_s': None,
                        'returncode': -999,
                        'output_pdb': None,
                        'stderr': repr(e),
                    }

                append_result(input_path, res)

    print(
        f'{label}: processed {len(rows)} structures '
        f'in {time.perf_counter() - batch_t0:.2f}s'
    )
    return rows


def run_dataset(input_dir, output_dir, max_workers=4):
    input_dir = Path(input_dir)
    input_paths = sorted(
        list(input_dir.glob('*.pdb'))
        + list(input_dir.glob('*.cif'))
        + list(input_dir.glob('*.mmcif'))
    )
    return run_structure_benchmark(
        input_paths,
        output_dir,
        run_phenix_geometry_minimization,
        label='PHENIX',
        max_workers=max_workers,
    )


STANDARD_DNA_MONOMERS = frozenset({'DA', 'DC', 'DG', 'DT'})
ALLOWED_NONPOLY_COMPONENTS = frozenset({
    'ACT', 'CA', 'CL', 'DMS', 'DOD', 'EDO', 'FMT', 'GOL', 'HOH', 'IOD', 'K',
    'MG', 'MN', 'NA', 'NH4', 'NO3', 'PEG', 'PO4', 'SO4', 'TRS', 'ZN',
})


def mmcif_list(mmcif_dict, key):
    values = mmcif_dict.get(key, [])
    if isinstance(values, str):
        return [values]
    return list(values)


@lru_cache(maxsize=None)
def mmcif_nonstandard_reasons(path_str):
    mmcif_dict = MMCIF2Dict(path_str)
    reasons = []

    flags = mmcif_list(mmcif_dict, '_entity_poly.nstd_monomer')
    if any(str(flag).strip().lower() == 'yes' for flag in flags):
        reasons.append('nstd_monomer')

    entity_ids = mmcif_list(mmcif_dict, '_entity_poly.entity_id')
    poly_types = mmcif_list(mmcif_dict, '_entity_poly.type')
    poly_types_norm = [str(poly_type).strip().lower() for poly_type in poly_types]
    dna_entity_ids = {
        str(entity_id).strip()
        for entity_id, poly_type in zip(entity_ids, poly_types_norm)
        if poly_type == "polydeoxyribonucleotide"
    }
    if any(
        'ribonucleotide' in poly_type or 'hybrid' in poly_type
        for poly_type in poly_types_norm
        if poly_type != 'polydeoxyribonucleotide'
    ):
        reasons.append('rna_or_hybrid_polymer')

    atom_entity_ids = mmcif_list(mmcif_dict, '_atom_site.label_entity_id')
    atom_comp_ids = mmcif_list(mmcif_dict, '_atom_site.label_comp_id')
    if dna_entity_ids and atom_entity_ids and atom_comp_ids:
        dna_comp_ids = {
            str(comp_id).strip().upper()
            for entity_id, comp_id in zip(atom_entity_ids, atom_comp_ids)
            if (
                str(entity_id).strip() in dna_entity_ids
                and str(comp_id).strip() not in {'', '.', '?'}
            )
        }
        unexpected_dna_comp_ids = sorted(dna_comp_ids - STANDARD_DNA_MONOMERS)
        if unexpected_dna_comp_ids:
            reasons.append(
                f'dna_components={",".join(unexpected_dna_comp_ids[:5])}'
            )

    nonpoly_ids = {
        str(mon_id).strip().upper()
        for mon_id in mmcif_list(mmcif_dict, '_pdbx_nonpoly_scheme.mon_id')
        if str(mon_id).strip() not in {'', '.', '?'}
    }
    unexpected_nonpoly_ids = sorted(nonpoly_ids - ALLOWED_NONPOLY_COMPONENTS)
    if unexpected_nonpoly_ids:
        reasons.append(f'nonpoly={",".join(unexpected_nonpoly_ids[:5])}')

    return tuple(reasons)


def filter_standard_raw_paths(input_paths):
    standard_paths = []
    skipped_entries = []
    for path in input_paths:
        reasons = ()
        if path.suffix.lower() in ('.cif', '.mmcif'):
            reasons = mmcif_nonstandard_reasons(str(path))
        if reasons:
            skipped_entries.append(f'{path.stem} ({reasons[0]})')
            continue
        standard_paths.append(path)

    if skipped_entries:
        preview = ', '.join(skipped_entries[:10])
        extra = '' if len(skipped_entries) <= 10 else f' ... (+{len(skipped_entries) - 10})'
        print(f'Skipped {len(skipped_entries)} structures with nonstandard chemistry: {preview}{extra}')

    return standard_paths


def collect_test_dataset_raw_paths(dataset):
    raw_dir = Path(dataset.base.raw_dir)
    raw_paths_by_id = {}
    for base_idx, _ in dataset.virtual_entries:
        pdb_id = Path(dataset.base.data_list[base_idx]).parent.name
        raw_paths_by_id[pdb_id] = raw_dir / f'{pdb_id}.cif'

    missing_paths = [path for path in raw_paths_by_id.values() if not path.exists()]
    if missing_paths:
        missing_str = '\n'.join(str(path) for path in missing_paths[:10])
        raise FileNotFoundError(
            'Missing raw test structures:\n'
            f'{missing_str}'
        )

    return [raw_paths_by_id[pdb_id] for pdb_id in sorted(raw_paths_by_id)]


def residue_backbone_positions(chain_records):
    residue_positions = {}
    for chain_key, chain, _ in chain_records:
        for nucleotide in chain:
            residue_key = (chain_key, int(nucleotide.resid))
            residue_positions[residue_key] = {
                atom_name: np.asarray(atom_pos, dtype=np.float64)
                for atom_name, atom_pos in default_atoms_provider(nucleotide)
                if atom_name in BACKBONE_ATOMS
            }
    return residue_positions


def backbone_local_array(atom_positions, origin_world, frame_world):
    local_coords = np.full((len(BACKBONE_ATOMS), 3), np.nan, dtype=np.float64)
    for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
        atom_pos = atom_positions.get(atom_name)
        if atom_pos is None or not np.isfinite(atom_pos).all():
            continue
        local_coords[atom_idx] = world_to_local_np(atom_pos, origin_world, frame_world).reshape(3)
    return local_coords


@lru_cache(maxsize=None)
def parse_chain_records_for_backbone_rmsd(path, window_size=3):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'1 A\^3 CRYST1 record, this is usually a placeholder\..*',
            category=UserWarning,
            module=r'MDAnalysis\.coordinates\.PDB',
        )
        _, chain_records = parse_dna(
            str(path),
            use_full_nucleotide=True,
            window_size=window_size,
        )
    return chain_records


def collect_backbone_local_arrays(ref_chain_records, *position_maps):
    local_arrays = []
    seen_residues = set()
    for chain_key, _, windows in ref_chain_records:
        for window, _, data in windows:
            for nt_idx, nucleotide in enumerate(window):
                residue_key = (chain_key, int(nucleotide.resid))
                if residue_key in seen_residues:
                    continue
                seen_residues.add(residue_key)

                atoms_per_map = [positions.get(residue_key) for positions in position_maps]
                if any(atoms is None for atoms in atoms_per_map):
                    continue

                origin_world = data.nt_origins_world[nt_idx].numpy()
                frame_world = data.nt_frames_world[nt_idx].numpy()
                if np.isnan(origin_world).any() or np.isnan(frame_world).any():
                    continue

                local_arrays.append((
                    tuple(
                        backbone_local_array(atoms, origin_world, frame_world)
                        for atoms in atoms_per_map
                    ),
                    bool(data.is_chain_edge_nt[nt_idx].item()),
                ))
    return local_arrays


def compute_structure_vs_ref_backbone_rmsd(
    reference_path,
    output_path,
    window_size=3,
):
    ref_chain_records = parse_chain_records_for_backbone_rmsd(reference_path, window_size=window_size)
    output_chain_records = parse_chain_records_for_backbone_rmsd(output_path, window_size=window_size)

    ref_positions = residue_backbone_positions(ref_chain_records)
    output_positions = residue_backbone_positions(output_chain_records)

    local_arrays = collect_backbone_local_arrays(
        ref_chain_records,
        ref_positions,
        output_positions,
    )
    rmsds = []
    rmsd_is_edge = []
    for (ref_local, output_local), is_edge in local_arrays:
        rmsd = local_backbone_rmsd(output_local, ref_local)
        if np.isfinite(rmsd):
            rmsds.append(float(rmsd))
            rmsd_is_edge.append(is_edge)

    if not rmsds:
        empty_summary = rmsd_summary_with_suffix([])
        return {
            'success': False,
            'n_residues': 0,
            'mean_rmsd': None,
            'median_rmsd': None,
            'p90_rmsd': None,
            **empty_summary,
        }

    rmsd_arr = np.asarray(rmsds, dtype=np.float64)
    rmsd_is_edge_arr = np.asarray(rmsd_is_edge, dtype=bool)
    rmsd_summary = rmsd_summary_with_suffix(rmsd_arr, rmsd_is_edge_arr)
    return {
        'success': True,
        'n_residues': int(rmsd_summary['n_finite']),
        'mean_rmsd': rmsd_summary['avg_mean_rmsd'],
        'median_rmsd': rmsd_summary['avg_median_rmsd'],
        'p90_rmsd': rmsd_summary['avg_p90_rmsd'],
        **rmsd_summary,
    }


def summarize_structure_vs_ref_backbone_rmsd(label, rows, window_size=3):
    per_structure_summaries = []
    for row in rows:
        if not row['success'] or not row['output_pdb']:
            continue
        try:
            rmsd_stats = compute_structure_vs_ref_backbone_rmsd(
                row['input_path'],
                row['output_pdb'],
                window_size=window_size,
            )
        except Exception:
            continue
        if rmsd_stats['avg_median_rmsd'] is not None:
            per_structure_summaries.append(rmsd_stats)

    if not per_structure_summaries:
        print(f'No successful {label}/ref RMSD pairs were available.')
        return {
            'n_rmsd_compared': 0,
            'median_backbone_rmsd': None,
            **{
                f'{key}_backbone_rmsd': None
                for key, _ in RMSD_SUMMARY_FIELDS
            },
        }

    metric_means = {}
    for key, _ in RMSD_SUMMARY_FIELDS:
        vals = [
            stats[f'{key}_rmsd']
            for stats in per_structure_summaries
            if stats[f'{key}_rmsd'] is not None
        ]
        metric_means[f'{key}_backbone_rmsd'] = (
            float(np.mean(np.asarray(vals, dtype=np.float64)))
            if vals else None
        )

    summary = {
        'n_rmsd_compared': len(per_structure_summaries),
        'median_backbone_rmsd': float(np.median(np.asarray(
            [
                stats['avg_median_rmsd']
                for stats in per_structure_summaries
                if stats['avg_median_rmsd'] is not None
            ],
            dtype=np.float64,
        ))),
        **metric_means,
    }
    print(f'\n{label}/ref backbone RMSD:')
    for key, metric_label in RMSD_SUMMARY_FIELDS:
        print(
            f'  {metric_label:>12}: '
            f'{format_rmsd_value(summary[f"{key}_backbone_rmsd"])}'
        )
    print(f'  {"n":>12}: {summary["n_rmsd_compared"]}')
    return summary


def empty_runtime_rmsd_stats(prefix):
    return {
        'success': False,
        'n_residues': 0,
        f'{prefix}_mean_rmsd': None,
        f'{prefix}_median_rmsd': None,
        **empty_prefixed_rmsd_stats(prefix),
    }


def runtime_rmsd_stats_for_prefix(stats, prefix):
    return {
        'success': stats['success'],
        'n_residues': stats['n_residues'],
        f'{prefix}_mean_rmsd': stats['mean_rmsd'],
        f'{prefix}_median_rmsd': stats['median_rmsd'],
        **prefixed_rmsd_stats(stats, prefix),
    }


def summarize_runtime_comparison(model_rows, phenix_rows, window_size=3):
    model_by_id = {row['id']: row for row in model_rows}
    phenix_by_id = {row['id']: row for row in phenix_rows}

    comparison_rows = []
    for pdb_id in sorted(model_by_id.keys() & phenix_by_id.keys()):
        model_row = model_by_id[pdb_id]
        phenix_row = phenix_by_id[pdb_id]
        model_time = model_row['wall_time_s']
        phenix_time = phenix_row['wall_time_s']
        speedup = ''
        model_rmsd_stats = empty_runtime_rmsd_stats('model_vs_ref')
        phenix_rmsd_stats = empty_runtime_rmsd_stats('phenix_vs_ref')
        if (
            model_row['success']
            and phenix_row['success']
            and model_time not in (None, 0)
            and phenix_time is not None
        ):
            speedup = phenix_time / model_time
            try:
                model_stats = compute_structure_vs_ref_backbone_rmsd(
                    model_row['input_path'],
                    model_row['output_pdb'],
                    window_size=window_size,
                )
                model_rmsd_stats = runtime_rmsd_stats_for_prefix(
                    model_stats,
                    'model_vs_ref',
                )
                phenix_stats = compute_structure_vs_ref_backbone_rmsd(
                    phenix_row['input_path'],
                    phenix_row['output_pdb'],
                    window_size=window_size,
                )
                phenix_rmsd_stats = runtime_rmsd_stats_for_prefix(
                    phenix_stats,
                    'phenix_vs_ref',
                )
            except Exception as e:
                model_rmsd_stats = empty_runtime_rmsd_stats('model_vs_ref') | {'error': repr(e)}
                phenix_rmsd_stats = empty_runtime_rmsd_stats('phenix_vs_ref') | {'error': repr(e)}

        comparison_row = {
            'id': pdb_id,
            'model_success': model_row['success'],
            'model_wall_time_s': model_time,
            'phenix_success': phenix_row['success'],
            'phenix_wall_time_s': phenix_time,
            'phenix_over_model_speedup': speedup,
            'model_output_pdb': model_row['output_pdb'],
            'phenix_output_pdb': phenix_row['output_pdb'],
            'model_vs_ref_backbone_mean_rmsd': model_rmsd_stats['model_vs_ref_mean_rmsd'],
            'model_vs_ref_backbone_median_rmsd': model_rmsd_stats['model_vs_ref_median_rmsd'],
            'phenix_vs_ref_backbone_mean_rmsd': phenix_rmsd_stats['phenix_vs_ref_mean_rmsd'],
            'phenix_vs_ref_backbone_median_rmsd': phenix_rmsd_stats['phenix_vs_ref_median_rmsd'],
        }
        for key, _ in RMSD_SUMMARY_FIELDS:
            comparison_row[f'model_vs_ref_backbone_{key}_rmsd'] = (
                model_rmsd_stats[f'model_vs_ref_{key}_rmsd']
            )
            comparison_row[f'phenix_vs_ref_backbone_{key}_rmsd'] = (
                phenix_rmsd_stats[f'phenix_vs_ref_{key}_rmsd']
            )
        comparison_rows.append(comparison_row)

    valid_rows = [
        row
        for row in comparison_rows
        if row['phenix_over_model_speedup'] != ''
    ]
    if not valid_rows:
        summary = {
            'n_compared': 0,
            'model_median_wall_time_s': None,
            'phenix_median_wall_time_s': None,
            'median_phenix_over_model_speedup': None,
            'n_rmsd_compared': 0,
            'median_model_vs_ref_backbone_rmsd': None,
            'median_phenix_vs_ref_backbone_rmsd': None,
            **{
                f'model_vs_ref_backbone_{key}_rmsd': None
                for key, _ in RMSD_SUMMARY_FIELDS
            },
            **{
                f'phenix_vs_ref_backbone_{key}_rmsd': None
                for key, _ in RMSD_SUMMARY_FIELDS
            },
        }
        print('No successful model/PHENIX pairs were available.')
        return comparison_rows, summary

    model_times = np.array([row['model_wall_time_s'] for row in valid_rows], dtype=float)
    phenix_times = np.array([row['phenix_wall_time_s'] for row in valid_rows], dtype=float)
    speedups = np.array([row['phenix_over_model_speedup'] for row in valid_rows], dtype=float)
    rmsd_rows = [
        row for row in comparison_rows
        if row['model_vs_ref_backbone_median_rmsd'] is not None
    ]
    summary = {
        'n_compared': len(valid_rows),
        'model_median_wall_time_s': float(np.median(model_times)),
        'phenix_median_wall_time_s': float(np.median(phenix_times)),
        'median_phenix_over_model_speedup': float(np.median(speedups)),
        'n_rmsd_compared': len(rmsd_rows),
        'median_model_vs_ref_backbone_rmsd': (
            float(np.median([
                row['model_vs_ref_backbone_median_rmsd']
                for row in rmsd_rows
            ]))
            if rmsd_rows else None
        ),
        'median_phenix_vs_ref_backbone_rmsd': (
            float(np.median([
                row['phenix_vs_ref_backbone_median_rmsd']
                for row in rmsd_rows
            ]))
            if rmsd_rows else None
        ),
    }
    for key, _ in RMSD_SUMMARY_FIELDS:
        model_vals = [
            row[f'model_vs_ref_backbone_{key}_rmsd']
            for row in rmsd_rows
            if row[f'model_vs_ref_backbone_{key}_rmsd'] is not None
        ]
        phenix_vals = [
            row[f'phenix_vs_ref_backbone_{key}_rmsd']
            for row in rmsd_rows
            if row[f'phenix_vs_ref_backbone_{key}_rmsd'] is not None
        ]
        summary[f'model_vs_ref_backbone_{key}_rmsd'] = (
            float(np.mean(np.asarray(model_vals, dtype=np.float64)))
            if model_vals else None
        )
        summary[f'phenix_vs_ref_backbone_{key}_rmsd'] = (
            float(np.mean(np.asarray(phenix_vals, dtype=np.float64)))
            if phenix_vals else None
        )
    model_line = f'Median model time: {summary["model_median_wall_time_s"]:.3f}s'
    if summary['median_model_vs_ref_backbone_rmsd'] is not None:
        model_line += (
            f', median Base2Backbone RMSD: '
            f'{summary["median_model_vs_ref_backbone_rmsd"]:.3f} A'
        )
    print(model_line)

    print(f'Median PHENIX time: {summary["phenix_median_wall_time_s"]:.3f}s')

    print(f'Median speedup: {summary["median_phenix_over_model_speedup"]:.2f}x')
    if rmsd_rows:
        print_rmsd_metric_summary(
            'Base2Backbone/ref backbone RMSD:',
            {
                key: summary[f'model_vs_ref_backbone_{key}_rmsd']
                for key, _ in RMSD_SUMMARY_FIELDS
            } | {
                'n_finite': len(rmsd_rows),
                'n_total': len(rmsd_rows),
            },
        )
        print_rmsd_metric_summary(
            'PHENIX/ref backbone RMSD:',
            {
                key: summary[f'phenix_vs_ref_backbone_{key}_rmsd']
                for key, _ in RMSD_SUMMARY_FIELDS
            } | {
                'n_finite': len(rmsd_rows),
                'n_total': len(rmsd_rows),
            },
        )
    return comparison_rows, summary


def pick_runtime_subset(input_paths, subset_size=10, seed=42, require_standard_monomers=True):
    rng = random.Random(int(seed))
    shuffled_paths = list(input_paths)
    rng.shuffle(shuffled_paths)

    subset_paths = []
    skipped_entries = []
    for path in shuffled_paths:
        if require_standard_monomers:
            reasons = ()
            if path.suffix.lower() in ('.cif', '.mmcif'):
                reasons = mmcif_nonstandard_reasons(str(path))
            if reasons:
                skipped_entries.append(f'{path.stem} ({reasons[0]})')
                continue
        subset_paths.append(path)
        if len(subset_paths) >= int(subset_size):
            break

    return sorted(subset_paths, key=lambda path: path.stem)


def print_benchmark_summary(label, rows):
    n_total = len(rows)
    success_rows = [row for row in rows if row['success'] and row['wall_time_s'] is not None]
    print(f'{label}: {len(success_rows)}/{n_total} successful')
    if success_rows:
        times = np.array([row['wall_time_s'] for row in success_rows], dtype=float)
        print(
            f'{label}: median={np.median(times):.3f}s '
            f'mean={np.mean(times):.3f}s '
            f'min={np.min(times):.3f}s '
            f'max={np.max(times):.3f}s'
        )


def benchmark_timesteps_vs_phenix_on_subset(
    output_dir,
    subset_size=10,
    subset_seed=42,
    phenix_max_workers=4,
    num_timesteps_list=None,
    require_standard_monomers=True,
):
    if num_timesteps_list is None:
        num_timesteps_list = [5, 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    input_paths = collect_test_dataset_raw_paths(test_dataset)
    subset_paths = pick_runtime_subset(
        input_paths,
        subset_size=subset_size,
        seed=subset_seed,
        require_standard_monomers=require_standard_monomers,
    )
    output_dir = Path(output_dir)

    print('Subset IDs:', [path.stem for path in subset_paths])
    phenix_rows = run_structure_benchmark(
        subset_paths,
        output_dir / 'phenix',
        run_phenix_geometry_minimization,
        label='PHENIX',
        max_workers=phenix_max_workers,
    )
    print_benchmark_summary('PHENIX', phenix_rows)
    summarize_structure_vs_ref_backbone_rmsd(
        'PHENIX',
        phenix_rows,
        window_size=test_dataset.base.window_size,
    )

    model_rows_by_timesteps = {}
    comparison_by_timesteps = {}
    summary_by_timesteps = {}
    for num_timesteps in num_timesteps_list:
        label = f'base2backbone ({num_timesteps} steps)'
        model_rows = run_structure_benchmark(
            subset_paths,
            output_dir / f'base2backbone_steps_{num_timesteps}',
            run_base2backbone_checkpoint_inference,
            label=label,
            max_workers=1,
            checkpoint_model=model,
            num_timesteps=num_timesteps,
            device=device,
            window_size=test_dataset.base.window_size,
        )
        print_benchmark_summary(label, model_rows)
        comparison_rows, summary = summarize_runtime_comparison(
            model_rows,
            phenix_rows,
            window_size=test_dataset.base.window_size,
        )
        model_rows_by_timesteps[num_timesteps] = model_rows
        comparison_by_timesteps[num_timesteps] = comparison_rows
        summary_by_timesteps[num_timesteps] = summary

    return {
        'subset_seed': subset_seed,
        'subset_paths': subset_paths,
        'phenix_rows': phenix_rows,
        'model_rows_by_timesteps': model_rows_by_timesteps,
        'comparison_by_timesteps': comparison_by_timesteps,
        'summary_by_timesteps': summary_by_timesteps,
    }


def compare_model_vs_phenix_on_test_dataset(
    output_dir,
    phenix_max_workers=4,
    checkpoint_model=None,
    model_device=None,
    num_timesteps=None,
    require_standard_monomers=True,
):
    input_paths = collect_test_dataset_raw_paths(test_dataset)
    if require_standard_monomers:
        input_paths = filter_standard_raw_paths(input_paths)
    output_dir = Path(output_dir)

    model_rows = run_structure_benchmark(
        input_paths,
        output_dir / 'base2backbone',
        run_base2backbone_inference,
        label='base2backbone',
        max_workers=1,
        checkpoint_model=model if checkpoint_model is None else checkpoint_model,
        device=device if model_device is None else model_device,
        window_size=test_dataset.base.window_size,
        num_timesteps=num_timesteps,
    )
    phenix_rows = run_structure_benchmark(
        input_paths,
        output_dir / 'phenix',
        run_phenix_geometry_minimization,
        label='PHENIX',
        max_workers=phenix_max_workers,
    )
    summarize_structure_vs_ref_backbone_rmsd(
        'PHENIX',
        phenix_rows,
        window_size=test_dataset.base.window_size,
    )
    comparison_rows, summary = summarize_runtime_comparison(
        model_rows,
        phenix_rows,
        window_size=test_dataset.base.window_size,
    )
    return {
        'input_paths': input_paths,
        'model_rows': model_rows,
        'phenix_rows': phenix_rows,
        'comparison_rows': comparison_rows,
        'summary': summary,
    }


runtime_subset_seed = 42
runtime_subset_benchmark = benchmark_timesteps_vs_phenix_on_subset(
    Path(run_dir) / 'runtime_benchmark_subset',
    subset_size=10,
    subset_seed=runtime_subset_seed,
    phenix_max_workers=4,
    num_timesteps_list=NUM_TIMESTEPS_LIST,
)
