# %%
# Imports
import os.path as osp
import random
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Any, cast

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import py3Dmol
import requests
import seaborn as sns
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from bbregen import utils
from bbregen.dataset import DNADataModule, PyGDataset
from bbregen.model import PytorchLightningModule
from bbregen.predict import MODEL_DIR, predict_backbone, write_structure
from bbregen.torsion_constants import N_LATENT
from bbregen.torsion_geometry import build_backbone_from_torsions
from pynamod.atomic_analysis.nucleotides_parser import nucleotide_graphs

# Angle channel order in tensors: α, β, γ, ε, ζ, χ, P (no backbone δ; τ_m is separate / log τ in latent).
TOR_NAMES = ['α', 'β', 'γ', 'ε', 'ζ', 'χ', 'P']

# %%
# Load model and dataset
log_dir = osp.join('..', 'logs')
run_filename = osp.join('torsions', 'baseline')
run_dir = osp.join(log_dir, run_filename)
ckpt_path = utils.find_best_checkpoint(run_dir)
test_dataset_path = osp.join(run_dir, 'test_dataset.pt')
event_files = glob(osp.join(run_dir, 'events.*'))

try:
    test_dataset = torch.load(test_dataset_path, weights_only=False)
except FileNotFoundError:
    raise FileNotFoundError(f'`{test_dataset_path}` not found. Ensure training completed.')

target_modes = ('all', 'central', 'edge')
test_indices_per_mode = {
    'all': list(range(len(test_dataset))),
    'central': list(test_dataset.central_virtual),
    'edge': list(test_dataset.edge_virtual),
}
test_datasets = {
    mode: torch.utils.data.Subset(test_dataset, indices)
    for mode, indices in test_indices_per_mode.items()
}
mode_colors = {
    'all': 'indigo',
    'central': utils.PBAR_COLOR,
    'edge': 'violet',
}
mode_linestyles = {'all': '-', 'central': '--', 'edge': '--'}

try:
    model = PytorchLightningModule.load_from_checkpoint(
        ckpt_path, weights_only=False, map_location='cpu'
    ).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'device: {device}')
except FileNotFoundError:
    raise FileNotFoundError(f'Checkpoint `{ckpt_path}` not found.')

# %%
# Load dataset
dataset = PyGDataset()
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


def _to_target_local(points_world):
    # Express world-space points in the target nucleotide frame.
    points_world = np.asarray(points_world, dtype=np.float64)
    return (points_world - o_t_raw) @ R_t_raw


def _to_local_frame(points_world, target_origin, target_frame):
    points_world = np.asarray(points_world, dtype=np.float64)
    return (points_world - target_origin) @ target_frame


def _format_nt_label(nucleotide_idx, target_idx):
    if nucleotide_idx == target_idx:
        return f'Нуклеотид {nucleotide_idx} (целевой)'
    return f'Нуклеотид {nucleotide_idx} (контекстный)'


def _add_local_axes(
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
    origin_local = _to_local_frame(origin_world, target_origin, target_frame).reshape(3)
    axis_specs = [('X', 'red', 0), ('Y', 'green', 1), ('Z', 'blue', 2)]
    for axis_name, color, axis_idx in axis_specs:
        end_world = origin_world + axis_length * frame_world[:, axis_idx]
        end_local = _to_local_frame(end_world, target_origin, target_frame).reshape(3)
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


def _load_dataset_backbone_segments(bb_local, valid_mask):
    rename_bb = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}
    bb_set = set(utils.backbone_atoms)
    graph = nucleotide_graphs['A']
    bonds = []
    seen = set()
    for src_idx, dst_idx in graph.edges():
        src_name = rename_bb.get(
            graph.nodes[src_idx]['atom'].name,
            graph.nodes[src_idx]['atom'].name.rstrip('AB'),
        )
        dst_name = rename_bb.get(
            graph.nodes[dst_idx]['atom'].name,
            graph.nodes[dst_idx]['atom'].name.rstrip('AB'),
        )
        if src_name in bb_set and dst_name in bb_set and src_name != dst_name:
            key = tuple(sorted([src_name, dst_name]))
            if key not in seen:
                seen.add(key)
                bonds.append((src_name, dst_name))

    atom_names = [
        utils.backbone_atoms[j]
        for j in range(len(utils.backbone_atoms))
        if valid_mask[j]
    ]
    coords_by_name = {
        atom_name: xyz
        for atom_name, xyz in zip(atom_names, bb_local)
    }
    return [
        (coords_by_name[name_a], coords_by_name[name_b])
        for name_a, name_b in bonds
        if name_a in coords_by_name and name_b in coords_by_name
    ]


bb_local_per_nt = []
for i in range(ws):
    bb_world = raw_data.bb_xyz_world[i].numpy()
    valid = ~np.any(np.isnan(bb_world), axis=1)
    bb_local = _to_target_local(bb_world[valid])
    bb_local_per_nt.append((i, bb_local, valid))

fig = go.Figure()

source_lines_x, source_lines_y, source_lines_z = [], [], []
for _i, pts, valid in bb_local_per_nt:
    for p1, p2 in _load_dataset_backbone_segments(pts, valid):
        source_lines_x.extend([p1[0], p2[0], None])
        source_lines_y.extend([p1[1], p2[1], None])
        source_lines_z.extend([p1[2], p2[2], None])
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
neighbor_pts = [pts for i, pts, _valid in bb_local_per_nt if i != tidx_raw and len(pts) > 0]
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
target_pts = bb_local_per_nt[tidx_raw][1]
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
origin_locals = _to_target_local(origin_worlds)
origin_labels = [
    _format_nt_label(i, tidx_raw)
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
    _add_local_axes(
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

fig.add_trace(go.Scatter3d(
    x=[0, axis_len], y=[0, 0], z=[0, 0],
    mode='lines',
    line=dict(color='red', width=5),
    showlegend=False,
))
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, axis_len], z=[0, 0],
    mode='lines',
    line=dict(color='green', width=5),
    showlegend=False,
))
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, axis_len],
    mode='lines',
    line=dict(color='blue', width=5),
    showlegend=False,
))

fig.add_trace(go.Cone(
    x=[axis_len], y=[0], z=[0],
    u=[axis_len], v=[0], w=[0],
    showscale=False, colorscale=[[0, 'red'], [1, 'red']],
    sizemode='absolute', sizeref=axis_tip_scale, anchor='tail',
    showlegend=False,
    hoverinfo='skip',
))
fig.add_trace(go.Cone(
    x=[0], y=[axis_len], z=[0],
    u=[0], v=[axis_len], w=[0],
    showscale=False, colorscale=[[0, 'green'], [1, 'green']],
    sizemode='absolute', sizeref=axis_tip_scale, anchor='tail',
    showlegend=False,
    hoverinfo='skip',
))
fig.add_trace(go.Cone(
    x=[0], y=[0], z=[axis_len],
    u=[0], v=[0], w=[axis_len],
    showscale=False, colorscale=[[0, 'blue'], [1, 'blue']],
    sizemode='absolute', sizeref=axis_tip_scale, anchor='tail',
    showlegend=False,
    hoverinfo='skip',
))

fig.add_trace(go.Scatter3d(
    x=[axis_len, 0.3, 0, 0],
    y=[0.3, axis_len, 0.3, 0],
    z=[0, 0, axis_len, 0.15],
    mode='text',
    text=['X', 'Y', 'Z', '(0, 0, 0)'],
    textposition=['middle right', 'middle right', 'top center', 'top center'],
    textfont=dict(size=14, color='black'),
    showlegend=False,
))

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

# %%
# Model
plt.rcParams['font.family'] = 'DejaVu Sans'
hp = model.hparams

MAX_X = 15
MAX_Y = MAX_X * 1.5
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_xlim(0, MAX_X)
ax.set_ylim(0, MAX_Y)
ax.axis('off')

palette = {
    'input': '#E8F4F8',
    'pre': '#FDF5E6',
    'embed': '#FFE8C1',
    'graph': '#EAE3F5',
    'output': '#D8F0DF',
    'detail': '#FFF8D6',
}

hidden_dim = int(hp['hidden_dim'])
num_layers = int(hp['num_layers'])
time_emb_dim = 32
node_dim = int(model.node_dim)
torsion_latent_dim = N_LATENT


def draw_box(x, y, width, height, title, desc, color):
    ax.add_patch(FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle='round,pad=0.15',
        facecolor=color,
        linewidth=2.2,
    ))
    ax.text(
        x + width / 2,
        y + height,
        title,
        ha='center',
        va='top',
        fontsize=13,
        fontweight='bold',
    )
    ax.text(
        x + width * 0.03,
        y + height - 0.4,
        desc,
        ha='left',
        va='top',
        linespacing=1.6,
    )


def draw_arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle='-|>',
        linewidth=2,
        mutation_scale=25,
        color='black',
    ))


main_x = MAX_X * 0.18
main_y = MAX_Y * 0.86
main_width = MAX_X * 0.28
main_height = MAX_Y * 0.11
arrow_length = MAX_Y * 0.05
detail_x = MAX_X * 0.66
detail_width = MAX_X * 0.25
step_stride = main_height + arrow_length

input_y = main_y
prep_y = main_y - step_stride
embed_y = main_y - 2 * step_stride
transformer_y = main_y - 3 * step_stride
output_y = main_y - 4 * step_stride
decode_y = main_y - 5 * step_stride

draw_box(
    main_x,
    input_y,
    main_width,
    main_height,
    r'Входное окно ($x_t$)',
    '• rel_origins, rel_frames\n'
    '• base_type, has_pair, chain_end_class, is_target\n'
    f'• зашумлённый латент [θ (7), log τ_m] ({N_LATENT})\n'
    f'• embedding log σ ({time_emb_dim}), маска торсионов (без self-conditioning)',
    palette['input'],
)
draw_arrow(main_x + main_width / 2, input_y - MAX_Y * 0.01, main_x + main_width / 2, input_y - MAX_Y * 0.04)

draw_box(
    main_x,
    prep_y,
    main_width,
    main_height,
    'Подготовка входа',
    f'• вектор признаков нуклеотида ({node_dim})\n'
    '• целевой нуклеотид получает шумный латент и маску\n'
    '• остальные нуклеотиды дают только контекст',
    palette['pre'],
)
draw_arrow(main_x + main_width / 2, prep_y - MAX_Y * 0.01, main_x + main_width / 2, prep_y - MAX_Y * 0.04)

draw_box(
    detail_x,
    prep_y,
    detail_width,
    main_height,
    'Собираемые признаки',
    '• rel_origin (3) + rel_frame (9)\n'
    '• тип основания (4) + has pair (1)\n'
    '• chain-end class (3) + is_target (1)\n'
    f'• торсионный латент ({torsion_latent_dim}) + log σ ({time_emb_dim})',
    palette['detail'],
)
draw_arrow(detail_x, prep_y + main_height * 0.55, main_x + main_width * 1.04, prep_y + main_height * 0.55)

draw_box(
    main_x,
    embed_y,
    main_width,
    main_height,
    'Входной MLP',
    f'Linear({node_dim} → {hidden_dim})\n'
    'SiLU\n'
    f'Linear({hidden_dim} → {hidden_dim})',
    palette['embed'],
)
draw_arrow(main_x + main_width / 2, embed_y - MAX_Y * 0.01, main_x + main_width / 2, embed_y - MAX_Y * 0.04)

draw_box(
    main_x,
    transformer_y,
    main_width,
    main_height,
    'Трансформерный энкодер',
    f'• {num_layers} слоев\n'
    f'• скрытое пространство {hidden_dim}\n'
    f'• {int(hp["num_heads"])} голов внимания\n'
    '• обмен контекстом внутри окна',
    palette['graph'],
)
draw_arrow(main_x + main_width / 2, transformer_y - MAX_Y * 0.01, main_x + main_width / 2, transformer_y - MAX_Y * 0.04)

draw_box(
    main_x,
    output_y,
    main_width,
    main_height,
    'Голова предсказания',
    f'Linear({hidden_dim} → {N_LATENT})\n'
    '• predicted score (∇ log noisy density) для 7 углов и log τ_m',
    palette['output'],
)
draw_arrow(main_x + main_width / 2, output_y - MAX_Y * 0.01, main_x + main_width / 2, output_y - MAX_Y * 0.04)

draw_box(
    main_x,
    decode_y,
    main_width,
    main_height,
    'Обратная диффузия',
    f'• {int(hp["num_timesteps"])} шагов reverse VE score\n'
    f'• θ: σ∈[{hp.get("angular_sigma_min", "?")}, {hp.get("angular_sigma_max", "?")}], '
    f'log τ: σ∈[{hp.get("tau_sigma_min", "?")}, {hp.get("tau_sigma_max", "?")}]\n'
    '• ancestral sampler по углам с wrap + канал log τ_m',
    palette['detail'],
)

loop_x = 1.3
input_top_y = input_y + main_height
decode_bottom_y = decode_y
loop_path = mpath.Path(
    [
        (main_x + main_width / 2, decode_bottom_y),
        (main_x + main_width / 2, decode_bottom_y - 0.6),
        (loop_x, decode_bottom_y - 0.6),
        (loop_x, input_top_y + 0.6),
        (main_x + main_width / 2, input_top_y + 0.6),
        (main_x + main_width / 2, input_top_y),
    ],
    [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
    ],
)
ax.add_patch(FancyArrowPatch(
    path=loop_path,
    arrowstyle='-|>',
    linewidth=2,
    mutation_scale=25,
    color='black',
    linestyle='--',
))
ax.text(
    loop_x - 0.2,
    MAX_Y / 2,
    r'$x_t \rightarrow x_{t-\Delta}$' + '\n' + r'$(T \text{ шагов reverse VE score})$',
    fontsize=15,
    rotation=90,
    ha='center',
    va='center',
)

fig.tight_layout()
fig.savefig(osp.join(run_dir, 'model_arch.png'), dpi=200, bbox_inches='tight')
plt.show()

# %%
# Check optimizer state
_ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('epoch:', _ck.get('epoch'), '| global_step:', _ck.get('global_step'))
print('hyper_parameters lr:', (_ck.get('hyper_parameters') or {}).get('lr'))
for i, _opt in enumerate(_ck.get('optimizer_states', [])):
    print(f'\noptimizer_states[{i}]')
    for gi, g in enumerate(_opt['param_groups']):
        _pg = {k: v for k, v in g.items() if k != 'params'}
        _pg['n_param_ids'] = len(g['params'])
        print(f'  param_group[{gi}]:', _pg)
    _st = _opt.get('state', {})
    print(f'  state: {len(_st)} tensors with buffers')
    if _st:
        _pid0 = next(iter(_st))
        print('  example buffers for param', _pid0, ':', {
            k: tuple(v.shape) if hasattr(v, 'shape') else v
            for k, v in _st[_pid0].items()
        })
for i, sch in enumerate(_ck.get('lr_schedulers') or []):
    print(f'\nlr_schedulers[{i}]:', {
        k: v for k, v in sch.items()
        if k in ('_last_lr', 'last_epoch', 'best', 'num_bad_epochs', 'cooldown_counter', 'factor', 'patience')
    })

# %%
# Training
from tensorboard.backend.event_processing import \
    event_accumulator  # noqa: E402

plt.rcParams['font.family'] = 'Nunito'


def load_event_accumulator(path):
    ea = event_accumulator.EventAccumulator(
        path, size_guidance={'scalars': 0, 'histograms': 0, 'images': 0}
    )
    ea.Reload()
    return ea


def scalars_to_dataframe(ea, tag):
    rows = [(s.step, s.value) for s in ea.Scalars(tag)]
    return pd.DataFrame(rows, columns=['epoch', 'value'])


metric_tags = {
    'train_loss':                          ('all',     'train_loss'),
    'val_rmsd':                            ('all',     'val_rmsd'),
    'val_rmsd_central':                    ('central', 'val_rmsd'),
    'val_rmsd_edge':                       ('edge',    'val_rmsd'),
    'test_rmsd':                           ('all',     'test_rmsd'),
    'test_rmsd_central':                   ('central', 'test_rmsd'),
    'test_rmsd_edge':                      ('edge',    'test_rmsd'),
}
tracked_tags = set(metric_tags)

dfs = []
for ef in event_files:
    ea = load_event_accumulator(ef)
    for tag in ea.Tags()['scalars']:  # type: ignore
        if tag not in tracked_tags:
            continue
        df = scalars_to_dataframe(ea, tag)
        mode, metric = metric_tags[tag]
        df['mode'] = mode
        df['metric'] = metric
        dfs.append(df)

if not dfs:
    raise ValueError(f'No tracked TensorBoard scalars in `{run_dir}`.')

scalars = pd.concat(dfs, ignore_index=True)[['epoch', 'mode', 'metric', 'value']].reset_index(drop=True)
wide_per_mode = {
    mode: scalars.loc[scalars['mode'] == mode].pivot_table(
        index='epoch', columns='metric', values='value', aggfunc='last'
    )
    for mode in target_modes
}
wide = wide_per_mode['all']
if 'train_loss' in wide.columns:
    wide['train_noise_rmse'] = np.sqrt(wide['train_loss'].clip(lower=0))


def _plot_metric(ax, table, column, color, label, linestyle='-'):
    values = table[column].dropna()
    return ax.plot(
        values.index.to_numpy(),
        values.to_numpy(),
        color=color,
        linewidth=3,
        label=label,
        linestyle=linestyle,
    )[0]


validation_labels = {
    'all': 'все нуклеотиды',
    'central': 'центральные нуклеотиды',
    'edge': 'краевые нуклеотиды',
}


fig, ax = plt.subplots(figsize=(7, 4))
ax.tick_params(axis='both', labelsize=15)
if 'train_noise_rmse' in wide.columns:
    _plot_metric(ax, wide, 'train_noise_rmse', 'indigo', 'train_rmse')
if getattr(model, 'hparams', None) and 'swa_epoch_start' in model.hparams:
    swa_epoch = int(model.hparams['swa_epoch_start'])
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
ax.set_ylabel('RMSE шума за 1 шаг (Å)', fontsize=18)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'train.png'), bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(7, 4))
ax.tick_params(axis='both', labelsize=15)
for mode in target_modes:
    w = wide_per_mode[mode]
    if 'val_rmsd' in w.columns:
        _plot_metric(
            ax,
            w,
            'val_rmsd',
            mode_colors[mode],
            validation_labels[mode],
            mode_linestyles[mode],
        )
ax.set_yscale('log')
ax.set_xlabel('Эпоха', fontsize=18)
ax.set_ylabel('Медианный RMSD остова по окнам (Å)', fontsize=18)
ax.legend(fontsize=14)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'val.png'), bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(6, 3))
ax.tick_params(axis='both', labelsize=15)
test_values = [
    float(wide_per_mode[mode]['test_rmsd'].dropna().iloc[-1])
    for mode in target_modes
]
bars = ax.barh(
    [validation_labels[mode] for mode in target_modes],
    test_values,
    color=[mode_colors[mode] for mode in target_modes],
)
ax.bar_label(bars, labels=[f'{value:.2f}' for value in test_values], fontsize=14, padding=4)
ax.set_ylabel('RMSD (Å)', fontsize=18)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'test.png'), bbox_inches='tight', dpi=300)
plt.show()

# %%
# Results
test_pdb_to_local: dict[str, list[int]] = defaultdict(list)
_base_paths = test_dataset.base.data_list
test_paths = [_base_paths[w_idx] for w_idx, _ in test_dataset.virtual_entries]
for local_i, p in enumerate(test_paths):
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


def _backbone_local_in_target_frame(sample_data, nucleotide_idx, origin_world, frame_world):
    bb_world = sample_data.bb_xyz_world[nucleotide_idx].numpy()
    valid = ~np.any(np.isnan(bb_world), axis=1)
    if not valid.any():
        return [], np.empty((0, 3), dtype=np.float64)
    local = (bb_world[valid] - origin_world) @ frame_world
    # Drop rows where the frame transformation itself produced NaN
    row_valid = ~np.any(np.isnan(local), axis=1)
    names = [
        utils.backbone_atoms[j]
        for k, j in enumerate(range(len(utils.backbone_atoms)))
        if valid[j] and row_valid[k]
    ]
    return names, local[row_valid]


_RENAMES_BB = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}
_bb_set = set(utils.backbone_atoms)


def _get_backbone_bonds():
    G = nucleotide_graphs['A']
    bonds = []
    seen = set()
    for i, j in G.edges():
        na = _RENAMES_BB.get(G.nodes[i]['atom'].name, G.nodes[i]['atom'].name.rstrip('AB'))
        nb = _RENAMES_BB.get(G.nodes[j]['atom'].name, G.nodes[j]['atom'].name.rstrip('AB'))
        if na in _bb_set and nb in _bb_set and na != nb:
            key = tuple(sorted((str(na), str(nb))))
            if key not in seen:
                seen.add(key)
                bonds.append((na, nb))
    return bonds


_BACKBONE_BONDS = _get_backbone_bonds()


def _find_window_matching_sample(pyg_ds: PyGDataset, pdb_id: str, data: Data):
    """Reload structure and locate the sliding window whose backbone tensor matches `data`."""
    raw_path = osp.join(pyg_ds.raw_dir, f'{pdb_id}.cif')
    if not osp.exists(raw_path):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        _, chain_records = utils.parse_dna(
            raw_path,
            use_full_nucleotide=True,
            window_size=pyg_ds.window_size,
        )
    ref_bb = data.bb_xyz_world.detach().cpu()
    for _chain_key, _chain, windows in chain_records:
        for _window, _widx, wdata in windows:
            cand = wdata.bb_xyz_world.detach().cpu()
            if not torch.equal(torch.isnan(cand), torch.isnan(ref_bb)):
                continue
            if torch.allclose(
                torch.nan_to_num(cand),
                torch.nan_to_num(ref_bb),
                rtol=5e-4,
                atol=5e-4,
            ):
                return _window
    return None


def _world_to_target_local(xyz_world, origin_world, frame_world):
    return (
        (np.asarray(xyz_world, dtype=np.float64) - origin_world) @ frame_world
    ).reshape(3)


def _coords_local_per_nt(window, origin_world, frame_world):
    """Per-residue atom name → coordinates in the target nucleotide frame."""
    return [
        {
            name: _world_to_target_local(pos, origin_world, frame_world)
            for name, pos in utils.default_atoms_provider(nucleotide)
        }
        for nucleotide in window
    ]


def _bond_segments_from_nt_graph(coords_by_name: dict, restype_letter: str):
    """Intra-nucleotide bonds using the pynamod template graph (incl. base atoms)."""
    if restype_letter not in nucleotide_graphs:
        return []
    graph = nucleotide_graphs[restype_letter]
    segments = []
    seen = set()
    for i, j in graph.edges():
        na = utils.rename_atom(graph.nodes[i]['atom'].name)
        nb = utils.rename_atom(graph.nodes[j]['atom'].name)
        if na in coords_by_name and nb in coords_by_name and na != nb:
            key = tuple(sorted((str(na), str(nb))))
            if key in seen:
                continue
            seen.add(key)
            segments.append((coords_by_name[na], coords_by_name[nb]))
    return segments


def _phosphodiester_segments_local(data: Data, origin_world, frame_world):
    """Inter-residue O3'(i) — P(i+1) in target frame when backbone atoms exist."""
    ws_loc = data.bb_xyz_world.shape[0]
    bb = utils.backbone_atoms
    jo3 = bb.index("O3'")
    jp = bb.index('P')
    segments = []
    for i in range(ws_loc - 1):
        o3 = data.bb_xyz_world[i, jo3].numpy()
        p_next = data.bb_xyz_world[i + 1, jp].numpy()
        if np.isnan(o3).any() or np.isnan(p_next).any():
            continue
        segments.append((
            _world_to_target_local(o3, origin_world, frame_world),
            _world_to_target_local(p_next, origin_world, frame_world),
        ))
    return segments


def _collect_frame_geometry(origins_world, frames_world, target_origin, target_frame, target_idx):
    labels = []
    axis_entries = []
    local_points = []
    for i, (origin_world, frame_world) in enumerate(zip(origins_world, frames_world)):
        if np.isnan(origin_world).any():
            continue
        origin_local = _to_local_frame(origin_world, target_origin, target_frame).reshape(3)
        if np.isnan(origin_local).any():
            continue
        local_points.append(origin_local)
        labels.append(_format_nt_label(i, target_idx))
        if not np.isnan(frame_world).any():
            axis_entries.append((i, origin_world, frame_world))
    return np.asarray(local_points, dtype=np.float64), labels, axis_entries


def _ordered_segments(coords_by_name):
    segments = []
    for name_a, name_b in _BACKBONE_BONDS:
        if name_a in coords_by_name and name_b in coords_by_name:
            segments.append((coords_by_name[name_a], coords_by_name[name_b]))
    return segments


pos_full = []
target_mask = []
for i in range(data.bb_xyz_world.shape[0]):
    names_i, local_i = _backbone_local_in_target_frame(data, i, tidx_origin, tidx_frame)
    pos_full.extend(local_i.tolist())
    target_mask.extend([i == tidx] * len(names_i))
pos_full = np.asarray(pos_full, dtype=np.float64)
target_mask = np.asarray(target_mask, dtype=bool)
side_mask = ~target_mask
true_backbone = pos_full[target_mask]

with torch.no_grad():
    batch = cast(Any, Batch.from_data_list([data])).to(device)
    pred_theta, pred_tau_m = model.sample(batch)

restype = {v: k for k, v in utils.base_to_idx.items()}[
    int(data.base_types[tidx].argmax().item())
]
_o3_prev_vis = None
if bool(data.o3_prev_valid[tidx].item()):
    _o3_prev_vis = data.o3_prev_local[tidx].numpy()
pred_local_dict = build_backbone_from_torsions(
    pred_theta[0].cpu().numpy(),
    restype,
    o3_prev_local=_o3_prev_vis,
    tau_m=float(pred_tau_m[0].clamp(min=1e-3).item()),
)

true_local_dict = {
    name: xyz
    for name, xyz in zip(
        [utils.backbone_atoms[j] for j in range(len(utils.backbone_atoms)) if not np.isnan(data.bb_xyz_world[tidx].numpy()[j]).any()],
        true_backbone,
    )
}
pred_backbone = np.array(
    [pred_local_dict[name] for name in utils.backbone_atoms if name in pred_local_dict],
    dtype=np.float64,
)

_matched_window = _find_window_matching_sample(test_dataset.base, sample_pdb_id, data)
_full_source_segments: list[tuple[np.ndarray, np.ndarray]] = []
_target_base_pts: list[np.ndarray] = []
_side_base_pts: list[np.ndarray] = []
if _matched_window is not None:
    _per_nt_coords = _coords_local_per_nt(_matched_window, tidx_origin, tidx_frame)
    for nucleotide, cmap in zip(_matched_window, _per_nt_coords):
        _full_source_segments.extend(
            _bond_segments_from_nt_graph(cmap, nucleotide.restype)
        )
    _full_source_segments.extend(
        _phosphodiester_segments_local(data, tidx_origin, tidx_frame)
    )
    _bb_atom_names = set(utils.backbone_atoms)
    for i, cmap in enumerate(_per_nt_coords):
        for aname, xyz in cmap.items():
            if aname in _bb_atom_names:
                continue
            if i == tidx:
                _target_base_pts.append(xyz)
            else:
                _side_base_pts.append(xyz)
else:
    print(
        'Results: не удалось сопоставить окно с mmCIF; '
        'рисуем только связи остова цели.'
    )

fig = go.Figure()

if _full_source_segments:
    _lsx, _lsy, _lsz = [], [], []
    for _p1, _p2 in _full_source_segments:
        _lsx.extend([_p1[0], _p2[0], None])
        _lsy.extend([_p1[1], _p2[1], None])
        _lsz.extend([_p1[2], _p2[2], None])
    fig.add_trace(go.Scatter3d(
        x=_lsx,
        y=_lsy,
        z=_lsz,
        mode='lines',
        line=dict(color='rgba(35, 35, 35, 0.55)', width=6),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=_lsx,
        y=_lsy,
        z=_lsz,
        mode='lines',
        line=dict(color='rgba(210, 210, 210, 0.95)', width=3),
        showlegend=False,
        hoverinfo='skip',
    ))
elif _matched_window is None:
    _fb_only = _ordered_segments(true_local_dict)
    if _fb_only:
        _osx, _osy, _osz = [], [], []
        for _p1, _p2 in _fb_only:
            _osx.extend([_p1[0], _p2[0], None])
            _osy.extend([_p1[1], _p2[1], None])
            _osz.extend([_p1[2], _p2[2], None])
        fig.add_trace(go.Scatter3d(
            x=_osx,
            y=_osy,
            z=_osz,
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
    x=true_backbone[:, 0],
    y=true_backbone[:, 1],
    z=true_backbone[:, 2],
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

if _target_base_pts:
    _tbp = np.stack(_target_base_pts, axis=0)
    fig.add_trace(go.Scatter3d(
        x=_tbp[:, 0],
        y=_tbp[:, 1],
        z=_tbp[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color='#2ca02c',
            opacity=0.94,
            symbol='circle',
            line=dict(width=1.0, color='rgba(20, 20, 20, 0.65)'),
        ),
        name='Исходное: целевой нуклеотид, основание',
    ))

fig.add_trace(go.Scatter3d(
    x=pred_backbone[:, 0],
    y=pred_backbone[:, 1],
    z=pred_backbone[:, 2],
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

if _side_base_pts:
    _sbp = np.stack(_side_base_pts, axis=0)
    fig.add_trace(go.Scatter3d(
        x=_sbp[:, 0],
        y=_sbp[:, 1],
        z=_sbp[:, 2],
        mode='markers',
        marker=dict(
            size=9,
            color='#6acb3d',
            opacity=0.98,
            symbol='circle',
            line=dict(width=1.4, color='rgba(15, 55, 10, 0.95)'),
        ),
        name='Исходное: окружение, основание',
    ))

for segments, color, width, name in [
    (_ordered_segments(pred_local_dict), 'rgba(8, 65, 140, 0.95)', 6, 'Связи сгенерированного остова'),
]:
    if not segments:
        continue
    lines_x, lines_y, lines_z = [], [], []
    for p1, p2 in segments:
        lines_x.extend([p1[0], p2[0], None])
        lines_y.extend([p1[1], p2[1], None])
        lines_z.extend([p1[2], p2[2], None])
    fig.add_trace(go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        mode='lines',
        line=dict(color=color, width=width),
        showlegend=False,
        hoverinfo='skip',
    ))

shared_names = [name for name in utils.backbone_atoms if name in true_local_dict and name in pred_local_dict]
if shared_names:
    corr_x, corr_y, corr_z = [], [], []
    for name in shared_names:
        p_true = true_local_dict[name]
        p_pred = pred_local_dict[name]
        corr_x.extend([p_true[0], p_pred[0], None])
        corr_y.extend([p_true[1], p_pred[1], None])
        corr_z.extend([p_true[2], p_pred[2], None])
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
results_origin_locals, results_origin_labels, results_axis_entries = _collect_frame_geometry(
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
    _add_local_axes(
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

all_points = np.vstack([pos_full, pred_backbone])
radius = float(np.max(np.linalg.norm(all_points, axis=1)))
axis_len = max(4.0, radius * 1.15) / 5.0
for axis_vals, color in [
    ([0, axis_len, 0, 0, 0, 0], 'red'),
    ([0, 0, 0, axis_len, 0, 0], 'green'),
    ([0, 0, 0, 0, 0, axis_len], 'blue'),
]:
    fig.add_trace(go.Scatter3d(
        x=[axis_vals[0], axis_vals[1]],
        y=[axis_vals[2], axis_vals[3]],
        z=[axis_vals[4], axis_vals[5]],
        mode='lines',
        line=dict(color=color, width=5),
        showlegend=False,
    ))

fig.add_trace(go.Cone(
    x=[axis_len], y=[0], z=[0], u=[1], v=[0], w=[0],
    showscale=False, colorscale=[[0, 'red'], [1, 'red']],
    sizemode='absolute', sizeref=0.45, anchor='tail', showlegend=False,
))
fig.add_trace(go.Cone(
    x=[0], y=[axis_len], z=[0], u=[0], v=[1], w=[0],
    showscale=False, colorscale=[[0, 'green'], [1, 'green']],
    sizemode='absolute', sizeref=0.45, anchor='tail', showlegend=False,
))
fig.add_trace(go.Cone(
    x=[0], y=[0], z=[axis_len], u=[0], v=[0], w=[1],
    showscale=False, colorscale=[[0, 'blue'], [1, 'blue']],
    sizemode='absolute', sizeref=0.45, anchor='tail', showlegend=False,
))
fig.add_trace(go.Scatter3d(
    x=[axis_len, 0.3, 0, 0],
    y=[0.3, axis_len, 0.3, 0],
    z=[0, 0, axis_len, 0.25],
    mode='text',
    text=['X', 'Y', 'Z', '(0, 0, 0)'],
    textposition=['middle right', 'middle right', 'top center', 'top center'],
    textfont=dict(size=14, color='black'),
    showlegend=False,
))

fig.update_layout(
    title=f'Эксперимент {run_filename}: PDB {sample_pdb_id}, окно {sample_window}',
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

# %%
# Inference
inference_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
raw_inference_path = osp.join('..', 'data', 'raw', f'{inference_pdb_id}.cif')
print(f'Generating backbone for {inference_pdb_id}...')

generated_pdb_path = osp.join(run_dir, f'generated_backbone_{inference_pdb_id}.pdb')
predictions, inference_chain_records = predict_backbone(
    raw_inference_path,
    MODEL_DIR,
    device=device,
    show_progress=True,
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', PDBConstructionWarning)
    _, full_chain_records = utils.parse_dna(
        raw_inference_path,
        use_full_nucleotide=True,
        window_size=test_dataset.base.window_size,
    )

original_predictions = {}
for _, chain, _ in full_chain_records:
    for nucleotide in chain:
        exp_positions = dict(utils.default_atoms_provider(nucleotide))
        for atom_name in utils.backbone_atoms:
            xyz = exp_positions.get(atom_name)
            if xyz is not None:
                original_predictions[(nucleotide.segid, int(nucleotide.resid), atom_name)] = xyz

write_structure(full_chain_records, predictions, generated_pdb_path)
original_pdb_path = osp.join(run_dir, f'original_backbone_{inference_pdb_id}.pdb')
write_structure(full_chain_records, original_predictions, original_pdb_path)
print(f'Wrote {generated_pdb_path}  ({len(predictions)} backbone atoms)')
print(f'Wrote {original_pdb_path}  ({len(original_predictions)} backbone atoms)')

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

# %%
# Estimate dataset noise


@lru_cache
def _dna_seqs(pdb_id):
    q = ('{ entry(entry_id: "%s") { polymer_entities { entity_poly {'
         'rcsb_entity_polymer_type pdbx_seq_one_letter_code_can } } } }') % pdb_id
    r = requests.post('https://data.rcsb.org/graphql', json={'query': q}, timeout=10)
    return [
        e['entity_poly']['pdbx_seq_one_letter_code_can']
        for e in r.json()['data']['entry']['polymer_entities']
        if e['entity_poly']['rcsb_entity_polymer_type'] == 'DNA'
    ]


@lru_cache
def _similar_entries(seq):
    r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json={
        'query': {'type': 'terminal', 'service': 'sequence', 'parameters': {
            'evalue_cutoff': 1, 'identity_cutoff': 1.0,
            'sequence_type': 'dna', 'value': seq}},
        'return_type': 'entry',
    }, timeout=15)
    return [x['identifier'] for x in r.json().get('result_set', [])]


@lru_cache
def _parsed_chain_records(pdb_id):
    path = osp.join('..', 'data', 'raw', f'{pdb_id}.cif')
    if not osp.exists(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(requests.get(
            f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=60
        ).content)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        _, chain_records = utils.parse_dna(
            path, use_full_nucleotide=True, window_size=test_dataset.base.window_size,
        )
    return chain_records


def _central_window_index(chain_records):
    ci = test_dataset.base.window_size // 2
    idx = {}
    for _, _, windows in chain_records:
        for window, _, data in windows:
            central_nt = window[ci]
            idx[(central_nt.segid, int(central_nt.resid))] = data
    return idx


def _central_backbone_local(data):
    """Backbone atom positions in the local frame of the central nucleotide."""
    ws = data.bb_xyz_world.shape[0]
    ci = ws // 2
    bb_world = data.bb_xyz_world[ci].numpy()    # [n_bb, 3]
    o_t = data.nt_origins_world[ci].numpy()
    R_t = data.nt_frames_world[ci].numpy()
    valid = ~np.any(np.isnan(bb_world), axis=1)
    local = (bb_world - o_t) @ R_t             # transform to local frame
    return {
        utils.backbone_atoms[j]: local[j]
        for j in range(len(utils.backbone_atoms))
        if valid[j]
    }


def _experimental_rmsd_windows(pdb_id1, pdb_id2):
    idx1 = _central_window_index(_parsed_chain_records(pdb_id1))
    idx2 = _central_window_index(_parsed_chain_records(pdb_id2))
    values = []
    for key in set(idx1) & set(idx2):
        atoms1 = _central_backbone_local(idx1[key])
        atoms2 = _central_backbone_local(idx2[key])
        shared = [n for n in utils.backbone_atoms if n in atoms1 and n in atoms2]
        if not shared:
            continue
        pos1 = np.array([atoms1[n] for n in shared])
        pos2 = np.array([atoms2[n] for n in shared])
        values.append(float(np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))))
    return values


def _noise_pairs_for_pdb(pdb_id):
    values = []
    try:
        partners = {e for seq in _dna_seqs(pdb_id) for e in _similar_entries(seq)} - {pdb_id}
        for pid in sorted(partners):
            try:
                w = _experimental_rmsd_windows(pdb_id, pid)
                if w:
                    values.append((pdb_id, pid, w))
            except Exception:
                pass
    except Exception:
        pass
    return values


test_pdb_ids = sorted({
    Path(test_dataset.base.data_list[w_idx]).parent.name
    for w_idx, _ in test_dataset.virtual_entries
})

rmsd_values = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(_noise_pairs_for_pdb, pid) for pid in test_pdb_ids]
    for future in as_completed(futures):
        for pdb_id, pid, wv in future.result():
            rmsd_values.extend(wv)
            print(f'{pdb_id} vs {pid}: median={np.median(wv):.2f} Å  n={len(wv)}')

if rmsd_values:
    print(f'\nЭкспериментальный порог  median={np.median(rmsd_values):.2f}  '
          f'mean={np.mean(rmsd_values):.2f}  n={len(rmsd_values)}')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rmsd_values, bins=50, color='skyblue', edgecolor='white')
    for mode, color in mode_colors.items():
        val = float(wide_per_mode[mode]['val_rmsd'].dropna().iloc[-1])
        ax.axvline(
            val,
            color=color,
            linewidth=2,
            label=f'валидационный RMSD, {validation_labels[mode]} ({val:.2f} Å)',
        )
    ax.axvline(
        np.median(rmsd_values),
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'медиана по экспериментальным структурам ({np.median(rmsd_values):.2f} Å)',
    )
    ax.set_xlim(0, 3)
    ax.set_xlabel('RMSD (Å)', fontsize=13)
    ax.set_ylabel('Количество структур', fontsize=13)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()

# %%
# Measure stochastic spread across independent diffusion runs on the same input
K = 10
N_SAMPLES = 60

_skip_atoms = {'OP1', 'OP2', 'P'}
_valid_bb_atoms = [a for a in utils.backbone_atoms if a not in _skip_atoms]
_idx_to_base = {v: k for k, v in utils.base_to_idx.items()}

_pool_size = len(test_dataset)
_inter_run_indices = rng.choice(
    _pool_size, size=min(N_SAMPLES, _pool_size), replace=False
).tolist()

inter_run_rmsds: list[float] = []
per_sample_medians: list[float] = []

with torch.no_grad():
    for _wi in tqdm(
            _inter_run_indices,
            desc='Inter-run RMSD',
            colour=utils.PBAR_COLOR,
    ):
        data = cast(Any, test_dataset[_wi].clone())
        tidx = int(data.target_nt_idx.item())
        restype = _idx_to_base[int(data.base_types[tidx].argmax().item())]
        _o3_ir = None
        if bool(data.o3_prev_valid[tidx].item()):
            _o3_ir = data.o3_prev_local[tidx].numpy()

        batch = cast(Any, Batch.from_data_list([data])).to(device)

        run_coords: list[dict] = []
        for _ in range(K):
            pred_theta, pred_tau_m = model.sample(batch)
            torsions_np = pred_theta[0].cpu().numpy()
            tau_m_val = float(pred_tau_m[0].clamp(min=1e-3).item())
            bb = build_backbone_from_torsions(
                torsions_np, restype, o3_prev_local=_o3_ir, tau_m=tau_m_val,
            )
            run_coords.append({k: v for k, v in bb.items() if k not in _skip_atoms})

        # intersection of atoms present in every run, ordered by backbone_atoms
        shared = set(run_coords[0].keys())
        for rc in run_coords[1:]:
            shared &= set(rc.keys())
        shared_atoms = [a for a in _valid_bb_atoms if a in shared]

        if len(shared_atoms) < 3:
            continue

        coords = np.array(
            [[rc[a] for a in shared_atoms] for rc in run_coords]
        )  # [K, n_atoms, 3]

        pair_rmsds: list[float] = []
        for r1 in range(K):
            for r2 in range(r1 + 1, K):
                diff = coords[r1] - coords[r2]
                pair_rmsds.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))

        inter_run_rmsds.extend(pair_rmsds)
        per_sample_medians.append(float(np.median(pair_rmsds)))

_med = float(np.median(inter_run_rmsds))
_mean = float(np.mean(inter_run_rmsds))
_p25, _p75 = np.percentile(inter_run_rmsds, [25, 75])
print(f'Inter-run RMSD  K={K}  N={len(per_sample_medians)} samples  '
      f'({K * (K - 1) // 2 * len(per_sample_medians)} pairs):')
print(f'  median={_med:.3f}  mean={_mean:.3f}  p25={_p25:.3f}  p75={_p75:.3f} Å')

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(inter_run_rmsds, bins=50, color='mediumpurple', edgecolor='white',
        label='попарный RMSD между независимыми прогонами')
ax.axvline(_med, color='black', linestyle='--', linewidth=1.8,
           label=f'медиана: {_med:.3f} Å')
if rmsd_values:
    _noise_med = float(np.median(rmsd_values))
    ax.axvline(_noise_med, color='steelblue', linestyle=':', linewidth=1.8,
               label=f'экспериментальный порог: {_noise_med:.3f} Å')
ax.set_xlabel('RMSD остова между прогонами (Å)', fontsize=13)
ax.set_ylabel('Количество структур', fontsize=13)
ax.legend(fontsize=11)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'inter_run_rmsd.png'), dpi=200, bbox_inches='tight')
plt.show()

# %%
# Compare with kNN baseline


# Reconstruct train split with the same fixed seed used during training
_dm = DNADataModule(batch_size=1)
_dm.setup()
train_dataset = _dm.train_dataset

_n_bb = len(utils.backbone_atoms)
_j_op1 = utils.backbone_atoms.index('OP1')
_j_op2 = utils.backbone_atoms.index('OP2')

# --- Build train feature matrix and local backbone targets ---
_train_feats: list[np.ndarray] = []
_train_locals: list[np.ndarray] = []  # each [n_bb, 3], NaN where atom absent

for _i in tqdm(
        range(len(train_dataset)),
        desc='kNN: building train index',
        colour=utils.PBAR_COLOR,
):
    _d = train_dataset[_i]
    _ti = int(_d.target_nt_idx.item())
    _train_feats.append(_d.rel_origins.flatten().numpy())          # [ws*3]
    _bb_w = _d.bb_xyz_world[_ti].numpy()                          # [n_bb, 3]
    _o = _d.nt_origins_world[_ti].numpy()                         # [3]
    _R = _d.nt_frames_world[_ti].numpy()                          # [3, 3]
    _train_locals.append((_bb_w - _o) @ _R)                       # local [n_bb, 3]

_train_feats_arr = np.array(_train_feats, dtype=np.float32)       # [N_train, ws*3]

_nn_index = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
_nn_index.fit(_train_feats_arr)


def _knn_local_rmsd(pred_local: np.ndarray, gt_local: np.ndarray) -> float:
    """RMSD in local frame; skips NaN atoms; permutation-invariant for OP1/OP2."""
    sq: list[float] = []
    for _j, _nm in enumerate(utils.backbone_atoms):
        if _nm in ('OP1', 'OP2'):
            continue
        _p, _g = pred_local[_j], gt_local[_j]
        if np.isnan(_p).any() or np.isnan(_g).any():
            continue
        sq.append(float(np.sum((_p - _g) ** 2)))
    # permutation-invariant matching for the two equivalent phosphate oxygens
    _p1, _p2 = pred_local[_j_op1], pred_local[_j_op2]
    _g1, _g2 = gt_local[_j_op1], gt_local[_j_op2]
    if not (np.isnan(_p1).any() or np.isnan(_p2).any()
            or np.isnan(_g1).any() or np.isnan(_g2).any()):
        _d_str = np.sum((_p1 - _g1) ** 2) + np.sum((_p2 - _g2) ** 2)
        _d_swp = np.sum((_p1 - _g2) ** 2) + np.sum((_p2 - _g1) ** 2)
        sq.append(float(min(_d_str, _d_swp)) / 2)
    return float(np.sqrt(np.mean(sq))) if sq else np.nan


# --- Evaluate on test dataset ---
_knn_rmsds: list[float] = []
_knn_is_edge: list[bool] = []

for _i in tqdm(
        range(len(test_dataset)),
        desc='kNN: evaluating test',
        colour=utils.PBAR_COLOR,
):
    _d = test_dataset[_i]
    _ti = int(_d.target_nt_idx.item())
    _feat = _d.rel_origins.flatten().numpy()[None].astype(np.float32)  # [1, ws*3]
    _, _nb_idx = _nn_index.kneighbors(_feat)
    _nn_i = int(_nb_idx[0, 0])

    _bb_w = _d.bb_xyz_world[_ti].numpy()
    _o = _d.nt_origins_world[_ti].numpy()
    _R = _d.nt_frames_world[_ti].numpy()
    _gt_local = (_bb_w - _o) @ _R

    _knn_rmsds.append(_knn_local_rmsd(_train_locals[_nn_i], _gt_local))
    _knn_is_edge.append(bool(_d.is_chain_edge_nt[_ti].item()))

_knn_rmsds_arr = np.array(_knn_rmsds, dtype=np.float64)
_knn_is_edge_arr = np.array(_knn_is_edge)

print('kNN backbone RMSD (local frame):')
for _label, _mask in [
    ('all',     np.ones(len(_knn_rmsds_arr), dtype=bool)),
    ('central', ~_knn_is_edge_arr),
    ('edge',    _knn_is_edge_arr),
]:
    _vals = _knn_rmsds_arr[_mask & np.isfinite(_knn_rmsds_arr)]
    if len(_vals):
        print(f'  {_label:>7}:  median={np.median(_vals):.3f}  mean={np.mean(_vals):.3f}'
              f'  n={len(_vals)}')

print('\nModel val RMSD (last epoch):')
for _label, _w in [('all', wide)] + [(m, wide_per_mode[m]) for m in ('central', 'edge')]:
    if 'val_rmsd' in _w.columns:
        _v = float(_w['val_rmsd'].dropna().iloc[-1])
        print(f'  {_label:>7}: {_v:.3f} Å')
