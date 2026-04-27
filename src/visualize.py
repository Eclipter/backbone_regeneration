# %%
# Imports
import os.path as osp
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
from io import StringIO
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import py3Dmol
import requests
import seaborn as sns
import torch
from Bio.PDB import Atom, Chain
from Bio.PDB import Model as PDBModel
from Bio.PDB import Residue, Structure
from Bio.PDB.mmcifio import MMCIFIO
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tensorboard.backend.event_processing import event_accumulator
from torch_geometric.data import Batch
from tqdm import tqdm

import utils
from dataset import PyGDataset
from model import PytorchLightningModule
from predict import write_structure

# %%
# Load model and dataset
log_dir = osp.join('..', 'logs')
run_filename = osp.join('fixed_phosphorus', 'baseline')
run_dir = osp.join(log_dir, run_filename)
ckpt_path = utils.find_best_checkpoint(run_dir)
test_dataset_path = osp.join(run_dir, 'test_dataset.pt')
event_files = glob(osp.join(run_dir, 'events.*'))

try:
    test_dataset = torch.load(test_dataset_path, weights_only=False)
except FileNotFoundError:
    raise FileNotFoundError(f'File `{test_dataset_path}` not found. Ensure training completed successfully.')

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
    'central': mcolors.to_rgba(utils.PBAR_COLOR, 0.5),
    'edge': mcolors.to_rgba('violet', 0.5),
}
mode_linestyles = {
    'all': '-',
    'central': '--',
    'edge': '--',
}

try:
    model = PytorchLightningModule.load_from_checkpoint(ckpt_path, weights_only=False, map_location='cpu').eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:2'
    model.to(device)
    print(device)
except FileNotFoundError:
    raise FileNotFoundError(f'Best checkpoint `{ckpt_path}` not found. Ensure training completed successfully.')

# %%
# Load dataset
dataset = PyGDataset()
rng = np.random.default_rng()
idx = rng.integers(len(dataset))
data = dataset[idx]

pdb_id = Path(dataset.data_list[idx]).parent.name

pos_local = data.pos.numpy()  # type: ignore
edge_index = data.edge_index.numpy() if (hasattr(data, 'edge_index') and data.edge_index is not None) else None  # type: ignore
central_mask = data.central_mask.numpy().astype(bool)  # type: ignore

central_atoms_pos = pos_local[central_mask]
centroid_local = np.mean(central_atoms_pos, axis=0)
dist_to_origin = np.linalg.norm(centroid_local)

# --- LOCAL VIEW (ALIGNED) ---
fig = go.Figure()

# Non-central atoms
non_central_pos = pos_local[~central_mask]
fig.add_trace(go.Scatter3d(
    x=non_central_pos[:, 0],
    y=non_central_pos[:, 1],
    z=non_central_pos[:, 2],
    mode='markers',
    marker=dict(size=5, color='#B366FF', opacity=0.5),
    name='Соседние нуклеотиды'
))

# Central atoms
fig.add_trace(go.Scatter3d(
    x=central_atoms_pos[:, 0],
    y=central_atoms_pos[:, 1],
    z=central_atoms_pos[:, 2],
    mode='markers',
    marker=dict(size=10, color='red', opacity=1.0),
    name='Центральный нуклеотид'
))

# Add bonds
if edge_index is not None:
    lines_x, lines_y, lines_z = [], [], []
    for i in range(edge_index.shape[1]):
        p1 = pos_local[edge_index[0, i]]
        p2 = pos_local[edge_index[1, i]]
        lines_x.extend([p1[0], p2[0], None])
        lines_y.extend([p1[1], p2[1], None])
        lines_z.extend([p1[2], p2[2], None])

    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='gray', width=2),
        name='Связи'
    ))

# --- CUSTOM AXES (ARROWS) ---
axis_len = 5.0

# 1. Lines
fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5), name='Local X', showlegend=False))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines', line=dict(color='green', width=5), name='Local Y', showlegend=False))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines', line=dict(color='blue', width=5), name='Local Z', showlegend=False))

# 2. Cones (Arrowheads)
fig.add_trace(go.Cone(
    x=[axis_len], y=[0], z=[0],
    u=[1], v=[0], w=[0],
    showscale=False, colorscale=[[0, 'red'], [1, 'red']],
    sizemode="absolute", sizeref=0.5, anchor='tail', name='X Arrow'
))
fig.add_trace(go.Cone(
    x=[0], y=[axis_len], z=[0],
    u=[0], v=[1], w=[0],
    showscale=False, colorscale=[[0, 'green'], [1, 'green']],
    sizemode="absolute", sizeref=0.5, anchor='tail', name='Y Arrow'
))
fig.add_trace(go.Cone(
    x=[0], y=[0], z=[axis_len],
    u=[0], v=[0], w=[1],
    showscale=False, colorscale=[[0, 'blue'], [1, 'blue']],
    sizemode="absolute", sizeref=0.5, anchor='tail', name='Z Arrow'
))

# 3. Text Labels
# Shifted labels to the side and origin raised
fig.add_trace(go.Scatter3d(
    x=[axis_len, 0.5, 0, 0],
    y=[0.5, axis_len, 0.5, 0],
    z=[0, 0, axis_len, 0.2],
    mode='text',
    text=['X', 'Y', 'Z', '(0, 0, 0)'],
    textposition=['middle right', 'middle right', 'top center', 'top center'],
    textfont=dict(size=14, color='black'),
    showlegend=False
))

fig.update_layout(
    title=f'PDB ID: {pdb_id}',
    scene=dict(
        xaxis=dict(visible=False),  # Hide default axis
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    width=800, height=800,
    margin=dict(r=0, l=0, b=0, t=40)
)
fig.show()


# %%
# Model
plt.rcParams['font.family'] = 'DejaVu Sans'

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
    'equiv': '#EAE3F5',
    'output': '#D8F0DF',
    'detail': '#FFF8D6',
    'feature': '#F0F8FF'
}

# hidden_dim = int(model.hparams.hidden_dim)
# num_layers = int(model.hparams.num_layers)
# time_emb_dim = int(model.time_emb_dim)
# n_atom_types = int(model.n_atom_types)
# n_base_types = len(utils.base_to_idx)
# n_chain_end_classes = int(utils.N_CHAIN_END_CLASSES)
# input_feature_dim = int(model.gnn.embedding_in.in_features)

hidden_dim = 256
num_layers = 5
time_emb_dim = 32
n_atom_types = 27
n_base_types = 4
n_chain_end_classes = 2
input_feature_dim = 256 + 32 + 4 + 2 + 2


def draw_box(x, y, width, height, title, desc, desc_align, color):
    ax.add_patch(FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle='round,pad=0.15',
        facecolor=color,
        linewidth=2.2
    ))

    # Title
    ax.text(
        x + width/2,
        y + height,
        title,
        ha='center',
        va='top',
        fontsize=13,
        fontweight='bold'
    )
    # Description
    if desc_align == 'left':
        ax.text(
            x + width*0.01,
            y + height - 0.4,
            desc,
            ha=desc_align,
            va='top',
            linespacing=1.6
        )
    else:
        ax.text(
            x + width/2,
            y + height - 0.4,
            desc,
            ha=desc_align,
            va='top',
            linespacing=1.6
        )


def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle='-|>',
        linewidth=2,
        mutation_scale=25,  # Arrow head size
        color='black'
    )
    ax.add_patch(arrow)


# ========== MAIN FLOW (left column) ==========
main_x = MAX_X * 0.2
main_y = MAX_Y * 0.85
main_width = MAX_X * 0.25
main_height = MAX_Y * 0.1
arrow_length = MAX_Y * 0.05

detail_x = MAX_X * 0.7
detail_y = MAX_Y * 0.72
detail_width = MAX_X * 0.22
step_stride = main_height + arrow_length

input_y = main_y
prep_y = main_y - step_stride
embed_y = main_y - 2 * step_stride
egnn_y = main_y - 3 * step_stride
eps_y = main_y - 4 * step_stride
output_y = main_y - 5 * step_stride

# Input
draw_box(
    main_x,
    input_y,
    main_width,
    main_height,
    r'Входные данные ($x_t$)',
    fr'• $h \in \mathbb{{R}}^{{N \times {n_atom_types}}}$ (типы атомов)' + '\n' +
    r'• $x \in \mathbb{R}^{N \times 3}$ (координаты окна)' + '\n' +
    r'• $x_t^{\mathrm{target}} \in \mathbb{R}^{M \times 3}$ (зашумленные backbone)' + '\n' +
    r'• $\mathrm{edge\_index} \in \mathbb{Z}^{2 \times E}$, $t$',
    'left',
    palette['input']
)
draw_arrow(
    main_x + main_width/2,
    input_y-MAX_Y*0.01,
    main_x + main_width/2,
    input_y-MAX_Y*0.04
)

# Input preparation
draw_box(
    main_x,
    prep_y,
    main_width,
    main_height,
    'Подготовка входа',
    r'строим $\mathrm{target\_mask} = \mathrm{is\_target} \wedge \mathrm{backbone\_mask}$' + '\n' +
    r'разворачиваем $t$ по узлам, считаем' + '\n' +
    r'синусоидальный embedding и подставляем' + '\n' +
    fr'в граф только $x_t^{{\mathrm{{target}}}}$, итог: $\mathbb{{R}}^{{N \times {input_feature_dim}}}$',
    'left',
    palette['pre']
)
draw_arrow(
    detail_x,
    prep_y + main_height*0.55,
    main_x + main_width*1.05,
    prep_y + main_height*0.55
)
draw_box(
    main_x+main_width+MAX_X*0.1,
    prep_y,
    main_width*1.1,
    main_height,
    'Собираемые признаки',
    fr'• one-hot атома: ${n_atom_types}$, embedding времени: ${time_emb_dim}$' + '\n' +
    fr'• тип основания: ${n_base_types}$, has pair: $1$' + '\n' +
    fr'• chain-end class: ${n_chain_end_classes}$, is_target: $1$',
    'left',
    palette['embed']
)
draw_arrow(
    main_x + main_width/2,
    prep_y-MAX_Y*0.01,
    main_x + main_width/2,
    prep_y-MAX_Y*0.04
)

# Embedding
draw_box(
    main_x,
    embed_y,
    main_width,
    main_height,
    'Входной MLP слой',
    fr'линейный слой: ${input_feature_dim} \rightarrow {hidden_dim}$' + '\n' +
    'SiLU активация\n' +
    r'координаты и ребра передаем' + '\n' +
    r'дальше без изменения формы',
    'left',
    palette['embed']
)
draw_arrow(
    main_x + main_width/2,
    embed_y-MAX_Y*0.01,
    main_x + main_width/2,
    embed_y-MAX_Y*0.04
)

# Equivariant blocks
draw_box(
    main_x,
    egnn_y,
    main_width,
    main_height,
    'EGNN backbone',
    fr'SE(3)-эквивариантная свертка ({num_layers}x):' + '\n' +
    '• Сообщения по ребрам\n' +
    '• Агрегирование по узлам\n' +
    '• Обновление координат\n' +
    '• Обновление признаков',
    'left',
    palette['equiv']
)
draw_arrow(
    main_x + main_width/2,
    egnn_y-MAX_Y*0.01,
    main_x + main_width/2,
    egnn_y-MAX_Y*0.04
)

# Noise head and DDPM decode
draw_box(
    main_x,
    eps_y,
    main_width,
    main_height,
    r'$\epsilon$-голова + DDPM шаг',
    r'из финальных $h, x$ считаем' + '\n' +
    r'$\epsilon_\theta(x_t, t)$ только для цели' + '\n' +
    r'восстанавливаем $\hat{x}_0$ и параметры' + '\n' +
    r'$q(x_{t-1} \mid x_t, \hat{x}_0)$',
    'left',
    palette['output']
)
draw_arrow(
    main_x + main_width/2,
    eps_y-MAX_Y*0.01,
    main_x + main_width/2,
    eps_y-MAX_Y*0.04
)

# Output
draw_box(
    main_x,
    output_y,
    main_width,
    main_height,
    r'Выходные данные ($x_{t-1}$)',
    r'$x_{t-1}^{\mathrm{target}} \in \mathbb{R}^{M \times 3}$ (новые позиции цели)' + '\n' +
    r'контекстные атомы окна остаются фиксированными',
    'left',
    palette['output']
)

# ========== DETAIL OPERATIONS (right column) ==========
detail_steps = [
    ('Относительные координаты',
     r'$\vec{r}_{ij} = x_i - x_j$' + '\n' + r'$d_{ij} = \|\vec{r}_{ij}\|$'),
    ('Инвариантные сообщения',
     r'$m_{ij} = \text{MLP}([h_i, h_j, d_{ij}])$'),
    ('Агрегация',
     r'$\bar{m}_i = \sum_j m_{ji}$'),
    ('Обновление координат',
     r'$\Delta x_i = \sum_j \phi_x(m_{ji}) \frac{\vec{r}_{ji}}{d_{ji} + \epsilon}$'),
    ('Обновление признаков',
     r'$h_i := \mathrm{LayerNorm}(h_i + \text{MLP}([h_i, \bar{m}_i]))$'),
    (r'Предсказание $\epsilon$',
     r'$\epsilon_i = \sum_j \phi_\epsilon(m_{ji}) \frac{\vec{r}_{ji}}{d_{ji} + \epsilon}$')
]

detail_palette_steps = ['#FCE4EC', '#F3E5F5', '#E8F5E9', '#FFF8E1', '#E0F2F1', '#E3F2FD']
step_height = 1.8
detail_box_height = step_height - 0.55

for idx, (step_title, step_desc) in enumerate(detail_steps):
    y = detail_y - (idx + 1) * step_height
    draw_box(
        detail_x,
        y,
        detail_width,
        detail_box_height,
        step_title,
        step_desc,
        'center',
        detail_palette_steps[idx]
    )
    # Arrow between steps
    if idx < len(detail_steps) - 1:
        ax.add_patch(FancyArrowPatch(
            (detail_x + detail_width/2, y-0.2),
            (detail_x + detail_width/2, y-0.55),
            linewidth=2,
            arrowstyle='-|>',
            mutation_scale=10,  # Arrow head size
            color='black'
        ))


# ============ Lines connecting core blocks to their details ============
egnn_height = main_height
egnn_x_right = main_x + main_width + MAX_X*0.02
egnn_y_center = egnn_y + egnn_height/2

eps_height = main_height
eps_x_right = main_x + main_width + MAX_X*0.02
eps_y_center = eps_y + eps_height/2

rel_coords_y_bottom = detail_y - step_height
rel_coords_height = detail_box_height
rel_coords_y_top = rel_coords_y_bottom + rel_coords_height

eps_detail_y_bottom = detail_y - len(detail_steps)*step_height

# Line to top block
ax.add_patch(FancyArrowPatch(
    (egnn_x_right, egnn_y_center),
    (detail_x - MAX_X*0.02, rel_coords_y_top),
    linewidth=2,
    arrowstyle='-'  # Draw a line, not an arrow
))

# Line to bottom block
ax.add_patch(FancyArrowPatch(
    (eps_x_right, eps_y_center),
    (detail_x - MAX_X*0.02, eps_detail_y_bottom),
    linewidth=2,
    arrowstyle='-'  # Draw a line, not an arrow
))


# ============ Diffusion arrow ============
out_block_y = output_y
out_block_x_center = main_x + main_width/2

in_block_y = input_y
in_block_height = main_height
in_block_top_y = in_block_y + in_block_height
in_block_x_center = main_x + main_width/2

arrow_path_data = [
    (mpath.Path.MOVETO, (out_block_x_center, out_block_y)),  # Start from bottom of Output block
    (mpath.Path.LINETO, (out_block_x_center, out_block_y - 0.6)),  # Go down
    (mpath.Path.LINETO, (1.5, out_block_y - 0.6)),  # Go left
    (mpath.Path.LINETO, (1.5, in_block_top_y + 0.6)),  # Go up
    (mpath.Path.LINETO, (in_block_x_center, in_block_top_y + 0.6)),  # Go right
    (mpath.Path.LINETO, (in_block_x_center, in_block_top_y)),  # Connect to top of Input block
]
codes, verts = zip(*arrow_path_data)
arrow_path = mpath.Path(verts, codes)

ax.add_patch(FancyArrowPatch(
    path=arrow_path,
    arrowstyle='-|>',
    linewidth=2,
    mutation_scale=25,  # Arrow head size
    color='black',
    linestyle='--'
))
ax.text(
    1.2,
    MAX_Y/2,
    r'$x_t^{\mathrm{target}} \rightarrow x_{t-1}^{\mathrm{target}}$' + '\n' + r'$(T \text{ шагов DDPM})$',
    fontsize=15,
    rotation=90,
    ha='center',
    va='center'
)

plt.tight_layout()
diagram_path = osp.join('..', 'data', 'model.png')
plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
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
        print(f'  example buffers for param {_pid0}:', {k: tuple(v.shape) if hasattr(v, 'shape') else v for k, v in _st[_pid0].items()})
for i, sch in enumerate(_ck.get('lr_schedulers') or []):
    print(f'\nlr_schedulers[{i}]:', {k: v for k, v in sch.items() if k in ('_last_lr', 'last_epoch', 'best', 'num_bad_epochs', 'cooldown_counter', 'factor', 'patience')})

# %%
# Training
plt.rcParams['font.family'] = 'Nunito'


def load_event_accumulator(path):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={'scalars': 0, 'histograms': 0, 'images': 0}
    )
    ea.Reload()
    return ea


def scalars_to_dataframe(ea, tag):
    rows = []
    for s in ea.Scalars(tag):
        rows.append((s.step, s.value))
    return pd.DataFrame(rows, columns=['epoch', 'value'])


# Unified-model logging: train_rmse is noise-prediction RMSE, val_* is coordinate RMSD.
metric_tags = {
    'train_rmse': ('all', 'train_noise_rmse'),
    'val_rmsd': ('all', 'val_rmsd'),
    'central_val_rmsd': ('central', 'val_rmsd'),
    'edge_val_rmsd': ('edge', 'val_rmsd'),
    'test_rmsd': ('all', 'test_rmsd'),
    'central_test_rmsd': ('central', 'test_rmsd'),
    'edge_test_rmsd': ('edge', 'test_rmsd'),
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
    raise ValueError(f'No tracked TensorBoard scalars found in `{run_dir}`.')

scalars_by_epoch = pd.concat(dfs, ignore_index=True)[['epoch', 'mode', 'metric', 'value']] \
    .reset_index(drop=True)
# Aggregate duplicates per (epoch, metric): keep the last logged value (handles resumed runs / multiple event files).
wide_scalars_per_mode = {
    mode: scalars_by_epoch.loc[scalars_by_epoch['mode'] == mode].pivot_table(
        index='epoch',
        columns='metric',
        values='value',
        aggfunc='last'
    )
    for mode in target_modes
}
wide_scalars = wide_scalars_per_mode['all']


def _plot_metric(ax, table, column, color, label, linestyle='-'):
    values = table[column].dropna()
    return ax.plot(
        values.index.to_numpy(),
        values.to_numpy(),
        color=color,
        linewidth=3,
        label=label,
        linestyle=linestyle
    )[0]


fig, ax = plt.subplots(figsize=(7, 4))
ax.tick_params(axis='both', labelsize=15)
_plot_metric(
    ax,
    wide_scalars,
    'train_noise_rmse',
    'indigo',
    'train_rmse',
)
ax.axvline(17, color='red', linewidth=3, linestyle='--')
ax.text(
    17.5,
    0.95,
    'Старт стохастического\nусреднения весов',
    transform=ax.get_xaxis_transform(),
    color='red',
    fontsize=14,
    va='top',
    ha='left',
)
ax.set_xlabel('Эпоха', fontsize=18)
ax.set_ylabel('RMSE шума 1 шага (Å)', fontsize=18)
sns.despine(ax=ax, top=True, right=True)

fig.tight_layout()
fig.savefig(
    osp.join(run_dir, 'train.png'),
    bbox_inches='tight',
    dpi=300
)

plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.tick_params(axis='both', labelsize=15)
validation_labels = {
    'all': 'все нуклеотиды',
    'central': 'центральные нуклеотиды',
    'edge': 'краевые нуклеотиды',
}
for mode in target_modes:
    _plot_metric(
        ax,
        wide_scalars_per_mode[mode],
        'val_rmsd',
        mode_colors[mode],
        validation_labels[mode],
        mode_linestyles[mode]
    )
ax.set_yscale('log')
ax.set_xlabel('Эпоха', fontsize=18)
ax.set_ylabel('RMSD (Å)', fontsize=18)
ax.legend(fontsize=14)
sns.despine(ax=ax, top=True, right=True)

fig.tight_layout()
fig.savefig(
    osp.join(run_dir, 'val.png'),
    bbox_inches='tight',
    dpi=300
)

plt.show()

fig, ax = plt.subplots(figsize=(6, 3))
ax.tick_params(axis='both', labelsize=15)
test_values = [
    float(wide_scalars_per_mode[mode]['test_rmsd'].dropna().iloc[-1])
    for mode in target_modes
]
bars = ax.barh(
    [validation_labels[mode] for mode in target_modes],
    test_values,
    color=[mode_colors[mode] for mode in target_modes]
)
ax.bar_label(bars, labels=[f'{value:.2f}' for value in test_values], fontsize=14, padding=4)
ax.set_ylabel('RMSD (Å)', fontsize=18)
sns.despine(ax=ax, top=True, right=True)

fig.tight_layout()
fig.savefig(
    osp.join(run_dir, 'test.png'),
    bbox_inches='tight',
    dpi=300
)

plt.show()

# %%
# Results
_virtual_entries = test_dataset.virtual_entries
_base_paths = test_dataset.base.data_list
test_paths = [_base_paths[w_idx] for w_idx, _ in _virtual_entries]
test_target_types = [tt for _, tt in _virtual_entries]
test_pdb_to_local: dict[str, list[int]] = defaultdict(list)
for local_i, p in enumerate(test_paths):
    test_pdb_to_local[Path(p).parent.name].append(local_i)

sample_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
sample_idx = random.choice(test_pdb_to_local[sample_pdb_id])
sample_window = int(Path(test_paths[sample_idx]).stem)
data = test_dataset[sample_idx].clone()

# Build masks once to express all visualization classes.
pos_full = data.pos.cpu().numpy()
central_mask = data.central_mask.cpu().numpy().astype(bool)
backbone_mask = data.backbone_mask.cpu().numpy().astype(bool)
side_mask = ~central_mask
base_mask = ~backbone_mask
# Mirror the model's internal target mask (drives the `is_target` flag) so the
# indexing of `pred_backbone` stays consistent with the sample being predicted.
target_mask = model._target_mask(data).cpu().numpy().astype(bool)

edge_index = data.edge_index.cpu().numpy() if data.edge_index is not None else None
true_backbone = pos_full[target_mask]

with torch.no_grad():
    pred_backbone_raw = model.sample(data.clone().to(device)).cpu().numpy()

# Apply translation-only alignment for visualization to remove possible global drift.
# The model predicts coordinates, while this correction only matches global centroid.
centroid_shift = true_backbone.mean(axis=0) - pred_backbone_raw.mean(axis=0)
pred_backbone = pred_backbone_raw + centroid_shift

# WindowTargetDataset stores edge-target samples in the edge nucleotide's
# frame and central-target samples in the central nucleotide's frame, so pick
# whichever matches the current `is_target` mask.
is_target_mask = data.is_target.cpu().numpy().astype(bool)
frame_atom_idx = int(is_target_mask.nonzero()[0][0])
origin_true = data.origins[frame_atom_idx].cpu().numpy()
ref_frame_true = data.ref_frames[frame_atom_idx].cpu().numpy()

fig = go.Figure()

# Draw full true-window bonds with a two-layer style to improve visibility.
if edge_index is not None:
    lines_x, lines_y, lines_z = [], [], []
    for i in range(edge_index.shape[1]):
        src_idx, dst_idx = int(edge_index[0, i]), int(edge_index[1, i])
        if src_idx < dst_idx:
            p1 = pos_full[src_idx]
            p2 = pos_full[dst_idx]
            lines_x.extend([p1[0], p2[0], None])
            lines_y.extend([p1[1], p2[1], None])
            lines_z.extend([p1[2], p2[2], None])
    fig.add_trace(go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        mode='lines',
        line=dict(color='rgba(35, 35, 35, 0.55)', width=6),
        name='True bonds (window, outline)'
    ))
    fig.add_trace(go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        mode='lines',
        line=dict(color='rgba(210, 210, 210, 0.95)', width=3),
        name='True bonds (window)'
    ))

# Encode three classifications directly in legend and marker style:
# 1) backbone/base via color
# 2) central/side via size and edge thickness
# 3) true/generated via base color palette and outline style
true_classes = [
    ('True | Central | Backbone', central_mask & backbone_mask, '#d62728', 8, 0.96, 1.2),
    ('True | Central | Base', central_mask & base_mask, '#2ca02c', 7, 0.94, 1.0),
    ('True | Side | Backbone', side_mask & backbone_mask, '#ff9896', 7, 0.9, 1.0),
]

for name, mask, color, size, opacity, edge_width in true_classes:
    if np.any(mask):
        pts = pos_full[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                opacity=opacity,
                symbol='circle',
                line=dict(width=edge_width, color='rgba(20, 20, 20, 0.65)')
            ),
            name=name
        ))

# Draw side-base atoms separately at the top layer so they are not visually lost.
side_base_mask = side_mask & base_mask
if np.any(side_base_mask):
    side_base_pts = pos_full[side_base_mask]
    fig.add_trace(go.Scatter3d(
        x=side_base_pts[:, 0],
        y=side_base_pts[:, 1],
        z=side_base_pts[:, 2],
        mode='markers',
        marker=dict(
            size=9,
            color='#6acb3d',
            opacity=0.98,
            symbol='circle',
            line=dict(width=1.4, color='rgba(15, 55, 10, 0.95)')
        ),
        name='True | Side | Base'
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
        line=dict(width=2.4, color='rgba(10, 10, 10, 0.9)')
    ),
    name='Generated | Backbone'
))

# Build generated bonds by reusing central-backbone topology from the true graph.
if edge_index is not None:
    target_indices = np.where(target_mask)[0]
    target_global_to_local = {g: i for i, g in enumerate(target_indices)}
    gen_bonds_x, gen_bonds_y, gen_bonds_z = [], [], []
    for i in range(edge_index.shape[1]):
        src_idx, dst_idx = int(edge_index[0, i]), int(edge_index[1, i])
        if src_idx in target_global_to_local and dst_idx in target_global_to_local and src_idx < dst_idx:
            p1 = pred_backbone[target_global_to_local[src_idx]]
            p2 = pred_backbone[target_global_to_local[dst_idx]]
            gen_bonds_x.extend([p1[0], p2[0], None])
            gen_bonds_y.extend([p1[1], p2[1], None])
            gen_bonds_z.extend([p1[2], p2[2], None])
    if len(gen_bonds_x) > 0:
        fig.add_trace(go.Scatter3d(
            x=gen_bonds_x,
            y=gen_bonds_y,
            z=gen_bonds_z,
            mode='lines',
            line=dict(color='rgba(8, 65, 140, 0.95)', width=6),
            name='Generated bonds (topology from true graph)'
        ))

# Add pairwise GT->generated correspondence lines (same atom order as target_mask indexing).
if true_backbone.shape[0] == pred_backbone.shape[0]:
    corr_x, corr_y, corr_z = [], [], []
    for i in range(true_backbone.shape[0]):
        p_true = true_backbone[i]
        p_pred = pred_backbone[i]
        corr_x.extend([p_true[0], p_pred[0], None])
        corr_y.extend([p_true[1], p_pred[1], None])
        corr_z.extend([p_true[2], p_pred[2], None])
    fig.add_trace(go.Scatter3d(
        x=corr_x,
        y=corr_y,
        z=corr_z,
        mode='lines',
        line=dict(color='rgba(10, 25, 50, 0.82)', width=6, dash='dot'),
        name='True -> Generated links (outline)'
    ))
    fig.add_trace(go.Scatter3d(
        x=corr_x,
        y=corr_y,
        z=corr_z,
        mode='lines',
        line=dict(color='rgba(40, 170, 255, 0.98)', width=3, dash='dot'),
        name='True -> Generated links'
    ))

# Scale the displayed frame to current sample extent while keeping the origin fixed.
all_points = np.vstack([pos_full, pred_backbone])
radius = float(np.max(np.linalg.norm(all_points, axis=1)))
axis_len = max(4.0, radius * 1.15)

fig.add_trace(go.Scatter3d(
    x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines',
    line=dict(color='red', width=5), showlegend=False, name='Frame X'
))
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines',
    line=dict(color='green', width=5), showlegend=False, name='Frame Y'
))
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines',
    line=dict(color='blue', width=5), showlegend=False, name='Frame Z'
))

fig.add_trace(go.Cone(
    x=[axis_len], y=[0], z=[0], u=[1], v=[0], w=[0],
    showscale=False, colorscale=[[0, 'red'], [1, 'red']],
    sizemode='absolute', sizeref=0.45, anchor='tail', name='Frame X arrow'
))
fig.add_trace(go.Cone(
    x=[0], y=[axis_len], z=[0], u=[0], v=[1], w=[0],
    showscale=False, colorscale=[[0, 'green'], [1, 'green']],
    sizemode='absolute', sizeref=0.45, anchor='tail', name='Frame Y arrow'
))
fig.add_trace(go.Cone(
    x=[0], y=[0], z=[axis_len], u=[0], v=[0], w=[1],
    showscale=False, colorscale=[[0, 'blue'], [1, 'blue']],
    sizemode='absolute', sizeref=0.45, anchor='tail', name='Frame Z arrow'
))

fig.add_trace(go.Scatter3d(
    x=[axis_len, 0.3, 0, 0],
    y=[0.3, axis_len, 0.3, 0],
    z=[0, 0, axis_len, 0.25],
    mode='text',
    text=['X', 'Y', 'Z', '(0, 0, 0)'],
    textposition=['middle right', 'middle right', 'top center', 'top center'],
    textfont=dict(size=14, color='black'),
    showlegend=False
))

fig.update_layout(
    title=(
        f'Results: run={run_filename}, '
        f'pdb_id={sample_pdb_id}, window={sample_window}'
    ),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    width=1300,
    height=850,
    margin=dict(r=10, l=10, b=10, t=50),
    legend=dict(itemsizing='constant')
)

fig.show()

# %%
# Inference
# Map atom vocabulary index back to its canonical atom name / chemical element so
# that the resulting CIF files carry meaningful atom metadata.
idx_to_atom_name = {v: k for k, v in utils.atom_to_idx.items()}


def _atom_element(atom_name: str) -> str:
    # Element is the first alphabetic character, except standalone phosphorus.
    return 'P' if atom_name == 'P' else atom_name[0]


def graph_to_cif_string(atom_types, pos, structure_name='structure'):
    structure = Structure.Structure(structure_name)
    model = PDBModel.Model(0)
    chain = Chain.Chain('A')
    residue = Residue.Residue((' ', 1, ' '), 'UNK', ' ')

    for i, (atom_type_idx, coord) in enumerate(zip(atom_types, pos)):
        atom_name = idx_to_atom_name.get(int(atom_type_idx), 'X')
        element = _atom_element(atom_name)
        unique_atom_name = f'{atom_name}_{i+1}'
        atom = Atom.Atom(
            name=unique_atom_name, coord=coord, bfactor=0, occupancy=1.0, altloc=' ',
            fullname=unique_atom_name, serial_number=i+1, element=element
        )
        residue.add(atom)

    chain.add(residue)
    model.add(chain)
    structure.add(model)

    cif_io = StringIO()
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(cif_io)
    return cif_io.getvalue()


def _window_atom_meta_training_order(window):
    return [
        (nt.segid, int(nt.resid), name)
        for nt in window
        for name, _ in utils.default_atoms_provider(nt)
    ]


def _chain_windows_by_processed_idx(chain_records):
    return [
        window_record
        for _, _, windows in chain_records
        for window_record in windows
    ]


inference_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
inference_local_indices = sorted(
    test_pdb_to_local[inference_pdb_id],
    key=lambda i: int(Path(test_paths[i]).stem)
)
print(f'Reconstructing backbone for structure {inference_pdb_id}: '
      f'{len(inference_local_indices)} windows')

inference_windows = [
    test_dataset[local_i].clone()
    for local_i in tqdm(inference_local_indices, colour=utils.PBAR_COLOR)
]
raw_inference_path = osp.join('..', 'data', 'raw', f'{inference_pdb_id}.cif')
_, inference_chain_records = utils.parse_dna(
    raw_inference_path,
    use_full_nucleotide=True,
    window_size=test_dataset.base.window_size,
)
inference_chain_windows = _chain_windows_by_processed_idx(inference_chain_records)
target_atom_meta = []
for local_i, window_data in zip(inference_local_indices, inference_windows):
    window_idx = int(Path(test_paths[local_i]).stem)
    window, _, _ = inference_chain_windows[window_idx]
    window_meta = _window_atom_meta_training_order(window)
    target_idx = (window_data.is_target & window_data.backbone_mask).nonzero(as_tuple=True)[0].tolist()
    target_atom_meta.extend(window_meta[i] for i in target_idx)

frame_offsets = torch.tensor(
    [int(window.is_target.nonzero(as_tuple=True)[0][0]) for window in inference_windows],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    batched_windows = Batch.from_data_list(inference_windows).to(device)  # type: ignore
    mask = model._target_mask(batched_windows)
    gen_local = model.sample(batched_windows).float()

    # Invert the dataset's local transform for every window in the PyG batch.
    target_graph_ids = batched_windows.batch[mask]
    frame_atom_indices = batched_windows.ptr[:-1].to(device) + frame_offsets
    ref_frames = batched_windows.ref_frames[frame_atom_indices].float()[target_graph_ids]
    origins = batched_windows.origins[frame_atom_indices].float()[target_graph_ids].reshape(-1, 3)

    true_local = batched_windows.pos[mask].float()
    generated_pos = (
        torch.bmm(gen_local.unsqueeze(1), ref_frames.transpose(1, 2)).squeeze(1) + origins
    ).cpu().numpy()
    original_pos = (
        torch.bmm(true_local.unsqueeze(1), ref_frames.transpose(1, 2)).squeeze(1) + origins
    ).cpu().numpy()
    atom_types = torch.argmax(batched_windows.x[mask], dim=1).cpu().numpy()

generated_cif_data = graph_to_cif_string(atom_types, generated_pos, 'generated_structure')
original_cif_data = graph_to_cif_string(atom_types, original_pos, 'original_structure')
if len(target_atom_meta) != len(generated_pos):
    raise RuntimeError(
        f'Target metadata count ({len(target_atom_meta)}) does not match '
        f'predicted atom count ({len(generated_pos)}).'
    )
generated_predictions = {
    meta: xyz
    for meta, xyz in zip(target_atom_meta, generated_pos)
}
generated_pdb_path = osp.join(run_dir, f'generated_backbone_{inference_pdb_id}.pdb')
write_structure(inference_chain_records, generated_predictions, generated_pdb_path)
print(f'Saved generated backbone PDB to {generated_pdb_path}')

view = py3Dmol.view(width=800, height=400, linked=False, viewergrid=(1, 2))

view.addModel(generated_cif_data, 'cif', viewer=(0, 0))
view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}}, viewer=(0, 0))
view.addLabel(
    f'Generated backbone ({inference_pdb_id})',
    {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
    viewer=(0, 0)
)

view.addModel(original_cif_data, 'cif', viewer=(0, 1))
view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}}, viewer=(0, 1))
view.addLabel(
    f'Original backbone ({inference_pdb_id})',
    {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
    viewer=(0, 1)
)

view.zoomTo()
view.show()

# %%
# Estimate dataset noise


@lru_cache
def _dna_seqs(pdb_id):
    q = '''{ entry(entry_id: "%s") { polymer_entities { entity_poly {
        rcsb_entity_polymer_type pdbx_seq_one_letter_code_can } } } }''' % pdb_id
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
        'return_type': 'entry'
    }, timeout=15)
    return [x['identifier'] for x in r.json().get('result_set', [])]


@lru_cache
def _parsed_chain_records(pdb_id):
    path = osp.join('..', 'data', 'raw', f'{pdb_id}.cif')
    if not osp.exists(path):
        url = f'https://files.rcsb.org/download/{pdb_id}.cif'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(requests.get(url, timeout=60).content)
    _, chain_records = utils.parse_dna(
        path,
        use_full_nucleotide=True,
        window_size=test_dataset.base.window_size,
    )
    return chain_records


def _central_window_index(chain_records):
    idx = {}
    central_offset = test_dataset.base.window_size // 2
    for _, _, windows in chain_records:
        for window, _, data in windows:
            central_nt = window[central_offset]
            idx[(central_nt.segid, int(central_nt.resid))] = data
    return idx


def _central_backbone_by_atom_name(data):
    mask = data.central_mask & data.backbone_mask
    atom_type_ids = torch.argmax(data.x[mask], dim=1).cpu().numpy()
    positions = data.pos[mask].cpu().numpy()
    return {
        idx_to_atom_name[int(atom_type_id)]: position
        for atom_type_id, position in zip(atom_type_ids, positions)
    }


def _experimental_rmsd_windows(pdb_id1, pdb_id2):
    idx1 = _central_window_index(_parsed_chain_records(pdb_id1))
    idx2 = _central_window_index(_parsed_chain_records(pdb_id2))

    values = []
    for key in set(idx1) & set(idx2):
        atoms1 = _central_backbone_by_atom_name(idx1[key])
        atoms2 = _central_backbone_by_atom_name(idx2[key])
        atom_names = [
            atom_name for atom_name in utils.backbone_atoms
            if atom_name in atoms1 and atom_name in atoms2
        ]
        if not atom_names:
            continue
        pos1 = np.array([atoms1[atom_name] for atom_name in atom_names])
        pos2 = np.array([atoms2[atom_name] for atom_name in atom_names])
        values.append(float(np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))))
    return values


# collect test pdb ids once: the experimental floor is target-mode independent
test_pdb_ids = sorted({
    Path(test_dataset.base.data_list[w_idx]).parent.name
    for w_idx, _ in test_dataset.virtual_entries
})


def _noise_pairs_for_pdb(pdb_id):
    values = []
    try:
        partners = {e for seq in _dna_seqs(pdb_id) for e in _similar_entries(seq)} - {pdb_id}
        for pid in sorted(partners):
            try:
                window_values = _experimental_rmsd_windows(pdb_id, pid)
                if window_values:
                    values.append((pdb_id, pid, window_values))
            except Exception:
                pass
    except Exception:
        pass
    return values


rmsd_values = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(_noise_pairs_for_pdb, pdb_id) for pdb_id in test_pdb_ids]
    for future in as_completed(futures):
        for pdb_id, pid, window_values in future.result():
            rmsd_values.extend(window_values)
            print(
                f'{pdb_id} vs {pid}: median={np.median(window_values):.2f} Å '
                f'n_windows={len(window_values)}'
            )

print(f'\nExperimental window backbone RMSD  median={np.median(rmsd_values):.2f}  mean={np.mean(rmsd_values):.2f}  n={len(rmsd_values)}')

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(rmsd_values, bins=50, color='skyblue', edgecolor='white')
for mode, color in mode_colors.items():
    val = float(wide_scalars_per_mode[mode]['val_rmsd'].dropna().iloc[-1])
    ax.axvline(
        val,
        color=color,
        linewidth=2,
        label=f'model val RMSD [{mode}] {val:.2f} Å'
    )
ax.axvline(
    np.median(rmsd_values),
    color='black', linestyle='--',
    linewidth=1.5,
    label=f'медиана окна экспериментальных структур {np.median(rmsd_values):.2f} Å'
)
ax.set_xlabel('RMSD (Å)', fontsize=13)
ax.set_ylabel('Количество структур', fontsize=13)
ax.legend(fontsize=11)
sns.despine(ax=ax)
fig.tight_layout()
plt.show()
