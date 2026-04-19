# %% [markdown]
# ### Imports

# %%
import os.path as osp
import random
from collections import defaultdict
from glob import glob
from io import StringIO
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import py3Dmol
import seaborn as sns
import torch
from Bio.PDB import Atom, Chain
from Bio.PDB import Model as PDBModel
from Bio.PDB import Residue, Structure
from Bio.PDB.mmcifio import MMCIFIO
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tensorboard.backend.event_processing import event_accumulator

import utils as _utils
from config import PER_MODE
from dataset import PyGDataset
from model import PytorchLightningModule

plt.rcParams['font.family'] = 'DejaVu Sans'

# %% [markdown]
# ### Load model and dataset

# %%
log_dir = osp.join('..', 'logs')
target_modes = PER_MODE.keys()
run_filenames = {}
run_dirs = {}
ckpt_paths = {}
test_dataset_paths = {}
event_files = {}
for target_mode in target_modes:
    run_filenames[target_mode] = osp.join('full_backbone', target_mode, 'baseline')
    run_dirs[target_mode] = osp.join(log_dir, run_filenames[target_mode])
    ckpt_paths[target_mode] = osp.join(run_dirs[target_mode], 'checkpoints', 'last.ckpt')
    test_dataset_paths[target_mode] = osp.join(run_dirs[target_mode], 'test_dataset.pt')
    event_files[target_mode] = glob(osp.join(run_dirs[target_mode], 'events.*'))

test_datasets = {}
for target_mode in target_modes:
    try:
        test_datasets[target_mode] = torch.load(test_dataset_paths[target_mode], weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f'File `{test_dataset_paths[target_mode]}` not found. Ensure training completed successfully.')

models = {}
for target_mode in target_modes:
    try:
        models[target_mode] = PytorchLightningModule.load_from_checkpoint(ckpt_paths[target_mode], map_location='cpu').eval()
    except FileNotFoundError:
        raise FileNotFoundError(f'File `{ckpt_paths[target_mode]}` not found. Ensure you ran train.py.')

# %% [markdown]
# ### Dataset

# %%
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
    marker=dict(size=3, color='blue', opacity=0.5),
    name='Neighbors'
))

# Central atoms
fig.add_trace(go.Scatter3d(
    x=central_atoms_pos[:, 0],
    y=central_atoms_pos[:, 1],
    z=central_atoms_pos[:, 2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=1.0),
    name='Central Nucleotide'
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
        name='Bonds'
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

# %% [markdown]
# ### Model

# %%
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
detail_y = MAX_Y * 0.68
detail_width = MAX_X * 0.22

# Input
draw_box(
    main_x,
    main_y,
    main_width,
    main_height,
    r'Входные данные ($x_t$)',
    r'• $h \in \mathbb{R}^{N \times 27}$ (типы атомов)' + '\n' +
    r'• $x \in \mathbb{R}^{N \times 3}$ (координаты)' + '\n' +
    r'• $\text{матрица смежности} \in \mathbb{Z}^{2 \times E}$ (связи)' + '\n' +
    r'• $t$ (шаг диффузии)',
    'left',
    palette['input']
)
draw_arrow(
    main_x + main_width/2,
    main_y-MAX_Y*0.01,
    main_x + main_width/2,
    main_y-MAX_Y*0.04
)

# Feature augmentation
draw_box(
    main_x,
    main_y-(main_height+arrow_length),
    main_width,
    main_height,
    'Добавление признаков',
    'дублируем $t$ для каждого узла,\nдобавляем метки азотистого\nоснования и спаренность\n' +
    r'итоговые признаки: $\mathbb{R}^{N \times 30}$',
    'left',
    palette['pre']
)
draw_arrow(
    detail_x,
    MAX_Y*0.76,
    main_x + main_width*1.05,
    MAX_Y*0.76
)
draw_box(
    main_x+main_width+MAX_X*0.1,
    main_y-(main_height+arrow_length),
    main_width*1.1,
    main_height,
    'Дополнительные признаки',
    r'• Метка азотистого основания $\in \{0, 1, 2, 3\}^N$' + '\n' +
    r'• Метка спаренности нуклеотида $\in \{0, 1\}^N$',
    'left',
    palette['embed']
)
draw_arrow(
    main_x + main_width/2,
    main_y-(main_height+arrow_length)*2+MAX_Y*0.04,
    main_x + main_width/2,
    main_y-(main_height+arrow_length)*2+MAX_Y*0.04-arrow_length
)

# Embedding
draw_box(
    main_x,
    main_y-(main_height+arrow_length)*2,
    main_width,
    main_height,
    'Входной MLP слой',
    r'линейный слой: $30 \rightarrow 512$' + '\n' +
    'SiLU активация\n' +
    r'координаты: $\mathbb{R}^{N \times 3}$ (неизменны)',
    'left',
    palette['embed']
)
draw_arrow(
    main_x + main_width/2,
    MAX_Y*0.54,
    main_x + main_width/2,
    MAX_Y*0.51
)

# Equivariant blocks
draw_box(
    main_x,
    main_y-(main_height+arrow_length)*3,
    main_width,
    main_height,
    'EGNN',
    'SE(3)-эквивариантная свертка (7x):\n' +
    '• Сообщения по ребрам\n' +
    '• Агрегирование по узлам\n' +
    '• Обновление (нормировка) координат\n' +
    '• Обновление признаков',
    'left',
    palette['equiv']
)
draw_arrow(
    main_x + main_width/2,
    MAX_Y*0.39,
    main_x + main_width/2,
    MAX_Y*0.36
)

# Output embedding
draw_box(
    main_x,
    main_y-(main_height+arrow_length)*4,
    main_width,
    main_height,
    'Выходной MLP слой',
    r'$\text{MLP}(512) \rightarrow 512 \rightarrow 27$' + '\n' +
    'предсказываем шум типов' + '\n' +
    r'и $x_0$ для координат',
    'left',
    palette['output']
)
draw_arrow(
    main_x + main_width/2,
    MAX_Y*0.24,
    main_x + main_width/2,
    MAX_Y*0.21
)

# Output
draw_box(
    main_x,
    main_y-(main_height+arrow_length)*5,
    main_width,
    main_height,
    r'Выходные данные ($x_{t-1}$)',
    r'$h_{\text{out}} \in \mathbb{R}^{N \times 27}$ (логиты типов атомов)' + '\n' +
    r'$x_{\text{out}} \in \mathbb{R}^{N \times 3}$ (предсказанные координаты)',
    'left',
    palette['output']
)

# ========== DETAIL OPERATIONS (right column) ==========
detail_steps = [
    ('Относительные координаты',
     r'$\vec{r}_{ij} = x_i - x_j$' + '\n' + r'$d_{ij} = \|\vec{r}_{ij}\|$'),
    ('Сообщения',
     r'$m_{ij} = \text{MLP}([h_i, h_j, d_{ij}])$'),
    ('Агрегация',
     r'$\bar{m}_i = \sum_j m_{ij}$'),
    ('Обновление координат',
     r'$\Delta x_i = \sum_j m_{ij} \frac{\vec{r}_{ij}}{d_{ij} + \epsilon}$'),
    ('Обновление признаков',
     '$h_i := h_i + MLP([h_i, m_i])+LayerNorm$')
]

detail_palette_steps = ['#FCE4EC', '#F3E5F5', '#E8F5E9', '#FFF8E1', '#E0F2F1']
step_height = 2

for idx, (step_title, step_desc) in enumerate(detail_steps):
    y = detail_y - (idx + 1) * step_height
    draw_box(
        detail_x,
        y,
        detail_width,
        step_height - 0.75,
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


# ============ Lines connecting EGNN block to its details ============
egnn_y_bottom = MAX_Y * 0.4
egnn_height = MAX_Y * 0.1
egnn_x_right = main_x + main_width + MAX_X*0.02
egnn_y_center = egnn_y_bottom + egnn_height/2

rel_coords_y_bottom = detail_y - step_height
rel_coords_height = step_height - 0.75
rel_coords_y_top = rel_coords_y_bottom + rel_coords_height

update_feat_y_bottom = detail_y - 5*step_height

# Line to top block
ax.add_patch(FancyArrowPatch(
    (egnn_x_right, egnn_y_center),
    (detail_x - MAX_X*0.02, rel_coords_y_top),
    linewidth=2,
    arrowstyle='-'  # Draw a line, not an arrow
))

# Line to bottom block
ax.add_patch(FancyArrowPatch(
    (egnn_x_right, egnn_y_center),
    (detail_x - MAX_X*0.02, update_feat_y_bottom),
    linewidth=2,
    arrowstyle='-'  # Draw a line, not an arrow
))


# ============ Diffusion arrow ============
out_block_y = MAX_Y * 0.09
out_block_x_center = main_x + main_width/2

in_block_y = MAX_Y * 0.86
in_block_height = MAX_Y * 0.1
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
    r'$x_{t-1} \rightarrow x_t \text{ (t раз)}$',
    fontsize=15,
    rotation=90,
    ha='center',
    va='center'
)

plt.tight_layout()
diagram_path = osp.join('..', 'data', 'model.png')
plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Check optimizer state

# %%
for target_mode in target_modes:
    print(f'=== target_mode = {target_mode} ===')
    _ck = torch.load(ckpt_paths[target_mode], map_location='cpu', weights_only=False)
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
    print()

# %% [markdown]
# ### Training

# %%


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


# New logging: only train_rmse and val_rmse are tracked, step == epoch
tracked_tags = {'train_rmse', 'val_rmse'}

wide_scalars_per_mode = {}
for target_mode in target_modes:
    dfs = []
    for ef in event_files[target_mode]:
        ea = load_event_accumulator(ef)
        for tag in ea.Tags()['scalars']:
            if tag not in tracked_tags:
                continue
            df = scalars_to_dataframe(ea, tag)
            df['tag'] = tag
            dfs.append(df)

    scalars_by_epoch = pd.concat(dfs, ignore_index=True)[['epoch', 'tag', 'value']] \
        .reset_index(drop=True)
    # Aggregate duplicates per (epoch, tag): keep the last logged value (handles resumed runs / multiple event files).
    wide_scalars_per_mode[target_mode] = scalars_by_epoch.pivot_table(
        index='epoch',
        columns='tag',
        values='value',
        aggfunc='last'
    )

wide_scalars_per_mode


# %%
_mode_list = list(target_modes)
_mode_colors = {'central': 'indigo', 'edge': 'violet'}
_mode_labels = {'central': 'остов центрального нуклеотида', 'edge': 'остов краевого нуклеотида'}
_type_cfg = {
    'train': {'tag': 'train_rmse', 'title': 'Функция потерь во время тренировки',
              'plot_name': 'training_curve.png', 'broken': False},
    'val':   {'tag': 'val_rmse',   'title': 'Функция потерь во время валидации',
              'plot_name': 'validation_curve.png', 'broken': True},
}


for type, cfg in _type_cfg.items():
    if cfg['broken']:
        # Derive two breaks from data: one between each curve's starting value,
        # and one below the lower start down to a fixed visual floor (3 Å).
        starts = sorted(
            [float(wide_scalars_per_mode[m][cfg['tag']].dropna().iloc[0]) for m in _mode_list],
            reverse=True
        )
        start_hi, start_lo = starts[0], starts[1]
        all_vals = np.concatenate([
            wide_scalars_per_mode[m][cfg['tag']].dropna().to_numpy()
            for m in _mode_list
        ])
        y_min = float(np.nanmin(all_vals))

        rel = 0.05  # relative padding around each start
        top_seg = (start_hi * (1 - rel), start_hi * (1 + rel))
        mid_seg = (start_lo * (1 - rel), start_lo * (1 + rel))
        bot_seg = (max(0.0, y_min * 0.9), 3.0)

        # Three stacked subplots sharing x: initial spike of the higher curve,
        # initial spike of the lower curve, and the converged tail below 3 Å.
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True,
            gridspec_kw={'height_ratios': [1, 1, 3], 'hspace': 0.1},
            layout='constrained'
        )

        for ax in (ax_top, ax_mid, ax_bot):
            for target_mode in _mode_list:
                wide_scalars = wide_scalars_per_mode[target_mode]
                ax.plot(
                    wide_scalars.index.to_numpy(),
                    wide_scalars[cfg['tag']].to_numpy(),
                    color=_mode_colors.get(target_mode),
                    linewidth=3,
                    label=_mode_labels.get(target_mode, target_mode),
                )
            ax.tick_params(axis='both', labelsize=15)
            ax.spines['right'].set_visible(False)

        ax_top.set_ylim(*top_seg)
        ax_mid.set_ylim(*mid_seg)
        ax_bot.set_ylim(*bot_seg)

        # Hide spines on the break boundaries.
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_mid.spines['top'].set_visible(False)
        ax_mid.spines['bottom'].set_visible(False)
        ax_mid.tick_params(bottom=False, labelbottom=False)
        ax_bot.spines['top'].set_visible(False)

        # Diagonal break markers at both seams.
        d = 0.5
        break_kw = dict(
            marker=[(-1, -d), (1, d)], markersize=12,
            linestyle='none', color='k', mec='k', mew=1, clip_on=False
        )
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kw)
        ax_mid.plot([0, 1], [1, 1], transform=ax_mid.transAxes, **break_kw)
        ax_mid.plot([0, 1], [0, 0], transform=ax_mid.transAxes, **break_kw)
        ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **break_kw)

        ax_top.set_title(cfg['title'], fontsize=18, fontweight='bold', pad=20)
        ax_bot.set_xlabel('Эпоха', fontsize=18)
        fig.supylabel('RMSE (Å)', fontsize=18)
        ax_top.legend(fontsize=14, loc='upper right')
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.tick_params(axis='both', labelsize=15)
        sns.despine(ax=ax, top=True, right=True)

        for target_mode in _mode_list:
            wide_scalars = wide_scalars_per_mode[target_mode]
            sns.lineplot(
                data=wide_scalars,
                x='epoch',
                y=cfg['tag'],
                color=_mode_colors.get(target_mode),
                linewidth=3,
                label=_mode_labels.get(target_mode, target_mode),
                ax=ax
            )

        ax.set_title(cfg['title'], fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Эпоха', fontsize=18)
        ax.set_ylabel('RMSE (Å)', fontsize=18)
        ax.legend(fontsize=14)

        fig.tight_layout()

    for target_mode in _mode_list:
        fig.savefig(
            osp.join(run_dirs[target_mode], cfg['plot_name']),
            bbox_inches='tight',
            dpi=300
        )

    plt.show()

# %% [markdown]
# ### Results

# %%
# Cache per-mode test path layout so downstream cells can reuse it cheaply.
test_paths_per_mode: dict[str, list[str]] = {}
test_pdb_to_local_per_mode: dict[str, dict[str, list[int]]] = {}
for target_mode in target_modes:
    test_paths_per_mode[target_mode] = [
        test_datasets[target_mode].dataset.data_list[i]
        for i in test_datasets[target_mode].indices
    ]
    pdb_to_local: dict[str, list[int]] = defaultdict(list)
    for local_i, p in enumerate(test_paths_per_mode[target_mode]):
        pdb_to_local[Path(p).parent.name].append(local_i)
    test_pdb_to_local_per_mode[target_mode] = pdb_to_local

for target_mode in target_modes:
    model = models[target_mode]
    test_dataset = test_datasets[target_mode]
    test_paths = test_paths_per_mode[target_mode]
    test_pdb_to_local = test_pdb_to_local_per_mode[target_mode]

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
    # Mirror the model's internal target mask so indexing of pred_backbone stays consistent
    # (e.g. for target_mode='central' only the central backbone atoms are denoised).
    target_mask = model._target_mask(data).cpu().numpy().astype(bool)

    edge_index = data.edge_index.cpu().numpy() if data.edge_index is not None else None
    true_backbone = pos_full[target_mask]

    # Run generation for all backbone atoms in the window.
    with torch.no_grad():
        pred_backbone_raw = model.sample(data).cpu().numpy()

    # Apply translation-only alignment for visualization to remove possible global drift.
    # The model predicts coordinates, while this correction only matches global centroid.
    centroid_shift = true_backbone.mean(axis=0) - pred_backbone_raw.mean(axis=0)
    pred_backbone = pred_backbone_raw + centroid_shift

    # In this dataset, positions are already in the true central nucleotide frame:
    # pos_local = (pos_global - origin_true) @ ref_frame_true.
    origin_true = data.origin.squeeze(0).cpu().numpy() if hasattr(data, 'origin') else None
    ref_frame_true = data.ref_frame.squeeze(0).cpu().numpy() if hasattr(data, 'ref_frame') else None

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
            f'Results: run={run_filenames[target_mode]}, '
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

# %% [markdown]
# ### Inference

# %%
# Map atom vocabulary index back to its canonical atom name / chemical element so
# that the resulting CIF files carry meaningful atom metadata.
_idx_to_atom_name = {v: k for k, v in _utils.atom_to_idx.items()}


def _atom_element(atom_name: str) -> str:
    # Element is the first alphabetic character, except standalone phosphorus.
    return 'P' if atom_name == 'P' else atom_name[0]


def graph_to_cif_string(atom_types, pos, structure_name='structure'):
    structure = Structure.Structure(structure_name)
    model = PDBModel.Model(0)
    chain = Chain.Chain('A')
    residue = Residue.Residue((' ', 1, ' '), 'UNK', ' ')

    for i, (atom_type_idx, coord) in enumerate(zip(atom_types, pos)):
        atom_name = _idx_to_atom_name.get(int(atom_type_idx), 'X')
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


for target_mode in target_modes:
    model = models[target_mode]
    test_dataset = test_datasets[target_mode]
    test_paths = test_paths_per_mode[target_mode]
    test_pdb_to_local = test_pdb_to_local_per_mode[target_mode]

    inference_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
    inference_local_indices = sorted(
        test_pdb_to_local[inference_pdb_id],
        key=lambda i: int(Path(test_paths[i]).stem)
    )
    print(f'[{target_mode}] Reconstructing backbone for structure {inference_pdb_id}: '
          f'{len(inference_local_indices)} windows')

    all_generated_pos: list[torch.Tensor] = []
    all_true_pos: list[torch.Tensor] = []
    all_atom_types: list[torch.Tensor] = []
    for local_i in inference_local_indices:
        window = test_dataset[local_i].clone()
        mask = model._target_mask(window)

        with torch.no_grad():
            gen_local = model.sample(window)

        # Invert the dataset's local transform pos_local = (pos_global - origin) @ R
        # via pos_global = pos_local @ R.T + origin to recover global coordinates
        ref_frame = window.ref_frame.squeeze(0).float()
        origin = window.origin.squeeze(0).float()
        gen_global = gen_local.float() @ ref_frame.T + origin
        true_global = window.pos[mask].float() @ ref_frame.T + origin

        all_generated_pos.append(gen_global)
        all_true_pos.append(true_global)
        all_atom_types.append(torch.argmax(window.x[mask], dim=1))

    generated_pos = torch.cat(all_generated_pos, dim=0).cpu().numpy()
    original_pos = torch.cat(all_true_pos, dim=0).cpu().numpy()
    atom_types = torch.cat(all_atom_types, dim=0).cpu().numpy()

    generated_cif_data = graph_to_cif_string(atom_types, generated_pos, 'generated_structure')
    original_cif_data = graph_to_cif_string(atom_types, original_pos, 'original_structure')

    view = py3Dmol.view(width=800, height=400, linked=False, viewergrid=(1, 2))

    view.addModel(generated_cif_data, 'cif', viewer=(0, 0))
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}}, viewer=(0, 0))
    view.addLabel(
        f'Generated backbone [{target_mode}] ({inference_pdb_id})',
        {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
        viewer=(0, 0)
    )

    view.addModel(original_cif_data, 'cif', viewer=(0, 1))
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}}, viewer=(0, 1))
    view.addLabel(
        f'Original backbone [{target_mode}] ({inference_pdb_id})',
        {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
        viewer=(0, 1)
    )

    view.zoomTo()
    view.show()
