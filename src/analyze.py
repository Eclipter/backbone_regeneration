# %%
# Imports
from dataset import DNADataModule
from sklearn.neighbors import NearestNeighbors
import os.path as osp
import random
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import py3Dmol
import requests
import seaborn as sns
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from matplotlib.transforms import Bbox
from torch_geometric.data import Batch, Data
from tqdm import tqdm

import utils
from dataset import PyGDataset
from model import PytorchLightningModule
from predict import predict_backbone, write_structure
from torsion_geometry import N_TORSIONS, build_backbone_from_torsions

TOR_NAMES = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'χ', 'P']

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
# 3D window visualization (backbone atoms in world coords, target nucleotide highlighted)
dataset = PyGDataset()
rng = np.random.default_rng()
raw_idx = rng.integers(len(dataset))
raw_data = cast(Data, dataset[raw_idx])
pdb_id = Path(dataset.data_list[raw_idx]).parent.name

ws = raw_data.bb_xyz_world.shape[0]
tidx_raw = int(raw_data.target_nt_idx.item())
nt_colors = ['#7B7BFF', '#FF4444', '#7BFF7B']  # left / central / right (ws=3)
nt_labels = ['5\' neighbour', 'central', '3\' neighbour']

fig = go.Figure()

# backbone atoms per nucleotide
for i in range(ws):
    bb = raw_data.bb_xyz_world[i].numpy()  # [n_bb, 3]
    valid = ~np.any(np.isnan(bb), axis=1)
    pts = bb[valid]
    names = [utils.backbone_atoms[j] for j in range(len(utils.backbone_atoms)) if valid[j]]
    color = '#FF4444' if i == tidx_raw else nt_colors[i % len(nt_colors)]
    size = 10 if i == tidx_raw else 6
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=size, color=color, opacity=0.9),
        text=names,
        name=f'nt {i} ({nt_labels[i] if i < len(nt_labels) else ""}) backbone',
    ))

# nucleotide origins
origins = raw_data.nt_origins_world.numpy()
fig.add_trace(go.Scatter3d(
    x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
    mode='markers+text',
    marker=dict(size=14, color='gold', opacity=1.0, symbol='diamond'),
    text=[f'nt {i}' for i in range(ws)],
    textposition='top center',
    name='nucleotide origins',
))

# target frame axes
o_t = raw_data.nt_origins_world[tidx_raw].numpy()
R_t = raw_data.nt_frames_world[tidx_raw].numpy()
ax_len = 3.0
for ax_i, ax_color, ax_name in [(0, 'red', 'X'), (1, 'green', 'Y'), (2, 'blue', 'Z')]:
    end = o_t + ax_len * R_t[:, ax_i]
    fig.add_trace(go.Scatter3d(
        x=[o_t[0], end[0]], y=[o_t[1], end[1]], z=[o_t[2], end[2]],
        mode='lines',
        line=dict(color=ax_color, width=5),
        showlegend=False,
        name=f'target frame {ax_name}',
    ))

fig.update_layout(
    title=f'PDB: {pdb_id} — backbone atoms (target nt = {tidx_raw})',
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
    ),
    width=800, height=700,
    margin=dict(r=0, l=0, b=0, t=40),
)
fig.show()

# %%
# Model architecture overview
plt.rcParams['font.family'] = 'DejaVu Sans'
hp = model.hparams
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

rows = [
    ['Input per nucleotide', f'rel_origin (3) + rel_frame (9) + base_type one-hot (4)\n'
     f'+ has_pair (1) + chain_end_class one-hot (3) + is_target_nt (1) = 21 dims\n'
     f'Target nt additionally: noisy sin/cos latents ({N_TORSIONS * 2}) + time emb (32)'
     f' + torsion_mask ({N_TORSIONS}) + self-cond latent ({N_TORSIONS * 2})'],
    ['node_dim', str(model.node_dim)],
    ['Architecture', f'in_MLP → TransformerEncoder ({hp["num_layers"]}×, '
     f'd_model={hp["hidden_dim"]}, heads={hp["num_heads"]}) → Linear'],
    ['Output', f'ε̂ ∈ ℝ^{N_TORSIONS * 2} (noise on sin/cos latents per target nt)'],
    ['Diffusion', f'DDPM in ℝ^{N_TORSIONS * 2} (latent per torsion), '
     f'{hp["num_timesteps"]} steps, {hp["beta_schedule"]}'],
    ['Train loss', 'MSE(ε̂, ε); mask expanded ×2 along latent; 50% self-conditioning'],
    ['Val metric', 'Backbone RMSD (Å) in local frame; central vs chain-edge targets'],
]

table = ax.table(
    cellText=rows,
    colLabels=['Component', 'Details'],
    cellLoc='left',
    loc='center',
    bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0),
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#6A4CA0')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 0:
        cell.set_facecolor('#EDE4F8')
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('#F8F4FF')
    cell.set_edgecolor('#CCBBEE')
    cell.PAD = 0.05

fig.tight_layout()
fig.savefig(osp.join(run_dir, 'model_arch.png'), dpi=200, bbox_inches='tight')
plt.show()

# %%
# Check optimizer state
_ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('epoch:', _ck.get('epoch'), '| global_step:', _ck.get('global_step'))
for i, _opt in enumerate(_ck.get('optimizer_states', [])):
    print(f'\noptimizer_states[{i}]')
    for gi, g in enumerate(_opt['param_groups']):
        _pg = {k: v for k, v in g.items() if k != 'params'}
        _pg['n_param_ids'] = len(g['params'])
        print(f'  param_group[{gi}]:', _pg)
for i, sch in enumerate(_ck.get('lr_schedulers') or []):
    print(f'\nlr_schedulers[{i}]:', {
        k: v for k, v in sch.items()
        if k in ('_last_lr', 'last_epoch', 'best', 'num_bad_epochs', 'cooldown_counter', 'patience')
    })

# %%
# Training curves
from tensorboard.backend.event_processing import \
    event_accumulator  # noqa: E402


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
    'train_loss':           ('all',     'train_loss'),
    'val_rmsd':             ('all',     'val_rmsd'),
    'val_rmsd_central':     ('central', 'val_rmsd'),
    'val_rmsd_edge':        ('edge',    'val_rmsd'),
    'test_rmsd':            ('all',     'test_rmsd'),
    'test_rmsd_central':    ('central', 'test_rmsd'),
    'test_rmsd_edge':       ('edge',    'test_rmsd'),
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

scalars = pd.concat(dfs, ignore_index=True)
wide_per_mode = {
    mode: scalars.loc[scalars['mode'] == mode].pivot_table(
        index='epoch', columns='metric', values='value', aggfunc='last'
    )
    for mode in target_modes
}
wide = wide_per_mode['all']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Train loss (MSE on noise)
ax = axes[0]
if 'train_loss' in wide.columns:
    vals = wide['train_loss'].dropna()
    ax.plot(vals.index.to_numpy(), vals.to_numpy(), color='indigo', linewidth=2)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Train loss (MSE)', fontsize=13)
sns.despine(ax=ax)

# Val RMSD (Å) per target class
ax = axes[1]
for mode in target_modes:
    w = wide_per_mode[mode]
    if 'val_rmsd' in w.columns:
        vals = w['val_rmsd'].dropna()
        ax.plot(vals.index.to_numpy(), vals.to_numpy(),
                color=mode_colors[mode], linestyle=mode_linestyles[mode],
                linewidth=2, label=mode)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Val RMSD (Å)', fontsize=13)
ax.legend(fontsize=11)
sns.despine(ax=ax)

fig.tight_layout()
fig.savefig(osp.join(run_dir, 'training_curves.png'), dpi=200, bbox_inches='tight')
plt.show()

# %%
# Torsion prediction: GT vs sampled (one random test window)
test_pdb_to_local: dict[str, list[int]] = defaultdict(list)
_base_paths = test_dataset.base.data_list
for local_i, (w_idx, _) in enumerate(test_dataset.virtual_entries):
    pdb = Path(_base_paths[w_idx]).parent.name
    test_pdb_to_local[pdb].append(local_i)

sample_local_i = rng.integers(len(test_dataset))
data = cast(Data, test_dataset[sample_local_i].clone())
tidx = int(data.target_nt_idx.item())

with torch.no_grad():
    batch = cast(Any, Batch.from_data_list([data])).to(device)
    gt_w, decoded = model.p_sample_loop(batch)
    pred_theta, _pred_tau_m = decoded

gt = gt_w[0].cpu().numpy()
pred = pred_theta[0].cpu().numpy()
mask = data.torsion_mask[tidx].numpy()  # [N_TORSIONS]

valid_idx = [i for i in range(N_TORSIONS) if mask[i]]
gt_valid = [gt[i] for i in valid_idx]
pred_valid = [pred[i] for i in valid_idx]
names_valid = [TOR_NAMES[i] for i in valid_idx]

x = np.arange(len(valid_idx))
w = 0.35
fig, ax = plt.subplots(figsize=(max(6, len(valid_idx) * 1.2), 4))
ax.bar(x - w / 2, gt_valid, w, label='GT', color='steelblue')
ax.bar(x + w / 2, pred_valid, w, label='Predicted', color='tomato')
ax.set_xticks(x)
ax.set_xticklabels(names_valid, fontsize=12)
ax.set_ylabel('Angle (rad)', fontsize=12)
ax.set_title('Target nucleotide: GT vs predicted torsions', fontsize=13)
ax.axhline(0, color='black', linewidth=0.8)
ax.legend()
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(osp.join(run_dir, 'torsion_prediction.png'), dpi=200, bbox_inches='tight')
plt.show()

cos_errs = [float(1.0 - np.cos(pred[i] - gt[i])) for i in valid_idx]
print('Per-torsion 1−cos error:')
for name, err in zip(names_valid, cos_errs):
    print(f'  {name}: {err:.4f}')

# %%
# Full-chain inference (one test structure)
inference_pdb_id = random.choice(sorted(test_pdb_to_local.keys()))
raw_inference_path = osp.join('..', 'data', 'raw', f'{inference_pdb_id}.cif')
print(f'Inferring backbone for {inference_pdb_id}…')

generated_pdb_path = osp.join(run_dir, f'generated_backbone_{inference_pdb_id}.pdb')
predictions, chain_records = predict_backbone(raw_inference_path, ckpt_path, device=device)
write_structure(chain_records, predictions, generated_pdb_path)
print(f'Wrote {generated_pdb_path}  ({len(predictions)} backbone atoms)')

with open(generated_pdb_path) as f:
    pdb_str = f.read()

view = py3Dmol.view(width=700, height=400)
view.addModel(pdb_str, 'pdb')
view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}})
view.addLabel(
    f'Generated backbone ({inference_pdb_id})',
    {'fontColor': 'black', 'backgroundColor': 'lightgray', 'backgroundOpacity': 0.8},
)
view.zoomTo()
view.show()

# %%
# Dataset noise floor (RMSD between identical-sequence structures in local nucleotide frame)


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
    print(f'\nExperimental floor  median={np.median(rmsd_values):.2f}  '
          f'mean={np.mean(rmsd_values):.2f}  n={len(rmsd_values)}')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rmsd_values, bins=50, color='skyblue', edgecolor='white')
    if 'val_rmsd' in wide.columns:
        val = float(wide['val_rmsd'].dropna().iloc[-1])
        ax.axvline(val, color='indigo', linewidth=2, label=f'val RMSD: {val:.2f} Å')
    ax.axvline(np.median(rmsd_values), color='black', linestyle='--', linewidth=1.5,
               label=f'experimental median: {np.median(rmsd_values):.2f} Å')
    ax.set_xlabel('Backbone RMSD in local frame (Å)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
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
    for _wi in tqdm(_inter_run_indices, desc='Inter-run RMSD'):
        data = cast(Any, test_dataset[_wi].clone())
        tidx = int(data.target_nt_idx.item())
        restype = _idx_to_base[int(data.base_types[tidx].argmax().item())]

        batch = cast(Any, Batch.from_data_list([data])).to(device)

        run_coords: list[dict] = []
        for _ in range(K):
            pred_theta, pred_tau_m = model.sample(batch)
            torsions_np = pred_theta[0].cpu().numpy()
            tau_m_val = float(pred_tau_m[0].clamp(min=1e-3).item())
            bb = build_backbone_from_torsions(
                torsions_np, restype, o3_prev_local=None, tau_m=tau_m_val,
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
        label='pairwise inter-run RMSD')
ax.axvline(_med, color='black', linestyle='--', linewidth=1.8,
           label=f'median: {_med:.3f} Å')
if rmsd_values:
    _noise_med = float(np.median(rmsd_values))
    ax.axvline(_noise_med, color='steelblue', linestyle=':', linewidth=1.8,
               label=f'exp. noise floor: {_noise_med:.3f} Å')
ax.set_xlabel('Backbone RMSD between runs (Å)', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
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

for _i in tqdm(range(len(train_dataset)), desc='kNN: building train index'):
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

for _i in tqdm(range(len(test_dataset)), desc='kNN: evaluating test'):
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
