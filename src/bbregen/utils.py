import os
import os.path as osp
import tempfile
import warnings
from collections import defaultdict
from glob import glob

import gemmi
import MDAnalysis as mda
import numpy as np
import requests
import torch
import torch.nn.functional as F
from MDAnalysis.exceptions import SelectionError
from torch_geometric.data import Data

from pynamod import CG_Structure  # pyright: ignore[reportAttributeAccessIssue]
from pynamod.atomic_analysis.nucleotides_parser import get_base_u  # pyright: ignore[reportMissingImports]

from .torsion_geometry import nucleotide_torsions_numpy, wrap_angle_rad

backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", 'OP1', 'OP2', 'P', "O3'", "O4'", "O5'"]
nucleic_acid_atoms = ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'O2', 'O4', 'O6']
nucleotide_atoms = nucleic_acid_atoms + backbone_atoms
atom_to_idx = {atom: i for i, atom in enumerate(nucleotide_atoms)}
base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

CHAIN_END_CLASS_INTERNAL = 0
CHAIN_END_CLASS_5_PRIME = 1
CHAIN_END_CLASS_3_PRIME = 2
N_CHAIN_END_CLASSES = 3

PBAR_COLOR = '#B366FF'


def _load_mda_universe_from_pdb_file(pdb_path):
    """Open a PDB file with MDAnalysis and silence noisy element warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Unknown element .+ found for some atoms\.',
            category=UserWarning,
            module=r'MDAnalysis\.topology\.PDBParser',
        )
        u = mda.Universe(pdb_path)
    u.guess_TopologyAttrs(context='default', to_guess=['elements'])
    return u


def mmcif_to_mda_universe(path):
    """Parse an mmCIF file using gemmi and return an MDAnalysis Universe."""
    st = gemmi.read_structure(path)
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    tmp_pdb = tmp.name
    tmp.close()
    try:
        pdb_opts = gemmi.PdbWriteOptions()
        pdb_opts.cryst1_record = False
        st.write_pdb(tmp_pdb, pdb_opts)
        u = _load_mda_universe_from_pdb_file(tmp_pdb)
    finally:
        os.unlink(tmp_pdb)
    return u


def find_best_checkpoint(run_dir):
    """Locate the best-monitor checkpoint of a run via ModelCheckpoint state."""
    ckpt_dir = osp.join(run_dir, 'checkpoints')
    candidates = sorted(glob(osp.join(ckpt_dir, '*.ckpt')))
    if not candidates:
        raise FileNotFoundError(f'No *.ckpt files in {ckpt_dir}.')

    last_path = osp.join(ckpt_dir, 'last.ckpt')
    source = last_path if last_path in candidates else candidates[0]

    state = torch.load(source, map_location='cpu', weights_only=False)
    for key, cb_state in state.get('callbacks', {}).items():
        if 'ModelCheckpoint' not in key:
            continue
        best_path = cb_state.get('best_model_path', '')
        if not best_path:
            continue
        local = osp.join(ckpt_dir, osp.basename(best_path))
        if osp.isfile(local):
            return local
        if osp.isfile(best_path):
            return best_path

    raise RuntimeError(f'best_model_path not present in ModelCheckpoint state of {source}.')


def get_pdb_ids():
    """Get PDB IDs from the RCSB PDB API with resolution ≤3 Å and DNA (≥3 residues)."""
    query = {
        'query': {
            'type': 'group',
            'logical_operator': 'and',
            'nodes': [
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'rcsb_entry_info.resolution_combined',
                        'operator': 'less_or_equal',
                        'negation': False,
                        'value': 3,
                    },
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_entity_polymer_type',
                        'operator': 'exact_match',
                        'negation': False,
                        'value': 'DNA',
                    },
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_sample_sequence_length',
                        'operator': 'greater_or_equal',
                        'negation': False,
                        'value': 3,
                    },
                },
                {
                    'type': 'group',
                    'nodes': [
                        {
                            'type': 'terminal',
                            'service': 'text',
                            'parameters': {
                                'attribute': 'exptl.method',
                                'operator': 'exact_match',
                                'negation': False,
                                'value': 'X-RAY DIFFRACTION',
                            },
                        },
                        {
                            'type': 'terminal',
                            'service': 'text',
                            'parameters': {
                                'attribute': 'exptl.method',
                                'operator': 'exact_match',
                                'negation': False,
                                'value': 'ELECTRON MICROSCOPY',
                            },
                        },
                    ],
                    'logical_operator': 'or',
                },
            ],
            'label': 'text',
        },
        'return_type': 'entry',
        'request_options': {
            'paginate': {'start': 0, 'rows': 10000},
            'results_content_type': ['experimental'],
            'sort': [{'sort_by': 'score', 'direction': 'desc'}],
            'scoring_strategy': 'combined',
        },
    }

    response = requests.post(
        'https://search.rcsb.org/rcsbsearch/v2/query',
        json=query,
        headers={'Content-Type': 'application/json'},
    )
    response.raise_for_status()
    data = response.json()
    return [item['identifier'] for item in data.get('result_set', [])]


def resolve_run_dir(run):
    """Map a user-facing experiment id (e.g. 'fixed_swa/baseline') to its log directory."""
    if osp.isabs(run):
        return run
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'logs')
    run_norm = osp.normpath(run)
    if run_norm.split(os.sep, 1)[0] == 'logs':
        run_norm = run_norm.split(os.sep, 1)[1] if os.sep in run_norm else ''
    return osp.normpath(osp.join(log_dir, run_norm))


def has_pair(structure, nucleotide):
    """Check if a nucleotide has a pair."""
    lead_idxs = structure.dna.pairs_list.lead_nucl_inds  # type: ignore
    lag_idxs = structure.dna.pairs_list.lag_nucl_inds  # type: ignore
    return nucleotide.ind in lead_idxs + lag_idxs


_ATOM_RENAMES = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}


def rename_atom(name):
    return _ATOM_RENAMES.get(name, name.rstrip('AB'))


def _is_heavy_atom(atom):
    return 'H' not in atom.name and getattr(atom, 'element', None) not in {'H', 'D'}


def default_atoms_provider(nucleotide):
    return [(rename_atom(a.name), a.position) for a in nucleotide.e_residue if _is_heavy_atom(a)]


def inference_atoms_provider(nucleotide):
    """Canonical heavy atoms for a nucleotide with zero-filled missing positions."""
    exp_positions = dict(default_atoms_provider(nucleotide))
    atoms = []
    for atom in get_base_u(nucleotide.restype):  # type: ignore
        if not _is_heavy_atom(atom):
            continue
        atom_name = rename_atom(atom.name)
        atoms.append((atom_name, exp_positions.get(atom_name, np.zeros(3, dtype=np.float32))))
    return atoms


def _heavy_xyz_dict(nucleotide):
    return {
        rename_atom(a.name): np.asarray(a.position, dtype=np.float64)
        for a in nucleotide.e_residue
        if _is_heavy_atom(a)
    }


def _neighbor_xyz_for_torsions(window, i):
    """Heavy-atom xyz for current, previous, and next nucleotides."""
    cur = window[i]
    xyz_cur = _heavy_xyz_dict(cur)
    if i > 0:
        xyz_prev = _heavy_xyz_dict(window[i - 1])
    else:
        prev_nt = cur.previous_nucleotide
        xyz_prev = _heavy_xyz_dict(prev_nt) if prev_nt is not None else {}
    if i < len(window) - 1:
        xyz_next = _heavy_xyz_dict(window[i + 1])
    else:
        next_nt = cur.next_nucleotide
        xyz_next = _heavy_xyz_dict(next_nt) if next_nt is not None else {}
    return xyz_cur, xyz_prev, xyz_next


def _build_window_data(window, window_idx, chain_len, chain_direction, structure, window_size):
    """Build the PyG window payload used by inference and training."""
    ref_frames_all = getattr(structure.dna.nucleotides, 'ref_frames')
    origins_all = getattr(structure.dna.nucleotides, 'origins')

    nt_origins = []
    nt_frames = []
    torsion_rows = []
    mask_rows = []
    has_pair_nt = []
    chain_end_class_list = []
    base_type_idx = []
    central_nt_mask = []
    is_chain_edge_nt = []

    tau_m_list: list[float] = []
    tau_m_mask_list: list[bool] = []
    pair_origins: list[torch.Tensor] = []
    pair_frames: list[torch.Tensor] = []

    pairs = structure.dna.pairs_list
    pair_map: dict[int, int] = {
        **{lead: lag for lead, lag in zip(pairs.lead_nucl_inds, pairs.lag_nucl_inds)},
        **{lag: lead for lead, lag in zip(pairs.lead_nucl_inds, pairs.lag_nucl_inds)},
    }

    for nucleotide_idx, nucleotide in enumerate(window):
        base_letter = nucleotide.restype
        base_type_idx.append(base_to_idx[base_letter])
        nt_has_pair = nucleotide.ind in pair_map
        position_in_chain = window_idx + nucleotide_idx
        if position_in_chain == 0:
            chain_end_class = CHAIN_END_CLASS_5_PRIME if chain_direction >= 0 else CHAIN_END_CLASS_3_PRIME
        elif position_in_chain == chain_len - 1:
            chain_end_class = CHAIN_END_CLASS_3_PRIME if chain_direction >= 0 else CHAIN_END_CLASS_5_PRIME
        else:
            chain_end_class = CHAIN_END_CLASS_INTERNAL

        is_chain_edge_nt.append(chain_end_class != CHAIN_END_CLASS_INTERNAL)
        has_pair_nt.append(nt_has_pair)
        chain_end_class_list.append(chain_end_class)
        central_nt_mask.append(nucleotide_idx == window_size // 2)

        xyz_cur, xyz_prev, xyz_next = _neighbor_xyz_for_torsions(window, nucleotide_idx)
        t_av, m_av, tau_mv, tau_m_ok = nucleotide_torsions_numpy(xyz_cur, xyz_prev, xyz_next, base_letter)
        t_av = np.asarray([wrap_angle_rad(x) for x in t_av], dtype=np.float64)
        torsion_rows.append(t_av)
        mask_rows.append(m_av)
        tau_m_list.append(float(tau_mv))
        tau_m_mask_list.append(bool(tau_m_ok))

        nt_origins.append(origins_all[nucleotide.ind].float().view(3))
        nt_frames.append(ref_frames_all[nucleotide.ind].float())

        if nucleotide.ind in pair_map:
            partner_ind = pair_map[nucleotide.ind]
            pair_origins.append(origins_all[partner_ind].float().view(3))
            pair_frames.append(ref_frames_all[partner_ind].float())
        else:
            pair_origins.append(torch.zeros(3, dtype=torch.float32))
            pair_frames.append(torch.zeros(3, 3, dtype=torch.float32))

    torsions = torch.tensor(np.stack(torsion_rows, axis=0), dtype=torch.float32)
    torsion_mask = torch.tensor(np.stack(mask_rows, axis=0), dtype=torch.bool)
    nt_origins_world = torch.stack(nt_origins, dim=0)
    nt_frames_world = torch.stack(nt_frames, dim=0)
    pair_origins_world = torch.stack(pair_origins, dim=0)
    pair_frames_world = torch.stack(pair_frames, dim=0)

    chain_end_class_long = torch.tensor(chain_end_class_list, dtype=torch.long)
    chain_end_class_tensor = F.one_hot(chain_end_class_long, num_classes=N_CHAIN_END_CLASSES).float()
    base_types_tensor = F.one_hot(
        torch.tensor(base_type_idx, dtype=torch.long),
        num_classes=len(base_to_idx),
    ).float()
    has_pair_tensor = torch.tensor(has_pair_nt, dtype=torch.bool)
    central_nt_mask_t = torch.tensor(central_nt_mask, dtype=torch.bool)
    is_chain_edge_nt_t = torch.tensor(is_chain_edge_nt, dtype=torch.bool)
    touches_chain_edge = bool(is_chain_edge_nt_t.any().item())
    target_nt_idx = torch.tensor(window_size // 2, dtype=torch.long)

    n_bb = len(backbone_atoms)
    bb_world = torch.full((window_size, n_bb, 3), float('nan'), dtype=torch.float32)
    for i, nucleotide in enumerate(window):
        exp = dict(default_atoms_provider(nucleotide))
        for j, atom_name in enumerate(backbone_atoms):
            if atom_name in exp:
                bb_world[i, j] = torch.tensor(np.asarray(exp[atom_name], dtype=np.float32))

    j_o3 = backbone_atoms.index("O3'")
    o3_prev_local_rows = []
    o3_prev_valid_rows = []
    for k in range(window_size):
        if k == 0:
            o3_prev_local_rows.append(torch.zeros(3, dtype=torch.float32))
            o3_prev_valid_rows.append(False)
        else:
            o3w = bb_world[k - 1, j_o3]
            if torch.isnan(o3w).any():
                o3_prev_local_rows.append(torch.zeros(3, dtype=torch.float32))
                o3_prev_valid_rows.append(False)
            else:
                ok = nt_origins[k]
                rk = nt_frames[k]
                o3_local = (o3w - ok) @ rk
                o3_prev_local_rows.append(o3_local.float())
                o3_prev_valid_rows.append(True)

    o3_prev_local_t = torch.stack(o3_prev_local_rows, dim=0)
    o3_prev_valid_t = torch.tensor(o3_prev_valid_rows, dtype=torch.bool)

    return Data(
        nt_origins_world=nt_origins_world,
        nt_frames_world=nt_frames_world,
        pair_origins_world=pair_origins_world,
        pair_frames_world=pair_frames_world,
        torsions=torsions,
        torsion_mask=torsion_mask,
        tau_m=torch.tensor(tau_m_list, dtype=torch.float32),
        tau_m_mask=torch.tensor(tau_m_mask_list, dtype=torch.bool),
        base_types=base_types_tensor,
        has_pair_nt=has_pair_tensor,
        chain_end_class=chain_end_class_tensor,
        central_nt_mask=central_nt_mask_t,
        is_chain_edge_nt=is_chain_edge_nt_t,
        touches_chain_edge=torch.tensor(touches_chain_edge, dtype=torch.bool),
        bb_xyz_world=bb_world,
        o3_prev_local=o3_prev_local_t,
        o3_prev_valid=o3_prev_valid_t,
        target_nt_idx=target_nt_idx,
    )


def parse_dna(path, use_full_nucleotide, window_size=3):
    """Parse a DNA structure and build sliding-window records."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdb':
        universe = _load_mda_universe_from_pdb_file(path)
    elif ext in ('.cif', '.mmcif'):
        universe = mmcif_to_mda_universe(path)
    else:
        raise ValueError(f'Unsupported input format: {ext!r} (expected .pdb/.cif/.mmcif)')

    structure = CG_Structure(mdaUniverse=universe)
    nucleic_atoms = structure.u.select_atoms('nucleic')  # type: ignore
    if len(nucleic_atoms) == 0:
        raise RuntimeError(f'No nucleic atoms found in {path}.')
    dna_segids = list(np.unique(nucleic_atoms.segids))
    structure.analyze_dna(leading_strands=dna_segids, use_full_nucleotide=use_full_nucleotide)

    nucleotides_by_chain = defaultdict(list)
    for segid, nucleotide in zip(
        structure.dna.nucleotides.segids,  # type: ignore
        structure.dna.nucleotides,
    ):
        atom_group = getattr(nucleotide, 'e_residue').atoms
        atom0 = atom_group[0] if len(atom_group) > 0 else None
        chain_key = ''
        if atom0 is not None:
            chain_key = getattr(atom0, 'chainID', '') or getattr(atom0, 'segid', '')
        if not chain_key:
            chain_key = segid
        if not chain_key:
            continue
        nucleotides_by_chain[chain_key].append(nucleotide)

    chain_records = []
    for chain_key, chain in nucleotides_by_chain.items():
        windows: list = []
        if len(chain) >= window_size:
            first_resid = chain[0].e_residue.resids[0]
            last_resid = chain[-1].e_residue.resids[0]
            chain_direction = 1 if last_resid >= first_resid else -1
            try:
                for window_idx in range(len(chain) - window_size + 1):
                    window = chain[window_idx: window_idx + window_size]
                    resids = [n.e_residue.resids[0] for n in window]
                    steps = [resids[i + 1] - resids[i] for i in range(len(resids) - 1)]
                    if not (all(s == 1 for s in steps) or all(s == -1 for s in steps)):
                        continue

                    data = _build_window_data(
                        window,
                        window_idx,
                        len(chain),
                        chain_direction,
                        structure,
                        window_size,
                    )
                    windows.append((window, window_idx, data))
            except (KeyError, SelectionError):
                pass
        chain_records.append((chain_key, chain, windows))

    return structure, chain_records
