"""DNA parsing and sliding-window materialization for training and inference."""

import os
from collections import defaultdict

import numpy as np
import requests
import torch
import torch.nn.functional as F
from MDAnalysis.exceptions import SelectionError
from torch_geometric.data import Data

from pynamod import CG_Structure  # pyright: ignore[reportAttributeAccessIssue]

from ..io import (
    default_atoms_provider,
    heavy_xyz_dict,
    load_mmcif_universe,
    load_pdb_universe,
)
from ..geometry import nucleotide_torsions_numpy, wrap_angle_rad
from .vocab import (
    BACKBONE_ATOMS,
    BASE_TO_INDEX,
    CHAIN_END_CLASS_3_PRIME,
    CHAIN_END_CLASS_5_PRIME,
    CHAIN_END_CLASS_INTERNAL,
    N_CHAIN_END_CLASSES,
)


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


def _neighbor_xyz_for_torsions(window, nucleotide_idx):
    """Heavy-atom xyz for current, previous, and next nucleotides."""
    current = window[nucleotide_idx]
    xyz_cur = heavy_xyz_dict(current)
    if nucleotide_idx > 0:
        xyz_prev = heavy_xyz_dict(window[nucleotide_idx - 1])
    else:
        previous_nt = current.previous_nucleotide
        xyz_prev = heavy_xyz_dict(previous_nt) if previous_nt is not None else {}
    if nucleotide_idx < len(window) - 1:
        xyz_next = heavy_xyz_dict(window[nucleotide_idx + 1])
    else:
        next_nt = current.next_nucleotide
        xyz_next = heavy_xyz_dict(next_nt) if next_nt is not None else {}
    return xyz_cur, xyz_prev, xyz_next


def build_window_data(window, window_idx, chain_len, chain_direction, structure, window_size):
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
        base_type_idx.append(BASE_TO_INDEX[base_letter])
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

        torsion_xyz = _neighbor_xyz_for_torsions(window, nucleotide_idx)
        torsions, torsion_mask, tau_m, tau_m_ok = nucleotide_torsions_numpy(
            torsion_xyz[0],
            torsion_xyz[1],
            torsion_xyz[2],
            base_letter,
        )
        torsions = np.asarray([wrap_angle_rad(x) for x in torsions], dtype=np.float64)
        torsion_rows.append(torsions)
        mask_rows.append(torsion_mask)
        tau_m_list.append(float(tau_m))
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
        num_classes=len(BASE_TO_INDEX),
    ).float()
    has_pair_tensor = torch.tensor(has_pair_nt, dtype=torch.bool)
    central_nt_mask_t = torch.tensor(central_nt_mask, dtype=torch.bool)
    is_chain_edge_nt_t = torch.tensor(is_chain_edge_nt, dtype=torch.bool)
    touches_chain_edge = bool(is_chain_edge_nt_t.any().item())
    target_nt_idx = torch.tensor(window_size // 2, dtype=torch.long)

    n_bb = len(BACKBONE_ATOMS)
    bb_world = torch.full((window_size, n_bb, 3), float('nan'), dtype=torch.float32)
    for row_idx, nucleotide in enumerate(window):
        experimental = dict(default_atoms_provider(nucleotide))
        for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in experimental:
                bb_world[row_idx, atom_idx] = torch.tensor(np.asarray(experimental[atom_name], dtype=np.float32))

    o3_index = BACKBONE_ATOMS.index("O3'")
    o3_prev_local_rows = []
    o3_prev_valid_rows = []
    for window_pos in range(window_size):
        if window_pos == 0:
            o3_prev_local_rows.append(torch.zeros(3, dtype=torch.float32))
            o3_prev_valid_rows.append(False)
        else:
            o3_world = bb_world[window_pos - 1, o3_index]
            if torch.isnan(o3_world).any():
                o3_prev_local_rows.append(torch.zeros(3, dtype=torch.float32))
                o3_prev_valid_rows.append(False)
            else:
                origin = nt_origins[window_pos]
                frame = nt_frames[window_pos]
                o3_local = (o3_world - origin) @ frame
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
        universe = load_pdb_universe(path)
    elif ext in ('.cif', '.mmcif'):
        universe = load_mmcif_universe(path)
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
        windows = []
        if len(chain) >= window_size:
            first_resid = chain[0].e_residue.resids[0]
            last_resid = chain[-1].e_residue.resids[0]
            chain_direction = 1 if last_resid >= first_resid else -1
            try:
                for window_idx in range(len(chain) - window_size + 1):
                    window = chain[window_idx: window_idx + window_size]
                    resids = [n.e_residue.resids[0] for n in window]
                    steps = [resids[i + 1] - resids[i] for i in range(len(resids) - 1)]
                    if not (all(step == 1 for step in steps) or all(step == -1 for step in steps)):
                        continue

                    data = build_window_data(
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
