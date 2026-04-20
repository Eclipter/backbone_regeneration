import os
import tempfile
from collections import defaultdict
from functools import lru_cache

import MDAnalysis as mda
import numpy as np
import requests
import torch
import torch.nn.functional as F
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO, Select
from MDAnalysis.exceptions import SelectionError
from torch_geometric.data import Data

from pynamod import CG_Structure
from pynamod.atomic_analysis.nucleotides_parser import build_graph, get_base_u

backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "OP1", "OP2", "P", "O3'", "O4'", "O5'"]
nucleic_acid_atoms = ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'O2', 'O4', 'O6']
nucleotide_atoms = nucleic_acid_atoms + backbone_atoms
atom_to_idx = {atom: i for i, atom in enumerate(nucleotide_atoms)}
base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def get_pdb_ids():
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
                        'value': 3
                    }
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_entity_polymer_type',
                        'operator': 'exact_match',
                        'negation': False,
                        'value': 'DNA'
                    }
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_sample_sequence_length',
                        'operator': 'greater_or_equal',
                        'negation': False,
                        'value': 3
                    }
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
                                'value': 'X-RAY DIFFRACTION'
                            }
                        },
                        {
                            'type': 'terminal',
                            'service': 'text',
                            'parameters': {
                                'attribute': 'exptl.method',
                                'operator': 'exact_match',
                                'negation': False,
                                'value': 'ELECTRON MICROSCOPY'
                            }
                        }
                    ],
                    'logical_operator': 'or'
                }
            ],
            'label': 'text'
        },
        'return_type': 'entry',
        'request_options': {
            'paginate': {
                'start': 0,
                'rows': 10000
            },
            'results_content_type': [
                'experimental'
            ],
            'sort': [
                {
                    'sort_by': 'score',
                    'direction': 'desc'
                }
            ],
            'scoring_strategy': 'combined'
        }
    }

    response = requests.post(
        'https://search.rcsb.org/rcsbsearch/v2/query',
        json=query,
        headers={'Content-Type': 'application/json'}
    )
    response.raise_for_status()
    data = response.json()
    pdb_ids = [item['identifier'] for item in data.get('result_set', [])]

    return pdb_ids


def _pdb_element_field_two_chars(atom):
    """Return the wwPDB element field using the validated BioPython element."""
    symbol = (getattr(atom, 'element', None) or '').strip().upper()
    # MDAnalysis does not recognize deuterium as a separate element in PDB input.
    if symbol == 'D':
        symbol = 'H'
    return f'{symbol[:2]:>2}'


def _is_pdb_compatible_residue(residue):
    """PDB has only three columns for resname; wider hetero names shift all later fields."""
    return len(residue.resname.strip()) <= 3


def _is_pdb_compatible_atom(atom):
    """Skip atoms with unknown elements because MDAnalysis will emit topology warnings for them."""
    return (getattr(atom, 'element', None) or '').strip().upper() != 'X'


class _PDBCompatibleSelect(Select):
    def accept_model(self, model):
        return 1

    def accept_chain(self, chain):
        return 1

    def accept_residue(self, residue):  # pyright: ignore[reportIncompatibleMethodOverride]
        return 1 if _is_pdb_compatible_residue(residue) else 0

    def accept_atom(self, atom):  # pyright: ignore[reportIncompatibleMethodOverride]
        return 1 if _is_pdb_compatible_atom(atom) else 0


def _iter_pdb_compatible_atoms(structure):
    for model in structure:
        for chain in model:
            for residue in chain.get_unpacked_list():
                if not _is_pdb_compatible_residue(residue):
                    continue
                for atom in residue.get_unpacked_list():
                    if _is_pdb_compatible_atom(atom):
                        yield atom


def _rewrite_pdb_for_mdanalysis(pdb_path, structure):
    """Normalize fixed-width PDB columns so MDAnalysis reads elements and charges correctly."""
    expected_atoms = list(_iter_pdb_compatible_atoms(structure))

    with open(pdb_path) as f:
        lines = f.readlines()

    out_lines = []
    i_atom = 0
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            atom = expected_atoms[i_atom]
            i_atom += 1
            row = line.rstrip('\n')
            if len(row) < 80:
                row = row + (' ' * (80 - len(row)))
            else:
                row = row[:80]
            elem2 = _pdb_element_field_two_chars(atom)
            row = row[:76] + elem2 + '  '
            out_lines.append(row + '\n')
        else:
            out_lines.append(line)

    if i_atom != len(expected_atoms):
        raise RuntimeError('Atom count mismatch when rewriting PDB for MDAnalysis.')

    with open(pdb_path, 'w') as f:
        f.writelines(out_lines)


def mmcif_to_mda_universe(path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('struct', path)
    assert structure is not None

    io = PDBIO()
    io.set_structure(structure)
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    tmp_pdb = tmp.name
    tmp.close()
    # Drop records that cannot be represented in fixed-width PDB without corrupting later columns.
    io.save(tmp_pdb, select=_PDBCompatibleSelect())

    # BioPython lines can be short or have ambiguous element / charge columns; MDAnalysis
    # then mis-reads resSeq (missing resid), formal charge, and elements. Normalize.
    _rewrite_pdb_for_mdanalysis(tmp_pdb, structure)

    u = mda.Universe(tmp_pdb)
    os.unlink(tmp_pdb)

    return u


# Cache edge_index per base-type window to avoid rebuilding reference graphs for every window
@lru_cache()
def get_edge_idx(base_types: tuple):
    all_edges = []
    atom_selections = []
    atom_offsets = [0]
    current_offset = 0

    # First, collect atom selections from standard structures and offsets
    for base_type in base_types:
        # Use reference nucleotide structures
        sel = get_base_u(base_type)
        if sel is None:
            raise ValueError(f'Unknown base type: {base_type}')
        atom_selections.append(sel)
        current_offset += len(sel)
        atom_offsets.append(current_offset)

    # Build graphs for individual nucleotides and add intra-nucleotide edges
    for i, sel in enumerate(atom_selections):
        graph = build_graph(sel)
        edges = torch.tensor(list(graph.edges), dtype=torch.long)
        all_edges.append(edges + atom_offsets[i])

    # Add inter-nucleotide edges (phosphodiester bonds)
    for i in range(len(base_types) - 1):
        sel1 = atom_selections[i]
        sel2 = atom_selections[i+1]

        # Get atom names using the same renaming scheme as in dataset processing
        atom_names1 = [rename_atom(a.name) for a in sel1]
        atom_names2 = [rename_atom(a.name) for a in sel2]

        try:
            # Find local index of O3' in the first nucleotide
            o3_idx_local = atom_names1.index("O3'")
            # Find local index of P in the second nucleotide
            p_idx_local = atom_names2.index("P")

            # Convert to global indices
            o3_idx_global = atom_offsets[i] + o3_idx_local
            p_idx_global = atom_offsets[i+1] + p_idx_local

            # Add edges for the bond (both directions for an undirected graph)
            bond = torch.tensor([[o3_idx_global, p_idx_global],
                                 [p_idx_global, o3_idx_global]], dtype=torch.long)
            all_edges.append(bond)
        except ValueError:
            # If O3' or P is not found, we cannot form the bond.
            # This might happen with modified residues or at the end of a chain, so we can safely skip.
            pass

    # Concatenate all edges and create the final edge_index tensor
    if not all_edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.cat(all_edges).t().contiguous()
    return edge_index


def has_pair(structure, nucleotide):
    lead_idxs = structure.dna.pairs_list.lead_nucl_inds  # type: ignore
    lag_idxs = structure.dna.pairs_list.lag_nucl_inds  # type: ignore

    return nucleotide.ind in lead_idxs+lag_idxs


def rename_atom(atom_name):
    if atom_name == 'O1P':
        return 'OP1'
    elif atom_name == 'O2P':
        return 'OP2'
    elif atom_name == 'O1A':
        return 'OP1'
    elif atom_name == 'O2A':
        return 'OP2'
    else:
        atom_name = atom_name.replace('A', '').replace('B', '')
        return atom_name


# Heavy atom = not hydrogen by name and not hydrogen/deuterium by element
def _is_heavy_atom(atom):
    return 'H' not in atom.name and getattr(atom, 'element', None) not in {'H', 'D'}


# (atom_name, position) pairs for heavy atoms in the experimental residue, in input order
def default_atoms_provider(nucleotide):
    return [(rename_atom(a.name), a.position) for a in nucleotide.e_residue if _is_heavy_atom(a)]


def inference_atoms_provider(nucleotide):
    """Canonical heavy atoms of the nucleotide; fill positions from the experimental residue
    when present, zeros otherwise. Backbone atoms are always zero-initialised at inference
    time and get replaced by Gaussian noise in the reverse diffusion loop.
    """
    exp_positions = dict(default_atoms_provider(nucleotide))
    atoms = []
    for atom in get_base_u(nucleotide.restype):  # type: ignore
        if not _is_heavy_atom(atom):
            continue
        atom_name = rename_atom(atom.name)
        atoms.append((atom_name, exp_positions.get(atom_name, np.zeros(3, dtype=np.float32))))
    return atoms


def _build_window_data(window, window_idx, chain_len, structure, window_size, atoms_provider):
    """Build a PyG Data object for a single window of consecutive nucleotides.

    atoms_provider controls where per-atom (name, position) pairs come from. Defaults
    to iterating `nucleotide.e_residue` (used at training time); inference can pass a
    reference-based provider when backbone atoms are missing from the input.
    """
    base_types = []
    central_mask = []
    atom_names = []
    atom_positions = []
    backbone_mask = []
    has_pair_list = []
    is_chain_edge_list = []
    for nucleotide_idx, nucleotide in enumerate(window):
        base_type = base_to_idx[nucleotide.restype]
        is_central = nucleotide_idx == window_size // 2
        nt_has_pair = has_pair(structure, nucleotide)
        position_in_chain = window_idx + nucleotide_idx
        is_chain_edge = position_in_chain == 0 or position_in_chain == chain_len - 1

        for atom_name, atom_position in atoms_provider(nucleotide):
            atom_names.append(atom_to_idx[atom_name])
            atom_positions.append(atom_position)
            backbone_mask.append(1 if atom_name in backbone_atoms else 0)
            base_types.append(base_type)
            central_mask.append(is_central)
            has_pair_list.append(nt_has_pair)
            is_chain_edge_list.append(is_chain_edge)

    edge_idx = get_edge_idx(tuple([nucleotide.restype for nucleotide in window]))

    atom_names_tensor = F.one_hot(
        torch.tensor(atom_names, dtype=torch.long),
        num_classes=len(atom_to_idx)
    ).float()
    pos_tensor = torch.tensor(np.asarray(atom_positions), dtype=torch.float)

    # Align to the central nucleotide's reference frame
    central_idx = window[window_size // 2].ind
    # Get R and origin from pynamod storage
    # R: (3, 3), origin: (1, 3)
    ref_frame = getattr(structure.dna.nucleotides, 'ref_frames')[central_idx].float()
    origin = getattr(structure.dna.nucleotides, 'origins')[central_idx].float()

    # Transform positions: (pos - origin) @ R
    pos_tensor = (pos_tensor - origin) @ ref_frame

    central_mask_tensor = torch.tensor(central_mask, dtype=torch.bool)
    backbone_mask_tensor = torch.tensor(backbone_mask, dtype=torch.bool)
    has_pair_tensor = torch.tensor(has_pair_list, dtype=torch.bool)
    is_chain_edge_tensor = torch.tensor(is_chain_edge_list, dtype=torch.bool)
    base_types_tensor = F.one_hot(
        torch.tensor(base_types, dtype=torch.long),
        num_classes=len(base_to_idx)
    ).float()

    return Data(
        x=atom_names_tensor,
        edge_index=edge_idx,
        pos=pos_tensor,
        origin=origin.unsqueeze(0),
        ref_frame=ref_frame.unsqueeze(0),
        central_mask=central_mask_tensor,
        backbone_mask=backbone_mask_tensor,
        has_pair=has_pair_tensor,
        is_chain_edge=is_chain_edge_tensor,
        base_types=base_types_tensor,
    )


def parse_dna(path, use_full_nucleotide, window_size=3, atoms_provider=default_atoms_provider):
    """Read a PDB/mmCIF file, run pynamod DNA analysis, group nucleotides by chain,
    slide sliding windows, and build PyG Data objects for every contiguous window.

    Returns:
        structure: the `CG_Structure` with `analyze_dna` applied.
        chain_records: list of (chain_key, chain, windows) where `chain` is the list of
            pynamod nucleotides grouped by chainID (fallback to segid), and `windows` is
            a list of (window, window_idx, data) for every window of `window_size`
            consecutive residues (gapped windows are skipped). `data` is the PyG Data
            object built with `atoms_provider`.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdb':
        universe = mda.Universe(path)
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

    # Group nucleotides by chains
    nucleotides_by_chain = defaultdict(list)
    for segid, nucleotide in zip(
        structure.dna.nucleotides.segids,  # type: ignore
        structure.dna.nucleotides
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

    # Build per-chain windows (skipping short chains and non-contiguous windows).
    # KeyError/SelectionError inside a chain abort the rest of that chain, matching
    # the original training-time behaviour.
    chain_records = []
    for chain_key, chain in nucleotides_by_chain.items():
        windows: list = []
        if len(chain) >= window_size:
            try:
                for window_idx in range(len(chain) - window_size + 1):
                    window = chain[window_idx: window_idx + window_size]

                    # Check for continuity of residues to avoid gaps in the chain
                    resids = [n.e_residue.resids[0] for n in window]
                    steps = [resids[i+1] - resids[i] for i in range(len(resids)-1)]
                    if not (all(s == 1 for s in steps) or all(s == -1 for s in steps)):
                        continue

                    data = _build_window_data(
                        window, window_idx, len(chain), structure, window_size, atoms_provider,
                    )
                    windows.append((window, window_idx, data))
            except (KeyError, SelectionError):
                pass
        chain_records.append((chain_key, chain, windows))

    return structure, chain_records


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
