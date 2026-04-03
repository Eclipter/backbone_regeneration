import os
import tempfile
from functools import lru_cache

import MDAnalysis as mda
import pytorch_lightning as pl
import requests
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO, Select
from lightning_fabric.utilities.rank_zero import rank_zero_only

from pynamod.atomic_analysis.nucleotides_parser import build_graph, get_base_u

backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "OP1", "OP2", "P", "O3'", "O4'", "O5'"]
nucleic_acid_atoms = ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'O2', 'O4', 'O6']
nucleotide_atoms = nucleic_acid_atoms + backbone_atoms
atom_to_idx = {atom: i for i, atom in enumerate(nucleotide_atoms)}
base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# DNA backbone torsion quads: (nucl_relative_offset, atom_name) × 4.
# Convention follows IUPAC: alpha, beta, gamma, delta, epsilon, zeta.
TORSION_QUADS: list[tuple] = [
    ((-1, "C3'"), (-1, "O3'"), (0, "P"),    (0, "O5'")),   # alpha
    ((-1, "O3'"), (0,  "P"),   (0, "O5'"),  (0, "C5'")),   # beta
    ((0,  "P"),   (0,  "O5'"), (0, "C5'"),  (0, "C4'")),   # gamma
    ((0,  "O5'"), (0,  "C5'"), (0, "C4'"),  (0, "C3'")),   # delta
    ((0,  "C5'"), (0,  "C4'"), (0, "C3'"),  (0, "O3'")),   # epsilon
    ((0,  "C4'"), (0,  "C3'"), (0, "O3'"),  (1, "P")),     # zeta
]
# Number of torsion features per nucleotide: 6 torsions × (sin + cos)
TORSION_FEATURE_DIM = 12


def _compute_dihedral_np(p0, p1, p2, p3) -> float:
    """Dihedral angle in radians for 4 points given as numpy arrays."""
    import numpy as np
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    b2n = b2 / (np.linalg.norm(b2) + 1e-8)
    v = b1 - np.dot(b1, b2n) * b2n
    w = b3 - np.dot(b3, b2n) * b2n
    x = np.dot(v, w)
    y = np.dot(np.cross(b2n, v), w)
    return float(np.arctan2(y, x))


def compute_nucleotide_torsion_features(
    nucl_pos_dicts: list[dict],
    nucl_idx: int,
) -> list[float]:
    """Compute 12 torsion features (sin+cos of 6 backbone torsions) for nucleotide nucl_idx.

    nucl_pos_dicts: list of {atom_name: np.ndarray position} dicts for each nucleotide in window.
    Returns list of 12 floats; entries for missing torsion quads are 0.
    """
    n = len(nucl_pos_dicts)
    features: list[float] = [0.0] * TORSION_FEATURE_DIM
    for torsion_idx, quad in enumerate(TORSION_QUADS):
        coords = []
        valid = True
        for (offset, atom_name) in quad:
            abs_idx = nucl_idx + offset
            if abs_idx < 0 or abs_idx >= n:
                valid = False
                break
            pos = nucl_pos_dicts[abs_idx].get(atom_name)
            if pos is None:
                valid = False
                break
            coords.append(pos)
        if valid:
            angle = _compute_dihedral_np(*coords)
            import math
            features[2 * torsion_idx] = math.sin(angle)
            features[2 * torsion_idx + 1] = math.cos(angle)
    return features


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


def get_edge_type_from_atom_types(edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Classify edges by type: 0 = intra-nucleotide, 1 = phosphodiester (O3'–P).

    Works on-the-fly from atom one-hot features so no dataset reprocessing is needed.
    Both directed halves of each phosphodiester bond are labelled 1.
    """
    o3_idx = atom_to_idx["O3'"]
    p_idx = atom_to_idx['P']
    src_atom = x[edge_index[0]].argmax(dim=1)
    dst_atom = x[edge_index[1]].argmax(dim=1)
    is_phosphodiester = (
        ((src_atom == o3_idx) & (dst_atom == p_idx))
        | ((src_atom == p_idx) & (dst_atom == o3_idx))
    )
    return is_phosphodiester.long()


# Cache edge_index per base-type window to avoid rebuilding reference graphs for every window
def make_edge_index_undirected(edge_index):
    if edge_index.numel() == 0:
        return edge_index
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(edge_index.t(), dim=0).t().contiguous()


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
    # Keep the reference graph undirected so every bonded pair can exchange messages both ways.
    return make_edge_index_undirected(edge_index)


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


class VisualizationCallback(pl.Callback):
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        val_dataloader = getattr(trainer, 'datamodule').val_dataloader()
        batch_or_data = next(iter(val_dataloader))

        # Handle both Batch and single Data objects
        if hasattr(batch_or_data, 'get_example'):
            # It's a Batch object, get the first graph
            graph = batch_or_data.get_example(0)
        else:
            # It's a single Data object
            graph = batch_or_data

        graph = graph.to(pl_module.device)

        true_pos, pred_pos, _, _ = getattr(pl_module, '_get_generations')(graph)

        # Reconstruct atom_names_idx
        target_mask = graph.central_mask & graph.backbone_mask
        atom_names_idx = torch.argmax(graph.x[target_mask], dim=1)
        idx_to_atom = {i: atom for atom, i in atom_to_idx.items()}

        # Use the same labels for both true and pred since we don't predict atom types
        atom_names = [f'{idx_to_atom[int(i.item())]}' for i in atom_names_idx]

        true_labels = [f'true_{name}' for name in atom_names]
        pred_labels = [f'pred_{name}' for name in atom_names]

        all_pos = torch.cat([true_pos, pred_pos], dim=0)
        all_labels = true_labels + pred_labels

        getattr(trainer.logger, 'experiment').add_embedding(
            all_pos,
            metadata=all_labels,
            global_step=trainer.global_step
        )
        # Save graph used for embedding so visualization can load matching sample
        graph_cpu = graph.cpu()
        save_path = os.path.join(str(getattr(trainer.logger, 'log_dir')), 'embedding_graph.pt')
        torch.save(graph_cpu, save_path)


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
