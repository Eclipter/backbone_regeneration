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


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
