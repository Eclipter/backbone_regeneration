import logging
import os
import tempfile
from functools import lru_cache

import MDAnalysis as mda
import pytorch_lightning as pl
import requests
import torch
from Bio.PDB import PDBIO, MMCIFParser
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


def mmcif_to_mda_universe(path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('struct', path)

    io = PDBIO()
    io.set_structure(structure)
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    tmp_pdb = tmp.name
    tmp.close()
    io.save(tmp_pdb)

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


class VisualizationCallback(pl.Callback):
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        val_dataloader = trainer.datamodule.val_dataloader()
        batch_or_data = next(iter(val_dataloader))

        # Handle both Batch and single Data objects
        if hasattr(batch_or_data, 'get_example'):
            # It's a Batch object, get the first graph
            graph = batch_or_data.get_example(0)
        else:
            # It's a single Data object
            graph = batch_or_data

        graph = graph.to(pl_module.device)

        true_pos, pred_pos = pl_module._get_generations(graph)

        # Reconstruct atom_names_idx
        target_mask = graph.central_mask & graph.backbone_mask
        atom_names_idx = torch.argmax(graph.x[target_mask], dim=1)
        idx_to_atom = {i: atom for atom, i in atom_to_idx.items()}

        # Use the same labels for both true and pred since we don't predict atom types
        atom_names = [f'{idx_to_atom[i.item()]}' for i in atom_names_idx]

        true_labels = [f'true_{name}' for name in atom_names]
        pred_labels = [f'pred_{name}' for name in atom_names]

        all_pos = torch.cat([true_pos, pred_pos], dim=0)
        all_labels = true_labels + pred_labels

        trainer.logger.experiment.add_embedding(
            all_pos,
            metadata=all_labels,
            global_step=trainer.global_step
        )


class NoUnusedParametersWarningFilter(logging.Filter):
    def filter(self, record):
        return 'find_unused_parameters=True' not in record.getMessage()


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
