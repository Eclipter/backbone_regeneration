import logging

import pytorch_lightning as pl
import requests
import torch
from lightning_fabric.utilities.rank_zero import rank_zero_only

from pynamod.atomic_analysis.nucleotides_parser import build_graph

backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O1P", "O2P", "P", "O3'", "O4'", "O5'"]
nucleic_acid_atoms = ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'O2', 'O4', 'O6']
nucleotide_atoms = nucleic_acid_atoms + backbone_atoms
atom_to_idx = {atom: i for i, atom in enumerate(nucleotide_atoms)}
base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 4}


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


def get_edge_idx(structure, window):
    sel_str = ' or '.join(f'(resid {nucleotide.resid} and segid {nucleotide.segid})' for nucleotide in window)
    sel = structure.u.select_atoms(sel_str)  # type: ignore
    sel = sel.select_atoms('not name H*')

    graph = build_graph(sel)

    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    edge_index = edges.t().contiguous()

    return edge_index


def has_pair(structure, nucleotide):
    lead_idxs = structure.dna.pairs_list.lead_nucl_inds  # type: ignore
    lag_idxs = structure.dna.pairs_list.lag_nucl_inds  # type: ignore

    return nucleotide.ind in lead_idxs+lag_idxs


class VisualizationCallback(pl.Callback):
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        val_dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))

        # Get a single example from the batch for visualization
        graph = batch.get_example(0)
        graph = graph.to(pl_module.device)

        target_mask = graph.central_mask & graph.backbone_mask
        true_pos = graph.pos[target_mask]
        true_atom_types_idx = torch.argmax(graph.x[target_mask], dim=1)

        pred_x, pred_pos = pl_module.sample(graph)
        pred_atom_types_idx = torch.argmax(pred_x, dim=1)

        idx_to_atom = {i: atom for atom, i in atom_to_idx.items()}
        true_atom_types = [f'true_{idx_to_atom[i.item()]}' for i in true_atom_types_idx]
        pred_atom_types = [f'pred_{idx_to_atom[i.item()]}' for i in pred_atom_types_idx]

        all_pos = torch.cat([true_pos, pred_pos], dim=0)
        all_labels = true_atom_types + pred_atom_types

        trainer.logger.experiment.add_embedding(
            all_pos,
            metadata=all_labels,
            tag=f'epoch_{trainer.current_epoch}_structure',
            global_step=trainer.global_step
        )


class NoUnusedParametersWarningFilter(logging.Filter):
    def filter(self, record):
        return 'find_unused_parameters=True' not in record.getMessage()


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
