import io
import logging

import MDAnalysis as mda
import numpy as np
import pytorch_lightning as pl
import requests
import torch

from pynamod.atomic_analysis.nucleotides_parser import (build_graph,
                                                        check_if_nucleotide,
                                                        get_base_u)

backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O1P", "O2P", "O3'", "O4'", "O5'"]
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

    graph = build_graph(sel)

    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    edge_index = edges.t().contiguous()

    return edge_index


def get_ref_nucleotide(structure, nucleotide):
    sel = structure.u.select_atoms(f'resid {nucleotide.resid} and segid {nucleotide.segid}')  # type: ignore
    _, ref_nucleotide, base_name = check_if_nucleotide(sel)
    return ref_nucleotide, base_to_idx[base_name]


def has_pair(structure, nucleotide):
    lead_idxs = structure.dna.pairs_list.lead_nucl_inds  # type: ignore
    lag_idxs = structure.dna.pairs_list.lag_nucl_inds  # type: ignore

    return nucleotide.ind in lead_idxs+lag_idxs


class VisualizationCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        # Получаем путь к структуре для визуализации
        structure_path = None
        if hasattr(pl_module, 'structure_path'):
            structure_path = pl_module.structure_path
        elif hasattr(pl_module, 'test_dataloader'):
            # Попытаемся получить путь из dataloader
            try:
                dataloader = pl_module.test_dataloader()
                if hasattr(dataloader.dataset, 'structure_path'):
                    structure_path = dataloader.dataset.structure_path
            except:
                pass

        if structure_path:
            print(f"Структура для анализа: {structure_path}")
            # Здесь можно добавить код для визуализации
            # например, сохранение графиков или 3D структур
        else:
            print("Путь к структуре не найден для визуализации")


class NoUnusedParametersWarningFilter(logging.Filter):
    def filter(self, record):
        return 'find_unused_parameters=True' not in record.getMessage()


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
