import logging

import pytorch_lightning as pl
import requests
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

atom_types = ['C', 'N', 'O', 'P']
atom_to_idx = {atom: i for i, atom in enumerate(atom_types)}
idx_to_atom = {i: atom for i, atom in enumerate(atom_types)}


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
                        'attribute': 'rcsb_repository_holdings_current.repository_content_types',
                        'operator': 'exact_match',
                        'negation': False,
                        'value': 'entry mmCIF'
                    }
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'rcsb_entry_info.resolution_combined',
                        'operator': 'less_or_equal',
                        'negation': False,
                        'value': 2.5
                    }
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_sample_sequence_length',
                        'operator': 'greater',
                        'negation': False,
                        'value': 3
                    }
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

    return pdb_ids[:1000]


def is_dna_chain(raw_path, chain_id):
    """
    Проверяет, относится ли цепь chain_id к ДНК (polydeoxyribonucleotide) по данным mmCIF.
    """
    mmcif_dict = MMCIF2Dict(raw_path)
    dna_entity_ids = {
        eid for eid, ptype in zip(mmcif_dict['_entity_poly.entity_id'], mmcif_dict['_entity_poly.type'])
        if ptype == 'polydeoxyribonucleotide'
    }
    for aid, eid in zip(mmcif_dict['_struct_asym.id'], mmcif_dict['_struct_asym.entity_id']):
        if aid == chain_id and eid in dna_entity_ids:
            return True
    return False


def collect_dna_residues(structure, chain_id, supported_resnames):
    """
    Возвращает список объектов residue из biopython для одной цепи chain_id,
    только для поддерживаемых нуклеотидных остатков (без воды и гетероостатков).
    """
    for model in structure:  # type: ignore
        for chain in model:
            if chain.id == chain_id:
                return [
                    res for res in chain.get_residues()
                    if res.get_id()[0] == ' ' and res.get_resname() in supported_resnames
                ]
    return []


class VisualizationCallback(pl.Callback):
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        datamodule = getattr(trainer, 'datamodule', None)  # type: ignore
        logger = getattr(trainer, 'logger', None)
        writer = logger.experiment  # type: ignore
        device = pl_module.device

        test_data_list = list(datamodule.test_dataset)  # type: ignore

        original_full_graph = test_data_list[0].to(device)

        # Prepare condition graph
        is_left = original_full_graph.nucleotide_mask == 0
        is_central = original_full_graph.central_mask
        is_right = original_full_graph.nucleotide_mask == 2
        is_base = ~original_full_graph.backbone_mask
        condition_mask = is_left | (is_central & is_base) | (is_right & is_base)
        condition_graph = original_full_graph.subgraph(condition_mask)

        # Get number of nodes to generate
        central_backbone_mask = original_full_graph.backbone_mask & original_full_graph.central_mask
        num_nodes_to_generate = central_backbone_mask.sum().item()

        if num_nodes_to_generate > 0:
            # Generate graph
            generated_nodes, generated_pos, _ = pl_module.sample(
                condition_graph, num_nodes=num_nodes_to_generate
            )

            # Re-center
            generated_pos = generated_pos + original_full_graph.centroid

            # Get original graph for comparison
            original_central_graph = original_full_graph.subgraph(central_backbone_mask)
            orig_pos = original_central_graph.pos + original_full_graph.centroid

            # Log generated structure
            generated_atom_types = [
                idx_to_atom[idx]
                for idx in torch.argmax(generated_nodes, dim=1).cpu().tolist()
            ]
            writer.add_embedding(
                generated_pos,
                metadata=generated_atom_types,
                tag='Generated Backbone'
            )

            # Log original structure
            orig_atom_types = [
                idx_to_atom[idx]
                for idx in torch.argmax(original_central_graph.x, dim=1).cpu().tolist()
            ]
            writer.add_embedding(
                orig_pos,
                metadata=orig_atom_types,
                tag='Original Backbone'
            )


class NoUnusedParametersWarningFilter(logging.Filter):
    def filter(self, record):
        return 'find_unused_parameters=True' not in record.getMessage()


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()
    print(f'API request resulted in {len(pdb_ids)} PDB IDs.')
    print()
