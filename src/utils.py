import numpy as np
import requests
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from torch.utils.data import DistributedSampler, Subset
from torch_geometric.loader import DataLoader

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
                        'value': 2
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
    print(f'API request resulted in {len(pdb_ids)} PDB IDs.')

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


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2):
    indices = np.random.permutation(len(dataset)).tolist()
    train_val_split = int(len(dataset) * train_ratio)
    val_test_split = int(len(dataset) * (train_ratio + val_ratio))

    train_indices = indices[:train_val_split]
    val_indices = indices[train_val_split:val_test_split]
    test_indices = indices[val_test_split:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def create_loaders(train_dataset, val_dataset, test_dataset, batch_size, world_size=1, rank=0):
    train_sampler = None
    shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, sampler=train_sampler
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def log_visualization(writer, model, test_data_list, device):
    'Logs a visualization of a generated sample to TensorBoard.'
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
        generated_nodes, generated_pos, _ = model.sample(
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


if __name__ == '__main__':
    print(get_pdb_ids())
