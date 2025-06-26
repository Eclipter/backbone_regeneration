import numpy as np
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from torch.utils.data import DistributedSampler, Subset
from torch_geometric.loader import DataLoader


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
    print(f'API request resulted in {len(pdb_ids)} PDB IDs')

    return pdb_ids[:500]


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


if __name__ == '__main__':
    print(get_pdb_ids()[:100])
