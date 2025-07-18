import io
import logging

import MDAnalysis as mda
import numpy as np
import pytorch_lightning as pl
import requests
import torch

from pynamod.atomic_analysis.base_structures import nucleotides_pdb
from pynamod.atomic_analysis.nucleotides_parser import get_base_u

backbone_atoms = ['C1*', 'C2*', 'C3*', 'C4*', 'C5*', 'O1P', 'O2P', 'O3*', 'O4*', 'O5*']
nucleic_acid_atoms = ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', 'C2', 'C4', 'C5', 'C6', 'C8', 'O2', 'O4', 'O6']
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
                },
                # {
                #     'type': 'terminal',
                #     'service': 'text',
                #     'parameters': {
                #         'attribute': 'pdbx_database_status.pdb_format_compatible',
                #         'operator': 'exact_match',
                #         'negation': False,
                #         'value': 'Y'
                #     }
                # }
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


# ============================================================================
# Функции для работы с локальными координатами атомов
# ============================================================================

def get_local_atom_coordinates(nucleotide, atom_names=None):
    """
    Получить локальные координаты атомов нуклеотида в его собственной системе отсчета.

    Args:
        nucleotide: объект Nucleotide из pynamod
        atom_names: список имен атомов (если None, возвращает все)

    Returns:
        dict: {atom_name: local_coords} - локальные координаты атомов
    """
    # Получаем эталонную структуру нуклеотида
    base_type = nucleotide.restype
    standard_structure = get_base_u(base_type)

    # Локальные координаты в эталонной системе отсчета
    local_coords = {}

    for atom in standard_structure.atoms:
        if atom_names is None or atom.name in atom_names:
            # Координаты уже в локальной системе эталонного нуклеотида
            local_coords[atom.name] = torch.from_numpy(atom.position).float()

    return local_coords


def transform_to_global_coordinates(local_coords, ref_frame, origin):
    """
    Преобразовать локальные координаты атомов в глобальные.

    Args:
        local_coords: dict {atom_name: local_coords}
        ref_frame: матрица поворота (3x3)
        origin: начало координат (3,)

    Returns:
        dict: {atom_name: global_coords}
    """
    global_coords = {}

    for atom_name, local_pos in local_coords.items():
        # Применяем поворот и сдвиг: global = R @ local + origin
        global_pos = torch.matmul(ref_frame, local_pos) + origin.squeeze()
        global_coords[atom_name] = global_pos

    return global_coords


def get_backbone_atoms_local_coords(base_type):
    """
    Получить локальные координаты только backbone атомов для данного типа основания.

    Args:
        base_type: тип основания ('A', 'T', 'G', 'C')

    Returns:
        dict: {atom_name: local_coords} для backbone атомов
    """
    # Список backbone атомов (сахарофосфатный остов)
    backbone_atoms_list = ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]

    # Получаем полную структуру нуклеотида с backbone
    full_nucleotide_pdb = get_full_nucleotide_structure(base_type)

    structure = mda.Universe(io.StringIO(full_nucleotide_pdb), format='PDB')

    local_coords = {}
    for atom in structure.atoms:
        if atom.name in backbone_atoms_list:
            local_coords[atom.name] = torch.from_numpy(atom.position).float()

    return local_coords


def get_full_nucleotide_structure(base_type):
    """
    Получить полную структуру нуклеотида включая backbone.
    Здесь используются стандартные координаты из литературы.
    """
    # Стандартные координаты backbone для B-формы ДНК
    # Эти координаты нужно взять из кристаллографических данных

    backbone_coords = {
        'P': [0.000, 0.000, 0.000],
        'OP1': [1.485, 0.000, 0.000],
        'OP2': [-0.742, 1.285, 0.000],
        "O5'": [-0.742, -1.285, 0.000],
        "C5'": [-0.742, -2.685, 0.000],
        "C4'": [-2.142, -3.427, 0.000],
        "O4'": [-3.342, -2.685, 0.000],
        "C3'": [-2.142, -4.827, 0.000],
        "O3'": [-3.342, -5.569, 0.000],
        "C2'": [-0.742, -5.569, 0.000],
        "C1'": [-2.142, -1.285, 0.000],
    }

    # Добавляем координаты основания из pynamod
    base_structure = get_base_u(base_type)

    # Формируем PDB строку
    pdb_lines = ["REMARK Full nucleotide structure"]
    atom_id = 1

    # Добавляем backbone атомы
    for atom_name, coords in backbone_coords.items():
        pdb_line = f"ATOM  {atom_id:5d}  {atom_name:<4s} {base_type} A   1    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00 20.00           {atom_name[0]}"
        pdb_lines.append(pdb_line)
        atom_id += 1

    # Добавляем атомы основания (со сдвигом к C1')
    base_offset = np.array([-2.142, -1.285, 0.000])  # позиция C1'
    for atom in base_structure.atoms:
        coords = atom.position + base_offset
        pdb_line = f"ATOM  {atom_id:5d}  {atom.name:<4s} {base_type} A   1    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00 20.00           {atom.name[0]}"
        pdb_lines.append(pdb_line)
        atom_id += 1

    pdb_lines.append("END")
    return '\n'.join(pdb_lines)


def atoms_to_local_coordinates(nucleotide, global_atom_coords):
    """
    Преобразовать глобальные координаты атомов в локальные координаты нуклеотида.

    Args:
        nucleotide: объект Nucleotide из pynamod
        global_atom_coords: dict {atom_name: global_coords}

    Returns:
        dict: {atom_name: local_coords}
    """
    ref_frame = nucleotide.ref_frame  # (3, 3)
    origin = nucleotide.origin.squeeze()  # (3,)

    local_coords = {}

    for atom_name, global_pos in global_atom_coords.items():
        # Обратное преобразование: local = R^T @ (global - origin)
        local_pos = torch.matmul(ref_frame.T, global_pos - origin)
        local_coords[atom_name] = local_pos

    return local_coords


def prepare_training_data(structure):
    """
    Получение локальных координат для обучения диффузионной модели.

    Args:
        structure: CG_Structure из pynamod

    Returns:
        list: список локальных координат backbone для каждого нуклеотида
    """
    local_coords_list = []

    for nucleotide in structure.dna.nucleotides:
        # Локальные координаты backbone в системе отсчета нуклеотида
        backbone_local = get_backbone_atoms_local_coords(nucleotide.restype)
        local_coords_list.append(backbone_local)

    return local_coords_list


def diffusion_predict_to_global(predicted_local_coords, central_nucleotide):
    """
    Преобразование предсказанных локальных координат в глобальные.

    Args:
        predicted_local_coords: dict с предсказанными локальными координатами
        central_nucleotide: центральный нуклеотид из контекста

    Returns:
        dict: глобальные координаты backbone атомов
    """
    global_coords = transform_to_global_coordinates(
        predicted_local_coords,
        central_nucleotide.ref_frame,
        central_nucleotide.origin
    )

    return global_coords


# Пример использования функций
def example_local_coords_usage():
    """
    Пример использования функций для работы с локальными координатами.
    """
    from pynamod import CG_Structure

    # Загружаем структуру
    structure = CG_Structure(file='example.pdb')
    structure.analyze_dna(leading_strands=['A'])

    # Берем первый нуклеотид
    nucleotide = structure.dna.nucleotides[0]

    # Получаем локальные координаты backbone атомов
    backbone_local = get_backbone_atoms_local_coords(nucleotide.restype)

    # Преобразуем в глобальные координаты
    backbone_global = transform_to_global_coordinates(
        backbone_local,
        nucleotide.ref_frame,
        nucleotide.origin
    )

    # Обратное преобразование
    backbone_local_check = atoms_to_local_coordinates(nucleotide, backbone_global)

    print("Локальные координаты backbone атомов:")
    for atom_name, coords in backbone_local.items():
        print(f"{atom_name}: {coords}")

    return backbone_local, backbone_global


if __name__ == '__main__':
    pdb_ids = get_pdb_ids()

    print(f'API request resulted in {len(pdb_ids)} PDB IDs.', end='\n')
    print(pdb_ids)
