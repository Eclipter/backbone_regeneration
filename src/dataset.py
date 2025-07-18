import os
import os.path as osp
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pynamod import CG_Structure
from pynamod.atomic_analysis.nucleotides_parser import check_if_nucleotide
from utils import atom_to_idx, backbone_atoms, base_to_idx, get_pdb_ids


class PyGDataset(Dataset):
    """
    Датасет для создания графов ДНК из PDB файлов.
    Использует скользящее окно из 3 нуклеотидов для генерации
    структуры сахарофосфатного остова центрального нуклеотида.
    """

    def __init__(self):
        self.window_size = 3

        self.pdb_ids = get_pdb_ids()

        root_path = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'data')
        super().__init__(root_path)

        self.data_list = []
        self.load_processed_paths()

    def __len__(self):
        return len(self.data_list)

    def get(self, idx):
        return torch.load(self.data_list[idx], weights_only=False)

    @property
    def raw_file_names(self):
        return [f'{pdb_id}.pdb' for pdb_id in self.pdb_ids]

    @property
    def processed_file_names(self):
        return 'processing_completion_tag.txt'

    def load_processed_paths(self):
        """Сканирует папку с обработанными данными и заполняет список путей."""
        paths = []
        for root, _, filenames in os.walk(self.processed_dir):
            for filename in filenames:
                if filename.endswith('.pt') and not filename.startswith('pre_'):
                    paths.append(osp.join(root, filename))

        paths.sort()
        self.data_list = paths

    def download(self):
        pdb_ids_to_download = [
            pdb_id
            for pdb_id in self.pdb_ids
            if not osp.exists(osp.join(self.raw_dir, f'{pdb_id}.pdb'))
        ]

        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(self.download_file, pdb_ids_to_download),
                total=len(pdb_ids_to_download),
                desc='Downloading PDB files',
                colour='#b366ff'
            ))

    def download_file(self, pdb_id):
        """Скачивает один PDB файл по PDB ID с помощью Biopython."""
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        response = requests.get(url)

        path = osp.join(self.raw_dir, f'{pdb_id}.pdb')
        with open(path, 'wb') as f:
            f.write(response.content)

    def process(self):
        shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir)
        with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:  # type: ignore
            list(tqdm(
                executor.map(self.process_file, self.pdb_ids),
                total=len(self.pdb_ids),
                colour='#b366ff'
            ))
        # list(tqdm(map(self.process_file, self.pdb_ids), total=len(self.pdb_ids), colour='#b366ff'))

        # Create a completion tag file
        open(osp.join(self.processed_dir, self.processed_file_names), 'w').close()

        # Check how many structures were processed out of fetched ones
        structure_folders = [d for d in os.listdir(self.processed_dir) if osp.isdir(osp.join(self.processed_dir, d))]
        empty_folders = [d for d in structure_folders if not os.listdir(osp.join(self.processed_dir, d))]
        print(f'\n{len(structure_folders)} structures were fetched, {len(empty_folders)} were skipped')

    def process_file(self, pdb_id):
        # Create a folder for the processed data objects
        pdb_id_processed_dir = osp.join(self.processed_dir, pdb_id)
        os.makedirs(pdb_id_processed_dir, exist_ok=True)

        # Create CG_Structure
        raw_path = osp.join(self.raw_dir, f'{pdb_id}.pdb')
        try:
            structure = CG_Structure(file=raw_path)
        except:
            return

        # Analyze DNA chains
        nucleic_atoms = structure.u.select_atoms('nucleic')  # type: ignore
        if len(nucleic_atoms) == 0:
            return
        dna_segids = list(np.unique(nucleic_atoms.segids))
        try:
            structure.analyze_dna(leading_strands=dna_segids)
        except:
            return

        # Group nucleotides by chains
        nucleotides_by_chain = defaultdict(list)
        for segid, nucleotide in zip(
            structure.dna.nucleotides.segids,  # type: ignore
            structure.dna.nucleotides
        ):
            # nucleotide
            nucleotides_by_chain[segid].append(nucleotide)

        # Iterate over chains
        data_idx = 0
        for chain in nucleotides_by_chain.values():
            if len(chain) < self.window_size:
                continue

            # Slide over a chain with a window
            for window_idx in range(len(chain) - self.window_size + 1):
                window = chain[window_idx: window_idx+self.window_size]

                # Iterate over nucleotides in a window
                base_names = []
                central_mask = []
                atom_names = []
                atom_positions = []
                backbone_mask = []
                has_pair_list = []
                for nucleotide_idx, nucleotide in enumerate(window):
                    # Collect nucleotide features

                    sel = structure.u.select_atoms(f'resid {nucleotide.resid} and segid {nucleotide.segid}')  # type: ignore
                    _, ref_nucleotide, base_name = check_if_nucleotide(sel)
                    base_names.append(base_to_idx[base_name])

                    central_mask.append(nucleotide_idx == self.window_size // 2)

                    lead_idxs = structure.dna.pairs_list.lead_nucl_inds  # type: ignore
                    lag_idxs = structure.dna.pairs_list.lag_nucl_inds  # type: ignore
                    has_pair = nucleotide.ind in lead_idxs+lag_idxs
                    has_pair_list.append(has_pair)

                    # Iterate over atoms in a nucleotide
                    for atom in ref_nucleotide:
                        atom_names.append(atom_to_idx[atom.name])
                        backbone_mask.append(1 if atom.name in backbone_atoms else 0)
                        atom_positions.append(atom.position)

                # # Convert features to tensors
                ohe_atom_names = F.one_hot(
                    torch.tensor(atom_names, dtype=torch.long),
                    num_classes=len(atom_to_idx)
                ).float()
                pos_tensor = torch.tensor(atom_positions, dtype=torch.float)
                central_mask_tensor = torch.tensor(central_mask, dtype=torch.bool)
                backbone_mask_tensor = torch.tensor(backbone_mask, dtype=torch.bool)
                has_pair_tensor = torch.tensor(has_pair_list, dtype=torch.bool)
                base_names_tensor = F.one_hot(
                    torch.tensor(base_names, dtype=torch.long),
                    num_classes=len(base_to_idx)
                ).float()

                # Create a data object
                data = Data(
                    x=ohe_atom_names,
                    pos=pos_tensor,
                    central_mask=central_mask_tensor,
                    backbone_mask=backbone_mask_tensor,
                    has_pair=has_pair_tensor,
                    base_names=base_names_tensor
                )

                # Save the data object
                torch.save(data, osp.join(pdb_id_processed_dir, f'{data_idx}.pt'))
                data_idx += 1


class DNADataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_ratio=0.7, val_ratio=0.2):
        super().__init__()

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_workers = 4

        # Индексы для сплитов - вычисляются один раз в prepare_data
        self.train_indices: Optional[list[int]] = None
        self.val_indices: Optional[list[int]] = None
        self.test_indices: Optional[list[int]] = None

    def prepare_data(self):
        # Создаём датасет один раз и вычисляем индексы для сплитов
        self.dataset = PyGDataset()

        indices = np.random.permutation(len(self.dataset)).tolist()

        train_val_split = int(len(self.dataset) * self.train_ratio)
        val_test_split = int(len(self.dataset) * (self.train_ratio + self.val_ratio))

        self.train_indices = indices[:train_val_split]
        self.val_indices = indices[train_val_split:val_test_split]
        self.test_indices = indices[val_test_split:]

    def setup(self):
        self.train_dataset = Subset(self.dataset, self.train_indices)  # type: ignore
        self.val_dataset = Subset(self.dataset, self.val_indices)  # type: ignore
        self.test_dataset = Subset(self.dataset, self.test_indices)  # type: ignore

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)  # type: ignore

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)  # type: ignore


if __name__ == '__main__':
    PyGDataset()

    # structure.dna.nucleotides: Nucleotides_Storage

    # structure.dna.nucleotides.resids: list[int] — индексы нуклеотидов (0-based, уникальны только в пределах цепи). Если все цепи лидирующие, то индексы в порядке возрастания, если цепи разделены, то сначала по возрастанию индексы лидирующих, затем — отстающих

    # structure.dna.nucleotides.segids: list[str] — segids нуклеотидов. Если все цепи лидирующие, то segids в порядке возрастания, если цепи разделены, то сначала по возрастанию segids лидирующих, затем — отстающих

    # structure.dna.nucleotides.restypes: list[str] — типы нуклеотидов (A, C, G, T). Если все цепи лидирующие, то restypes в порядке возрастания индексов, если цепи разделены, то сначала по возрастанию restypes лидирующих, затем — отстающих. Причем лидирующие цепи в одном направлении, отстающие — в обратном

    # structure.dna.nucleotides.ref_frames: torch.Tensor — система отсчета для каждого нуклеотида (N, 3, 3)

    # structure.dna.nucleotides.leading_strands: list[bool] — булевые значения для каждого нуклеотида, указывающие, является ли он лидирующим

    # structure = CG_Structure(file='data/raw/1A0A.pdb')
    # dna_segids = np.unique(structure.u.select_atoms('nucleic').segids)  # type: ignore
    # structure.analyze_dna(leading_strands=['C'])

    # print(len(structure.dna.nucleotides))
