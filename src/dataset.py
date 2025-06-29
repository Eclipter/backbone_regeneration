import os
import os.path as osp

import numpy as np
import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
from Bio.PDB.MMCIFParser import MMCIFParser
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config
from utils import atom_to_idx, collect_dna_residues, get_pdb_ids, is_dna_chain


class PyGDataset(Dataset):
    """
    Датасет для создания графов ДНК из mmCIF файлов.
    Использует скользящее окно из 3 нуклеотидов для генерации
    структуры сахарофосфатного остова центрального нуклеотида.
    """

    def __init__(self, bond_threshold):
        """
        Args:
            bond_threshold (float): Максимальное расстояние в ангстремах для определения связи.
            window_size (int): Размер скользящего окна (количество нуклеотидов).
        """
        self.bond_threshold = bond_threshold

        self.backbone_atoms = {
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"
        }
        self.supported_resnames = {'DA', 'DC', 'DG', 'DT', 'BRU'}
        self.window_size = 3

        self.pdb_ids = get_pdb_ids()
        self.data_list = []

        root_path = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'data')
        super().__init__(root_path)

        self.load_processed_paths()

    def load_processed_paths(self):
        """Сканирует папку с обработанными данными и заполняет список путей."""
        if not osp.exists(self.processed_dir):
            return

        paths = []
        for root, _, files in os.walk(self.processed_dir):
            for file in files:
                if file.endswith('.pt') and not file.startswith('pre_'):
                    paths.append(osp.join(root, file))

        paths.sort()
        self.data_list = paths

    @property
    def raw_file_names(self):
        return [f'{pdb_id}.cif' for pdb_id in self.pdb_ids]

    @property
    def processed_file_names(self):
        return 'processing_completion_tag.txt'

    def download(self):
        base_url = 'https://files.rcsb.org/download/'

        for pdb_id in tqdm(self.pdb_ids, desc='Downloading mmCIF files', colour='#AA80FF'):
            cif_path = osp.join(self.raw_dir, f'{pdb_id}.cif')
            if osp.exists(cif_path):
                continue

            url = f'{base_url}{pdb_id}.cif'
            content = requests.get(url).content
            with open(cif_path, 'wb') as f:
                f.write(content)

    def process(self):
        parser = MMCIFParser(QUIET=True)

        for raw_file_name in tqdm(self.raw_file_names, colour='#AA80FF'):
            pdb_id = raw_file_name.replace('.cif', '')
            pdb_processed_dir = osp.join(self.processed_dir, pdb_id)
            os.makedirs(pdb_processed_dir, exist_ok=True)

            raw_path = osp.join(self.raw_dir, raw_file_name)

            structure = parser.get_structure(pdb_id, raw_path)

            data_counter = 0
            for model in structure:  # type: ignore
                for chain in model:
                    if not is_dna_chain(raw_path, chain.id):
                        continue
                    nucleotides = collect_dna_residues(structure, chain.id, self.supported_resnames)

                    # Run through the chain
                    for i in range(len(nucleotides) - self.window_size + 1):
                        window = nucleotides[i:i + self.window_size]
                        positions = []
                        atom_features = []
                        backbone_mask = []
                        nucleotide_masks = []

                        # Run through all the nucleotides in the window
                        for j, nucleotide in enumerate(window):
                            # Run through all the atoms in the nucleotide
                            for atom in nucleotide:
                                if atom.element == 'H':
                                    continue
                                positions.append(atom.get_coord())
                                atom_features.append(atom_to_idx[atom.element])
                                backbone_mask.append(1 if atom.get_name() in self.backbone_atoms else 0)
                                nucleotide_masks.append(j)

                        # Calculate centroid and center positions
                        pos_tensor = torch.tensor(np.array(positions), dtype=torch.float)
                        centroid = pos_tensor.mean(dim=0)
                        pos_tensor_centered = pos_tensor - centroid

                        # Calculate the adjacency matrix for the window
                        dist_matrix = torch.cdist(pos_tensor, pos_tensor)
                        edge_index = (dist_matrix <= self.bond_threshold).nonzero().t().contiguous()
                        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

                        x = F.one_hot(
                            torch.tensor(atom_features, dtype=torch.long),
                            num_classes=len(atom_to_idx)
                        ).float()

                        central_mask = torch.tensor(nucleotide_masks) == self.window_size // 2

                        data = Data(
                            x=x,
                            edge_index=edge_index,
                            pos=pos_tensor_centered,
                            backbone_mask=torch.tensor(backbone_mask, dtype=torch.bool),
                            nucleotide_mask=torch.tensor(nucleotide_masks, dtype=torch.long),
                            central_mask=central_mask,
                            centroid=centroid
                        )

                        torch.save(data, osp.join(pdb_processed_dir, f'{data_counter}.pt'))
                        data_counter += 1

        open(osp.join(self.processed_dir, self.processed_file_names), 'w').close()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return torch.load(self.data_list[idx], weights_only=False)


class DNADataModule(pl.LightningDataModule):
    def __init__(self, bond_threshold, batch_size, train_ratio=0.7, val_ratio=0.2):
        super().__init__()
        self.bond_threshold = bond_threshold
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        PyGDataset(self.bond_threshold)

    def setup(self, stage=None):
        self.dataset = PyGDataset(self.bond_threshold)
        indices = np.random.permutation(len(self.dataset)).tolist()
        train_val_split = int(len(self.dataset) * self.train_ratio)
        val_test_split = int(len(self.dataset) * (self.train_ratio + self.val_ratio))

        train_indices = indices[:train_val_split]
        val_indices = indices[train_val_split:val_test_split]
        test_indices = indices[val_test_split:]

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore


if __name__ == '__main__':
    PyGDataset(config.BOND_THRESHOLD)
