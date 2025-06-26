import os
import os.path as osp

import numpy as np
import requests
import torch
import torch.nn.functional as F
from Bio.PDB.MMCIFParser import MMCIFParser
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from utils import collect_dna_residues, get_pdb_ids, is_dna_chain


class DNADataset(Dataset):
    """
    Датасет для создания графов ДНК из mmCIF файлов.
    Использует скользящее окно из 3 нуклеотидов для генерации
    структуры сахарофосфатного остова центрального нуклеотида.
    """

    def __init__(self, bond_threshold=1.9, window_size=3):
        """
        Args:
            bond_threshold (float): Максимальное расстояние в ангстремах для определения связи.
            window_size (int): Размер скользящего окна (количество нуклеотидов).
        """
        self.bond_threshold = bond_threshold
        self.window_size = window_size

        self.backbone_atoms = {
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"
        }
        self.atom_to_idx = {'C': 0, 'N': 1, 'O': 2, 'P': 3}
        self.supported_resnames = {'DA', 'DC', 'DG', 'DT', 'BRU'}

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
                                atom_features.append(self.atom_to_idx[atom.element])
                                backbone_mask.append(1 if atom.get_name() in self.backbone_atoms else 0)
                                nucleotide_masks.append(j)

                        # Skip if no atoms were found
                        if not positions:
                            continue

                        # Calculate the adjacency matrix for the window
                        pos_tensor = torch.tensor(np.array(positions), dtype=torch.float)
                        dist_matrix = torch.cdist(pos_tensor, pos_tensor)
                        # TODO: implement better bond threshold
                        edge_index = (dist_matrix <= self.bond_threshold).nonzero().t().contiguous()
                        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

                        x = F.one_hot(
                            torch.tensor(atom_features, dtype=torch.long),
                            num_classes=len(self.atom_to_idx)
                        ).float()

                        central_mask = torch.tensor(nucleotide_masks) == self.window_size // 2

                        data = Data(
                            x=x,
                            edge_index=edge_index,
                            pos=pos_tensor,
                            backbone_mask=torch.tensor(backbone_mask, dtype=torch.bool),
                            nucleotide_mask=torch.tensor(nucleotide_masks, dtype=torch.long),
                            central_mask=central_mask,
                            # pdb_id=pdb_id
                        )

                        torch.save(data, osp.join(pdb_processed_dir, f'{data_counter}.pt'))
                        data_counter += 1

        open(osp.join(self.processed_dir, self.processed_file_names), 'w').close()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return torch.load(self.data_list[idx], weights_only=False)


if __name__ == '__main__':
    DNADataset()
