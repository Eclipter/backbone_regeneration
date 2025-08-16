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

import utils
from pynamod import CG_Structure


class PyGDataset(Dataset):
    def __init__(self):
        self.window_size = 3

        self.pdb_ids = utils.get_pdb_ids()

        root_path = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'data')
        super().__init__(root_path)

        self.data_list = []
        self.load_processed_paths()

    def __len__(self):
        return len(self.data_list)

    # Required by torch_geometric.data.Dataset for internal length access
    def len(self):
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

        # Single-threaded processing for debugging
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
            nucleotides_by_chain[segid].append(nucleotide)

        # Iterate over chains
        data_idx = 0
        for chain in nucleotides_by_chain.values():
            try:
                if len(chain) < self.window_size:
                    continue

                # Slide over a chain with a window
                for window_idx in range(len(chain) - self.window_size + 1):
                    window = chain[window_idx: window_idx+self.window_size]

                    # Collect window features
                    edge_idx = utils.get_edge_idx(structure, window)

                    # Iterate over nucleotides in a window
                    base_types = []
                    central_mask = []
                    atom_names = []
                    atom_positions = []
                    backbone_mask = []
                    has_pair_list = []
                    for nucleotide_idx, nucleotide in enumerate(window):
                        # Get nucleotide features
                        base_type = utils.base_to_idx[nucleotide.restype]
                        is_central = nucleotide_idx == self.window_size // 2
                        has_pair = utils.has_pair(structure, nucleotide)

                        # Iterate over atoms in a nucleotide
                        nucleotide_universe = structure.u.select_atoms(f'resid {nucleotide.resid} and segid {nucleotide.segid}')
                        for atom in nucleotide_universe:
                            # Rename some atoms to avoid key errors and skip hydrogens
                            if atom.name == 'OP1':
                                atom_name = 'O1P'
                            elif atom.name == 'OP2':
                                atom_name = 'O2P'
                            elif 'H' in atom.name:
                                continue
                            else:
                                atom_name = atom.name

                            # Collect all features atom-wisely
                            atom_names.append(utils.atom_to_idx[atom_name])
                            backbone_mask.append(1 if atom_name in utils.backbone_atoms else 0)
                            atom_positions.append(atom.position)
                            base_types.append(base_type)
                            central_mask.append(is_central)
                            has_pair_list.append(has_pair)

                    # Convert features to tensors
                    ohe_atom_names = F.one_hot(
                        torch.tensor(atom_names, dtype=torch.long),
                        num_classes=len(utils.atom_to_idx)
                    ).float()
                    pos_tensor = torch.tensor(atom_positions, dtype=torch.float)
                    central_mask_tensor = torch.tensor(central_mask, dtype=torch.bool)
                    backbone_mask_tensor = torch.tensor(backbone_mask, dtype=torch.bool)
                    has_pair_tensor = torch.tensor(has_pair_list, dtype=torch.bool)
                    base_types_tensor = F.one_hot(
                        torch.tensor(base_types, dtype=torch.long),
                        num_classes=len(utils.base_to_idx)
                    ).float()

                    # Create a data object
                    data = Data(
                        x=ohe_atom_names,
                        edge_index=edge_idx,
                        pos=pos_tensor,
                        central_mask=central_mask_tensor,
                        backbone_mask=backbone_mask_tensor,
                        has_pair=has_pair_tensor,
                        base_types=base_types_tensor
                    )

                    # Save the data object
                    torch.save(data, osp.join(pdb_id_processed_dir, f'{data_idx}.pt'))
                    data_idx += 1
            except KeyError:
                continue


class DNADataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_ratio=0.7, val_ratio=0.2):
        super().__init__()

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.num_workers = 4

    def prepare_data(self):
        # Download and process data
        PyGDataset()

    def setup(self, stage: Optional[str] = None):
        dataset = PyGDataset()

        indices = np.random.permutation(len(dataset)).tolist()
        train_val_split = int(len(indices) * self.train_ratio)
        val_test_split = int(len(indices) * (self.train_ratio + self.val_ratio))

        train_indices = indices[:train_val_split]
        val_indices = indices[train_val_split:val_test_split]
        test_indices = indices[val_test_split:]

        self.train_dataset = Subset(dataset, train_indices)  # type: ignore
        self.val_dataset = Subset(dataset, val_indices)  # type: ignore
        self.test_dataset = Subset(dataset, test_indices)  # type: ignore

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
