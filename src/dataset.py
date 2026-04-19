import gzip
import multiprocessing as mp
import os
import os.path as osp
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, cast

import lightning.pytorch as pl
import numpy as np
import requests
import torch
import torch.nn.functional as F
from MDAnalysis.exceptions import SelectionError
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils
from config import SEED
from pynamod import CG_Structure

PBAR_COLOR = '#B366FF'
EDGE_CACHE_NAME = 'edge_windows.txt'


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
        return [f'{pdb_id}.cif' for pdb_id in self.pdb_ids]

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
            if (
                not osp.exists(osp.join(self.raw_dir, f'{pdb_id}.cif'))
                or osp.getsize(osp.join(self.raw_dir, f'{pdb_id}.cif')) == 0
            )
        ]

        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(self.download_file, pdb_ids_to_download),
                total=len(pdb_ids_to_download),
                desc='Downloading mmCIF files',
                colour=PBAR_COLOR
            ))

        # Single-threaded downloading for debugging
        # list(tqdm(
        #     map(self.download_file, pdb_ids_to_download),
        #     total=len(pdb_ids_to_download),
        #     desc='Downloading mmCIF files',
        #     colour=PBAR_COLOR
        # ))

        tag_path = osp.join(self.processed_dir, self.processed_file_names)
        if osp.exists(tag_path):
            os.remove(tag_path)

    def download_file(self, pdb_id):
        use_mirror = False
        path = osp.join(self.raw_dir, f'{pdb_id}.cif')
        if not use_mirror:
            url = f'https://files.rcsb.org/download/{pdb_id}.cif'
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if not response.content:
                raise ValueError(f'Empty mmCIF response for {pdb_id}')

            # Write into a temporary file first to avoid leaving empty targets behind.
            fd, tmp_path = tempfile.mkstemp(dir=self.raw_dir, suffix='.cif.tmp')
            try:
                with os.fdopen(fd, 'wb') as out_file:
                    out_file.write(response.content)
                os.replace(tmp_path, path)
            except Exception:
                if osp.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        else:
            url = f'https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/{pdb_id.lower()}.cif.gz'
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            fd, tmp_path = tempfile.mkstemp(dir=self.raw_dir, suffix='.cif.tmp')
            if url.endswith('.gz'):
                try:
                    response.raw.decode_content = True
                    with gzip.GzipFile(fileobj=response.raw) as gz_stream, os.fdopen(fd, 'wb') as out_file:
                        shutil.copyfileobj(gz_stream, out_file)
                    if osp.getsize(tmp_path) == 0:
                        raise ValueError(f'Empty mmCIF response for {pdb_id}')
                    os.replace(tmp_path, path)
                except Exception:
                    if osp.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
            else:
                try:
                    with os.fdopen(fd, 'wb') as out_file:
                        out_file.write(response.content)
                    if osp.getsize(tmp_path) == 0:
                        raise ValueError(f'Empty mmCIF response for {pdb_id}')
                    os.replace(tmp_path, path)
                except Exception:
                    if osp.exists(tmp_path):
                        os.remove(tmp_path)
                    raise

    def process(self):
        shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir)

        # Enables single-threaded processing when True
        debug = False

        edge_paths: list[str] = []
        if not debug:
            with ProcessPoolExecutor(
                max_workers=len(os.sched_getaffinity(0)),  # type: ignore
                mp_context=mp.get_context('spawn'),
            ) as executor:
                for per_file_edges in tqdm(
                    executor.map(self.process_file, self.pdb_ids),
                    total=len(self.pdb_ids),
                    colour=PBAR_COLOR
                ):
                    edge_paths.extend(per_file_edges)
        else:
            print('Using single-threaded processing for debugging purposes...')
            for per_file_edges in tqdm(
                map(self.process_file, self.pdb_ids),
                total=len(self.pdb_ids),
                colour=PBAR_COLOR
            ):
                edge_paths.extend(per_file_edges)

        # Persist the edge-windows cache
        edge_paths.sort()
        with open(osp.join(self.processed_dir, EDGE_CACHE_NAME), 'w') as f:
            f.write('\n'.join(edge_paths))

        # Create a completion tag file
        open(osp.join(self.processed_dir, self.processed_file_names), 'w').close()

        # Check how many structures were processed out of fetched ones
        structure_folders = [d for d in os.listdir(self.processed_dir) if osp.isdir(osp.join(self.processed_dir, d))]
        empty_folders = [d for d in structure_folders if not os.listdir(osp.join(self.processed_dir, d))]
        print(f'\n{len(structure_folders)} structures were fetched, {len(empty_folders)} were skipped')

    def process_file(self, pdb_id):
        edge_paths: list[str] = []
        print(2)

        # Create a folder for the processed data objects
        pdb_id_processed_dir = osp.join(self.processed_dir, pdb_id)
        os.makedirs(pdb_id_processed_dir, exist_ok=True)

        # Create CG_Structure
        raw_path = osp.join(self.raw_dir, f'{pdb_id}.cif')
        try:
            structure = CG_Structure(mdaUniverse=utils.mmcif_to_mda_universe(raw_path))
        except:
            return edge_paths

        # Analyze DNA chains
        nucleic_atoms = structure.u.select_atoms('nucleic')  # type: ignore
        if len(nucleic_atoms) == 0:
            return edge_paths
        dna_segids = list(np.unique(nucleic_atoms.segids))
        try:
            structure.analyze_dna(leading_strands=dna_segids, use_full_nucleotide=True)
        except:
            return edge_paths

        # Group nucleotides by chains
        nucleotides_by_chain = defaultdict(list)
        for segid, nucleotide in zip(
            structure.dna.nucleotides.segids,  # type: ignore
            structure.dna.nucleotides
        ):
            atom_group = getattr(nucleotide, 'e_residue').atoms
            atom0 = atom_group[0] if len(atom_group) > 0 else None
            chain_key = ''
            if atom0 is not None:
                chain_key = getattr(atom0, 'chainID', '') or getattr(atom0, 'segid', '')
            if not chain_key:
                chain_key = segid
            if not chain_key:
                continue
            nucleotides_by_chain[chain_key].append(nucleotide)

        # Iterate over chains
        data_idx = 0
        for chain in nucleotides_by_chain.values():
            if len(chain) < self.window_size:
                continue

            # Slide over a chain with a window
            try:
                for window_idx in range(len(chain) - self.window_size + 1):
                    window = chain[window_idx: window_idx + self.window_size]

                    # Check for continuity of residues to avoid gaps in the chain
                    resids = [n.e_residue.resids[0] for n in window]
                    steps = [resids[i+1] - resids[i] for i in range(len(resids)-1)]
                    if not (all(s == 1 for s in steps) or all(s == -1 for s in steps)):
                        continue

                    # Iterate over nucleotides (with sorted atoms) in a window
                    base_types = []
                    central_mask = []
                    atom_names = []
                    atom_positions = []
                    backbone_mask = []
                    has_pair_list = []
                    is_chain_edge_list = []
                    for nucleotide_idx, nucleotide in enumerate(window):
                        # Get nucleotide features
                        base_type = utils.base_to_idx[nucleotide.restype]
                        is_central = nucleotide_idx == self.window_size // 2
                        has_pair = utils.has_pair(structure, nucleotide)
                        # Chain endpoint flag: true iff the nucleotide is the first/last in its chain.
                        position_in_chain = window_idx + nucleotide_idx
                        is_chain_edge = position_in_chain == 0 or position_in_chain == len(chain) - 1

                        # Iterate over atoms in a nucleotide
                        for atom in nucleotide.e_residue:
                            # Rename some atoms to avoid some key errors and skip hydrogens
                            atom_name = utils.rename_atom(atom.name)
                            atom_element = getattr(atom, 'element', None)
                            if 'H' in atom.name or atom_element in {'H', 'D'}:
                                continue

                            # Collect all features atom-wisely
                            atom_names.append(utils.atom_to_idx[atom_name])
                            atom_positions.append(atom.position)
                            backbone_mask.append(1 if atom_name in utils.backbone_atoms else 0)
                            base_types.append(base_type)
                            central_mask.append(is_central)
                            has_pair_list.append(has_pair)
                            is_chain_edge_list.append(is_chain_edge)

                    # Collect window features
                    edge_idx = utils.get_edge_idx(tuple([nucleotide.restype for nucleotide in window]))

                    # Convert features to tensors
                    atom_names_tensor = F.one_hot(
                        torch.tensor(atom_names, dtype=torch.long),
                        num_classes=len(utils.atom_to_idx)
                    ).float()
                    # Convert atom positions to a np array before converting to a tensor
                    pos_tensor = torch.tensor(np.asarray(atom_positions), dtype=torch.float)

                    # Align to the central nucleotide's reference frame
                    central_idx = window[self.window_size // 2].ind
                    # Get R and origin from pynamod storage
                    # R: (3, 3), origin: (1, 3)
                    ref_frame = getattr(structure.dna.nucleotides, 'ref_frames')[central_idx].float()
                    origin = getattr(structure.dna.nucleotides, 'origins')[central_idx].float()

                    # Transform positions: (pos - origin) @ R
                    pos_tensor = (pos_tensor - origin) @ ref_frame

                    central_mask_tensor = torch.tensor(central_mask, dtype=torch.bool)
                    backbone_mask_tensor = torch.tensor(backbone_mask, dtype=torch.bool)
                    has_pair_tensor = torch.tensor(has_pair_list, dtype=torch.bool)
                    is_chain_edge_tensor = torch.tensor(is_chain_edge_list, dtype=torch.bool)
                    base_types_tensor = F.one_hot(
                        torch.tensor(base_types, dtype=torch.long),
                        num_classes=len(utils.base_to_idx)
                    ).float()

                    # Create a data object
                    data = Data(
                        x=atom_names_tensor,
                        edge_index=edge_idx,
                        pos=pos_tensor,
                        origin=origin.unsqueeze(0),
                        ref_frame=ref_frame.unsqueeze(0),
                        central_mask=central_mask_tensor,
                        backbone_mask=backbone_mask_tensor,
                        has_pair=has_pair_tensor,
                        is_chain_edge=is_chain_edge_tensor,
                        base_types=base_types_tensor
                    )

                    # Save the data object
                    save_path = osp.join(pdb_id_processed_dir, f'{data_idx}.pt')
                    torch.save(data, save_path)
                    if any(is_chain_edge_list):
                        edge_paths.append(save_path)
                    data_idx += 1
            except (KeyError, SelectionError):
                continue

        return edge_paths


class DNADataModule(pl.LightningDataModule):
    def __init__(self, target_mode, batch_size, train_ratio=0.7, val_ratio=0.2):
        super().__init__()

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_mode = target_mode

        self.num_workers = len(os.sched_getaffinity(0))
        self.train_generator = torch.Generator().manual_seed(SEED)

    def prepare_data(self):
        dataset = PyGDataset()

        if self.target_mode == 'edge':
            cache_path = osp.join(dataset.processed_dir, EDGE_CACHE_NAME)
            if not osp.exists(cache_path):
                with ThreadPoolExecutor() as executor:
                    flags = list(tqdm(
                        executor.map(self._window_has_chain_edge, dataset.data_list),
                        total=len(dataset.data_list),
                        desc='Filtering edge windows',
                        colour=PBAR_COLOR
                    ))
                kept = [p for p, keep in zip(dataset.data_list, flags) if keep]
                tmp_path = cache_path + '.tmp'
                with open(tmp_path, 'w') as f:
                    f.write('\n'.join(kept))
                os.replace(tmp_path, cache_path)

    @staticmethod
    def _window_has_chain_edge(path):
        return bool(torch.load(path, weights_only=False).is_chain_edge.any().item())

    def setup(self, stage: Optional[str] = None):
        dataset = PyGDataset()

        if self.target_mode == 'edge':
            cache_path = osp.join(dataset.processed_dir, EDGE_CACHE_NAME)
            with open(cache_path) as f:
                dataset.data_list = [line for line in f.read().splitlines() if line]

        # Structure-wise split: group windows by their parent PDB directory so
        # that all windows from a single structure end up in the same subset
        indices_by_structure: dict[str, list[int]] = defaultdict(list)
        for idx, path in enumerate(dataset.data_list):
            pdb_id = osp.basename(osp.dirname(path))
            indices_by_structure[pdb_id].append(idx)

        pdb_ids = sorted(indices_by_structure.keys())
        rng = np.random.default_rng(SEED)
        rng.shuffle(pdb_ids)

        n_structures = len(pdb_ids)
        train_val_split = int(n_structures * self.train_ratio)
        val_test_split = int(n_structures * (self.train_ratio + self.val_ratio))

        train_pdb_ids = pdb_ids[:train_val_split]
        val_pdb_ids = pdb_ids[train_val_split:val_test_split]
        test_pdb_ids = pdb_ids[val_test_split:]

        train_indices = [i for pdb_id in train_pdb_ids for i in indices_by_structure[pdb_id]]
        val_indices = [i for pdb_id in val_pdb_ids for i in indices_by_structure[pdb_id]]
        test_indices = [i for pdb_id in test_pdb_ids for i in indices_by_structure[pdb_id]]

        self.train_dataset: Subset[Data] = Subset(dataset, train_indices)
        self.val_dataset: Subset[Data] = Subset(dataset, val_indices)
        self.test_dataset: Subset[Data] = Subset(dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(
            cast(Dataset, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.train_generator,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )  # type: ignore

    def val_dataloader(self):
        return DataLoader(
            cast(Dataset, self.val_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )  # type: ignore

    def test_dataloader(self):
        return DataLoader(
            cast(Dataset, self.test_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )  # type: ignore


if __name__ == '__main__':
    PyGDataset()

    # structure.dna.nucleotides: Nucleotides_Storage

    # structure.dna.nucleotides.resids: list[int] — индексы нуклеотидов (0-based, уникальны только в пределах цепи). Если все цепи лидирующие, то индексы в порядке возрастания, если цепи разделены, то сначала по возрастанию индексы лидирующих, затем — отстающих

    # structure.dna.nucleotides.segids: list[str] — segids нуклеотидов. Если все цепи лидирующие, то segids в порядке возрастания, если цепи разделены, то сначала по возрастанию segids лидирующих, затем — отстающих

    # structure.dna.nucleotides.restypes: list[str] — типы нуклеотидов (A, C, G, T). Если все цепи лидирующие, то restypes в порядке возрастания индексов, если цепи разделены, то сначала по возрастанию restypes лидирующих, затем — отстающих. Причем лидирующие цепи в одном направлении, отстающие — в обратном

    # structure.dna.nucleotides.ref_frames: torch.Tensor — система отсчета для каждого нуклеотида (N, 3, 3)

    # structure.dna.nucleotides.leading_strands: list[bool] — булевые значения для каждого нуклеотида, указывающие, является ли он лидирующим
