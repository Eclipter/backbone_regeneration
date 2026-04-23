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
from torch.utils.data import Sampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils
from config import SEED

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

        # Persist the edge-windows cache so the sampler can classify windows
        # without having to load every Data file at setup time.
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

        pdb_id_processed_dir = osp.join(self.processed_dir, pdb_id)
        os.makedirs(pdb_id_processed_dir, exist_ok=True)

        # Load the mmCIF, run pynamod's DNA analysis, and materialise windows
        raw_path = osp.join(self.raw_dir, f'{pdb_id}.cif')
        try:
            _, chain_records = utils.parse_dna(
                raw_path, use_full_nucleotide=True, window_size=self.window_size,
            )
        except:
            return edge_paths

        data_idx = 0
        for _, _, windows in chain_records:
            for _, _, data in windows:
                save_path = osp.join(pdb_id_processed_dir, f'{data_idx}.pt')
                torch.save(data, save_path)
                if bool(data.is_chain_edge.any().item()):
                    edge_paths.append(save_path)
                data_idx += 1

        return edge_paths


class WindowTargetDataset(torch.utils.data.Dataset):
    """Exposes (window, target_type) virtual samples on top of PyGDataset.

    Every window contributes a central-target sample. Edge windows additionally
    contribute an edge-target sample, so the unified model can be trained on
    both tasks from the same backing windows. The `is_target` per-atom mask
    attached to each emitted Data tells the model which atoms to denoise.
    """

    CENTRAL = 0
    EDGE = 1

    def __init__(self, base: 'PyGDataset', window_indices: list[int], is_edge_flags: list[bool]):
        assert len(window_indices) == len(is_edge_flags)
        self.base = base
        # virtual_entries[i] = (window_idx_in_base, target_type)
        self.virtual_entries: list[tuple[int, int]] = []
        self.central_virtual: list[int] = []
        self.edge_virtual: list[int] = []
        for w_idx, is_edge in zip(window_indices, is_edge_flags):
            self.central_virtual.append(len(self.virtual_entries))
            self.virtual_entries.append((w_idx, self.CENTRAL))
            if is_edge:
                self.edge_virtual.append(len(self.virtual_entries))
                self.virtual_entries.append((w_idx, self.EDGE))

    def __len__(self):
        return len(self.virtual_entries)

    def __getitem__(self, i):
        w_idx, target_type = self.virtual_entries[i]
        data = self.base.get(w_idx).clone()
        if target_type == self.CENTRAL:
            data.is_target = data.central_mask.clone()
        else:
            data.is_target = data.is_chain_edge.clone()
            # For edge-target samples, express positions in the edge nucleotide's
            # own reference frame instead of the central one.
            edge_flat = data.is_chain_edge.view(-1)
            edge_atom_idx = int(torch.where(edge_flat)[0][0].item())
            utils.reframe_positions_to_atom(data, edge_atom_idx)
        return data


class EdgeCentralTargetSampler(Sampler[int]):
    """Weighted sampler yielding edge-target and central-target virtual entries
    in a fixed ratio (default 1:2).

    Note: the ratio is applied at the *target* level. Edge-target samples come
    only from edge windows; central-target samples come from any window, which
    matches the "edge window can be used to reconstruct the central backbone
    too, but not vice versa" requirement.
    """

    def __init__(
        self,
        dataset: WindowTargetDataset,
        edge_weight: float = 1.0,
        central_weight: float = 2.0,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        num_edge = len(dataset.edge_virtual)
        num_central = len(dataset.central_virtual)

        weights = torch.zeros(len(dataset), dtype=torch.double)
        if num_edge > 0:
            weights[dataset.edge_virtual] = edge_weight / num_edge
        if num_central > 0:
            weights[dataset.central_virtual] = central_weight / num_central

        self.weights = weights
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.generator = generator

    def __iter__(self):
        idx = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=self.generator
        )
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


class DNADataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_ratio=0.7, val_ratio=0.2,
                 edge_weight=1.0, central_weight=2.0):
        super().__init__()

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.edge_weight = edge_weight
        self.central_weight = central_weight

        self.num_workers = len(os.sched_getaffinity(0))
        self.train_generator = torch.Generator().manual_seed(SEED)

    def prepare_data(self):
        PyGDataset()

    def _load_edge_paths(self, dataset: 'PyGDataset') -> set[str]:
        cache_path = osp.join(dataset.processed_dir, EDGE_CACHE_NAME)
        if not osp.exists(cache_path):
            raise FileNotFoundError(
                f'Edge-window cache `{cache_path}` is missing. Re-run processing.'
            )
        with open(cache_path) as f:
            return {line for line in f.read().splitlines() if line}

    def setup(self, stage: Optional[str] = None):
        dataset = PyGDataset()
        edge_paths = self._load_edge_paths(dataset)

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

        def _wrap(indices: list[int]) -> WindowTargetDataset:
            flags = [dataset.data_list[i] in edge_paths for i in indices]
            return WindowTargetDataset(dataset, indices, flags)

        self.train_dataset = _wrap(train_indices)
        self.val_dataset = _wrap(val_indices)
        self.test_dataset = _wrap(test_indices)

    def train_dataloader(self):
        sampler = EdgeCentralTargetSampler(
            self.train_dataset,
            edge_weight=self.edge_weight,
            central_weight=self.central_weight,
            generator=self.train_generator,
        )
        return DataLoader(
            cast(Dataset, self.train_dataset),
            batch_size=self.batch_size,
            sampler=sampler,
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
