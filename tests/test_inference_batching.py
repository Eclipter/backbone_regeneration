import MDAnalysis as mda
import numpy as np
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from torch_geometric.data import Data

import base2backbone.inference as pred_mod
from base2backbone.data import BACKBONE_ATOMS, BASE_TO_INDEX, CHAIN_END_CLASS_INTERNAL, N_CHAIN_END_CLASSES
from base2backbone.torsion_constants import N_TORSIONS


class _FakeResidue:
    def __init__(self, resid):
        self.resids = [resid]


class _FakeNucleotide:
    def __init__(self, segid, resid, restype='DA'):
        self.segid = segid
        self.resid = resid
        self.restype = restype
        self.e_residue = _FakeResidue(resid)


class _FakeModel:
    def __init__(self):
        self.batch_sizes = []

    def sample(self, batch):
        self.batch_sizes.append(batch.num_graphs)
        return (
            torch.zeros(batch.num_graphs, N_TORSIONS, device=batch.torsions.device),
            torch.full((batch.num_graphs, 1), 0.3, device=batch.torsions.device),
        )


class _FakeUniverse:
    def __init__(self, num_frames):
        self.trajectory = range(num_frames)


def _make_window_data(window_size):
    base_types = torch.zeros(window_size, len(BASE_TO_INDEX), dtype=torch.float32)
    base_types[:, 0] = 1.0
    chain_end_class = torch.zeros(window_size, N_CHAIN_END_CLASSES, dtype=torch.float32)
    chain_end_class[:, CHAIN_END_CLASS_INTERNAL] = 1.0
    return Data(
        torsions=torch.zeros(window_size, N_TORSIONS, dtype=torch.float32),
        tau_m=torch.full((window_size,), 0.3, dtype=torch.float32),
        nt_origins_world=torch.zeros(window_size, 3, dtype=torch.float32),
        nt_frames_world=torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(window_size, 3, 3).clone(),
        pair_origins_world=torch.zeros(window_size, 3, dtype=torch.float32),
        pair_frames_world=torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(window_size, 3, 3).clone(),
        base_types=base_types,
        chain_end_class=chain_end_class,
    )


def _make_chain_record(segid, resids):
    chain = [_FakeNucleotide(segid, resid) for resid in resids]
    window_size = pred_mod.WINDOW_SIZE
    windows = []
    for window_idx in range(len(chain) - window_size + 1):
        windows.append((
            chain[window_idx:window_idx + window_size],
            window_idx,
            _make_window_data(window_size),
        ))
    return segid, chain, windows


def _stub_builder(theta_w, tau_w, ri, origins_w, frames_w, mask):  # noqa: ARG001
    batch_size, window_size = theta_w.shape[:2]
    coords = torch.zeros(
        batch_size,
        window_size,
        len(BACKBONE_ATOMS),
        3,
        device=theta_w.device,
    )
    for batch_idx in range(batch_size):
        for local_idx in range(window_size):
            coords[batch_idx, local_idx, :, 0] = float(batch_idx)
            coords[batch_idx, local_idx, :, 1] = float(local_idx)
    return coords


def _stub_output_universe(chain_records, predictions, generate_5prime_phosphate=False):  # noqa: ARG001
    n_atoms = len(predictions)
    atom_resindex = list(range(n_atoms))
    residue_segindex = [0] * n_atoms
    universe = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_atoms,
        n_segments=1,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )
    universe.add_TopologyAttr('names', ['P'] * n_atoms)
    universe.add_TopologyAttr('elements', ['P'] * n_atoms)
    universe.add_TopologyAttr('chainIDs', ['A'] * n_atoms)
    universe.add_TopologyAttr('resnames', ['DA'] * n_atoms)
    universe.add_TopologyAttr('resids', list(range(1, n_atoms + 1)))
    universe.add_TopologyAttr('segids', ['A'])
    universe.atoms.positions = np.zeros((n_atoms, 3), dtype=np.float32)
    return universe


def test_predict_backbone_batches_all_windows(monkeypatch):
    chain_records = [_make_chain_record('A', [10, 11, 12, 13])]
    model = _FakeModel()

    monkeypatch.setattr(pred_mod, 'parse_dna', lambda *args, **kwargs: (None, chain_records))
    monkeypatch.setattr(pred_mod, '_load_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(pred_mod, 'build_batch_window_backbone_from_torsions', _stub_builder)
    monkeypatch.setattr(pred_mod, '_build_output_universe', _stub_output_universe)

    output_universe = pred_mod.predict_backbone('dummy.pdb', device='cpu')

    assert model.batch_sizes == [6]
    assert output_universe.atoms.n_atoms == 4 * len(BACKBONE_ATOMS)
    assert len(output_universe.trajectory) == 1


def test_predict_backbone_trajectory_batches_frames(monkeypatch):
    chain_records_by_frame = [
        [_make_chain_record('A', [10, 11, 12, 13])],
        [_make_chain_record('A', [10, 11, 12, 13])],
    ]
    model = _FakeModel()
    universe = _FakeUniverse(num_frames=2)
    remaining = iter(chain_records_by_frame)

    monkeypatch.setattr(
        pred_mod,
        'parse_dna_universe',
        lambda *args, **kwargs: (None, next(remaining)),
    )
    monkeypatch.setattr(pred_mod, '_load_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(pred_mod, 'build_batch_window_backbone_from_torsions', _stub_builder)
    monkeypatch.setattr(pred_mod, '_build_output_universe', _stub_output_universe)

    output_universe = pred_mod.predict_backbone_trajectory(
        universe,
        device='cpu',
    )

    assert model.batch_sizes == [12]
    assert output_universe.atoms.n_atoms == 4 * len(BACKBONE_ATOMS)
    assert len(output_universe.trajectory) == 2


def test_write_structure_writes_multimodel_pdb_and_cif(tmp_path):
    frame0 = _stub_output_universe([], {(0, 0, i): np.zeros(3, dtype=np.float32) for i in range(3)})
    frame1 = _stub_output_universe([], {(0, 0, i): np.ones(3, dtype=np.float32) for i in range(3)})
    trajectory = pred_mod._build_output_trajectory([frame0, frame1])

    pdb_path = pred_mod.write_structure(trajectory, tmp_path / 'traj.pdb')
    cif_path = pred_mod.write_structure(trajectory, tmp_path / 'traj_out', output_format='.cif')

    pdb_text = (tmp_path / 'traj.pdb').read_text()
    assert pdb_path.endswith('.pdb')
    assert pdb_text.count('MODEL') == 2

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('traj', cif_path)
    assert len(list(structure.get_models())) == 2
