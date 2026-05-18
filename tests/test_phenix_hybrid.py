import numpy as np

from base2backbone.data import BACKBONE_ATOMS
from base2backbone.geometry.templates import get_template
from base2backbone.inference import (
    build_phenix_hybrid_universe,
    template_backbone_predictions_from_structure,
)


class _FakeResidue:
    def __init__(self, resid):
        self.resids = [resid]


class _FakeAtom:
    def __init__(self, name, position, element='C'):
        self.name = name
        self.position = position
        self.element = element


class _FakeNucleotide:
    def __init__(self, ind, segid, resid, restype='A'):
        self.ind = ind
        self.segid = segid
        self.resid = resid
        self.restype = restype
        self.e_residue = _FakeResidue(resid)


class _FakeDNANucleotides:
    def __init__(self, origins, ref_frames):
        self.origins = origins
        self.ref_frames = ref_frames


class _FakeDNA:
    def __init__(self, origins, ref_frames):
        self.nucleotides = _FakeDNANucleotides(origins, ref_frames)


class _FakeStructure:
    def __init__(self, origins, ref_frames):
        self.dna = _FakeDNA(origins, ref_frames)


def test_template_backbone_predictions_use_pynamod_frames():
    origin = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    frame = np.eye(3, dtype=np.float64)
    nt = _FakeNucleotide(0, 'A', 7, restype='A')
    structure = _FakeStructure([origin], [frame])
    chain_records = [('A', [nt], [])]

    predictions = template_backbone_predictions_from_structure(structure, chain_records)
    tpl = get_template('A')

    for atom_name in BACKBONE_ATOMS:
        if atom_name not in tpl:
            continue
        local = np.asarray(tpl[atom_name], dtype=np.float64)
        expected_world = (local @ frame.T + origin).astype(np.float32)
        np.testing.assert_allclose(
            predictions[('A', 7, atom_name)],
            expected_world,
            rtol=0,
            atol=1e-5,
        )


def test_phenix_hybrid_universe_uses_template_backbone_and_experimental_bases(monkeypatch):
    origin = np.zeros(3, dtype=np.float64)
    frame = np.eye(3, dtype=np.float64)
    nt = _FakeNucleotide(0, 'A', 1, restype='A')
    structure = _FakeStructure([origin], [frame])
    chain_records = [('A', [nt], [])]

    exp_base_xyz = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def fake_default_atoms_provider(nucleotide):  # noqa: ARG001
        return [('N9', exp_base_xyz), ("C1'", np.zeros(3, dtype=np.float32))]

    def fake_inference_atoms_provider(nucleotide):  # noqa: ARG001
        base_atoms = [('N9', exp_base_xyz)]
        backbone_atoms = [(name, np.zeros(3, dtype=np.float32)) for name in BACKBONE_ATOMS[:3]]
        return base_atoms + backbone_atoms

    monkeypatch.setattr(
        'base2backbone.inference.default_atoms_provider',
        fake_default_atoms_provider,
    )
    monkeypatch.setattr(
        'base2backbone.inference.inference_atoms_provider',
        fake_inference_atoms_provider,
    )

    universe = build_phenix_hybrid_universe(structure, chain_records)
    assert universe.atoms is not None
    positions = {
        (atom.resname, atom.name): np.asarray(atom.position, dtype=np.float32)
        for atom in universe.atoms
    }

    tpl = get_template('A')
    assert np.allclose(positions[('A', 'N9')], exp_base_xyz)
    assert np.allclose(positions[('A', "C1'")], tpl["C1'"].astype(np.float32))
