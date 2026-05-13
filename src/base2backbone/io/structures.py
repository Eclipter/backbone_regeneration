"""Structure loading and atom normalization helpers."""

import os
import tempfile
import warnings

import gemmi
import MDAnalysis as mda
import numpy as np

from pynamod.atomic_analysis.nucleotides_parser import \
    get_base_u  # pyright: ignore[reportMissingImports]

ATOM_RENAMES = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}


def load_pdb_universe(pdb_path):
    """Open a PDB file with MDAnalysis and silence noisy element warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Unknown element .+ found for some atoms\.',
            category=UserWarning,
            module=r'MDAnalysis\.topology\.PDBParser',
        )
        universe = mda.Universe(pdb_path)
    universe.guess_TopologyAttrs(context='default', to_guess=['elements'])
    return universe


def load_mmcif_universe(path):
    """Parse an mmCIF file using gemmi and return an MDAnalysis Universe."""
    structure = gemmi.read_structure(path)
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    tmp_pdb = tmp.name
    tmp.close()
    try:
        pdb_opts = gemmi.PdbWriteOptions()
        pdb_opts.cryst1_record = False
        structure.write_pdb(tmp_pdb, pdb_opts)
        universe = load_pdb_universe(tmp_pdb)
    finally:
        os.unlink(tmp_pdb)
    return universe


def rename_atom(name: str) -> str:
    return ATOM_RENAMES.get(name, name.rstrip('AB'))


def is_heavy_atom(atom) -> bool:
    return 'H' not in atom.name and getattr(atom, 'element', None) not in {'H', 'D'}


def default_atoms_provider(nucleotide):
    return [(rename_atom(atom.name), atom.position) for atom in nucleotide.e_residue if is_heavy_atom(atom)]


def inference_atoms_provider(nucleotide):
    """Canonical heavy atoms for a nucleotide with zero-filled missing positions."""
    experimental_positions = dict(default_atoms_provider(nucleotide))
    atoms = []
    for atom in get_base_u(nucleotide.restype):  # type: ignore
        if not is_heavy_atom(atom):
            continue
        atom_name = rename_atom(atom.name)
        atoms.append((atom_name, experimental_positions.get(atom_name, np.zeros(3, dtype=np.float32))))
    return atoms


def heavy_xyz_dict(nucleotide) -> dict[str, np.ndarray]:
    return {
        rename_atom(atom.name): np.asarray(atom.position, dtype=np.float64)
        for atom in nucleotide.e_residue
        if is_heavy_atom(atom)
    }


def has_pair(structure, nucleotide) -> bool:
    """Check if a nucleotide has a pair."""
    lead_idxs = structure.dna.pairs_list.lead_nucl_inds  # type: ignore
    lag_idxs = structure.dna.pairs_list.lag_nucl_inds  # type: ignore
    return nucleotide.ind in lead_idxs + lag_idxs
