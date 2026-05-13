"""Canonical nucleotide template geometry."""

import functools

import numpy as np
import torch

from pynamod.atomic_analysis.nucleotides_parser import \
    get_base_u  # pyright: ignore[reportMissingImports]

from .primitives import bond_angle_numpy, dihedral_rad
from .torsions import RING_TORSION_DEFS

_TEMPLATE_CACHE: dict[str, dict[str, np.ndarray]] = {}
_TEMPLATE_TAU_M: dict[str, float] = {}
_TEMPLATE_RENAMES = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}


def get_template(restype: str) -> dict[str, np.ndarray]:
    """Return heavy-atom positions in the canonical local frame."""
    if restype in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[restype]

    template: dict[str, np.ndarray] = {}
    for atom in get_base_u(restype):  # type: ignore[union-attr]
        raw_name = atom.name
        if raw_name is None:
            continue
        if 'H' in raw_name or getattr(atom, 'element', None) in {'H', 'D'}:
            continue
        if raw_name in _TEMPLATE_RENAMES:
            atom_name = _TEMPLATE_RENAMES[raw_name]
        else:
            atom_name = raw_name.rstrip('AB')
        template[atom_name] = np.asarray(atom.position, dtype=np.float64)

    nu_rad = np.array([
        dihedral_rad(template[a0], template[a1], template[a2], template[a3])
        for a0, a1, a2, a3 in RING_TORSION_DEFS
    ])
    _TEMPLATE_TAU_M[restype] = float(np.sqrt(0.4 * float(np.dot(nu_rad, nu_rad))))
    _TEMPLATE_CACHE[restype] = template
    return template


def get_template_tau_m(restype: str, default: float = 0.611) -> float:
    get_template(restype)
    return float(_TEMPLATE_TAU_M.get(restype, default))


@functools.lru_cache(maxsize=None)
def get_template_tensors(device_str: str) -> dict[str, torch.Tensor]:
    """Return template geometry constants as float32 tensors on ``device``."""
    device = torch.device(device_str)
    restypes = ('A', 'C', 'G', 'T')

    def bond_length(template: dict[str, np.ndarray], a: str, b: str) -> float:
        return float(np.linalg.norm(template[a] - template[b]))

    def bond_angle(template: dict[str, np.ndarray], a: str, b: str, c: str) -> float:
        return bond_angle_numpy(template[a], template[b], template[c])

    def dihedral(template: dict[str, np.ndarray], a: str, b: str, c: str, d: str) -> float:
        return float(dihedral_rad(template[a], template[b], template[c], template[d]))

    keys_3d = ('c1', 'o4', 'c2_ref', 'c4_ref', 'chi_n', 'chi_c')
    keys_1d = (
        'bl_c2_c1',
        'ba_o4_c1_c2',
        'bl_c3_c2',
        'ba_c1_c2_c3',
        'bl_c4_c3',
        'ba_c2_c3_c4',
        'bl_c5_c4',
        'ba_c3_c4_c5',
        'psi_c5',
        'bl_o5_c5',
        'ba_c4_c5_o5',
        'bl_o3_c3',
        'ba_c4_c3_o3',
        'r_po3',
        'bond_p_o5',
        'bond_p_o3_inter',
        'tpl_o3_sep_p',
        'bl_op1',
        'bl_op2',
        'ang_op1',
        'ang_op2',
        'psi_op1',
        'psi_op2',
        'psi_o3',
        'psi_o3_ring',
        'bl_o4_c4',
        'ba_c1_o4_c4',
        'ba_o4_c4_c3',
        'ring_chiral_triple',
    )

    rows_3d = {key: [] for key in keys_3d}
    rows_1d = {key: [] for key in keys_1d}

    for restype in restypes:
        template = get_template(restype)
        rows_3d['c1'].append(template["C1'"])
        rows_3d['o4'].append(template["O4'"])
        rows_3d['c2_ref'].append(template["C2'"])
        rows_3d['c4_ref'].append(template["C4'"])
        if restype in ('A', 'G'):
            rows_3d['chi_n'].append(template['N9'])
            rows_3d['chi_c'].append(template['C4'])
        else:
            rows_3d['chi_n'].append(template['N1'])
            rows_3d['chi_c'].append(template['C2'])

        rows_1d['bl_c2_c1'].append(bond_length(template, "C2'", "C1'"))
        rows_1d['ba_o4_c1_c2'].append(bond_angle(template, "O4'", "C1'", "C2'"))
        rows_1d['bl_c3_c2'].append(bond_length(template, "C3'", "C2'"))
        rows_1d['ba_c1_c2_c3'].append(bond_angle(template, "C1'", "C2'", "C3'"))
        rows_1d['bl_c4_c3'].append(bond_length(template, "C4'", "C3'"))
        rows_1d['ba_c2_c3_c4'].append(bond_angle(template, "C2'", "C3'", "C4'"))
        rows_1d['bl_c5_c4'].append(bond_length(template, "C5'", "C4'"))
        rows_1d['ba_c3_c4_c5'].append(bond_angle(template, "C3'", "C4'", "C5'"))
        rows_1d['psi_c5'].append(dihedral(template, "O4'", "C3'", "C4'", "C5'"))
        rows_1d['bl_o5_c5'].append(bond_length(template, "O5'", "C5'"))
        rows_1d['ba_c4_c5_o5'].append(bond_angle(template, "C4'", "C5'", "O5'"))
        rows_1d['bl_o3_c3'].append(bond_length(template, "O3'", "C3'"))
        rows_1d['ba_c4_c3_o3'].append(bond_angle(template, "C4'", "C3'", "O3'"))
        po5_len = bond_length(template, 'P', "O5'")
        rows_1d['r_po3'].append(po5_len)
        rows_1d['bond_p_o5'].append(po5_len)
        rows_1d['bond_p_o3_inter'].append(po5_len)
        rows_1d['tpl_o3_sep_p'].append(bond_length(template, "O3'", 'P'))
        rows_1d['bl_op1'].append(bond_length(template, 'P', 'OP1'))
        rows_1d['bl_op2'].append(bond_length(template, 'P', 'OP2'))
        rows_1d['ang_op1'].append(bond_angle(template, "O5'", 'P', 'OP1'))
        rows_1d['ang_op2'].append(bond_angle(template, "O5'", 'P', 'OP2'))
        rows_1d['psi_op1'].append(dihedral(template, "O3'", "O5'", 'P', 'OP1'))
        rows_1d['psi_op2'].append(dihedral(template, "O3'", "O5'", 'P', 'OP2'))
        rows_1d['psi_o3'].append(dihedral(template, "C5'", "C4'", "C3'", "O3'"))
        rows_1d['psi_o3_ring'].append(dihedral(template, "O4'", "C4'", "C3'", "O3'"))
        rows_1d['bl_o4_c4'].append(bond_length(template, "O4'", "C4'"))
        rows_1d['ba_c1_o4_c4'].append(bond_angle(template, "C1'", "O4'", "C4'"))
        rows_1d['ba_o4_c4_c3'].append(bond_angle(template, "O4'", "C4'", "C3'"))
        v2 = template["C2'"] - template["O4'"]
        v3 = template["C3'"] - template["O4'"]
        v4 = template["C4'"] - template["O4'"]
        rows_1d['ring_chiral_triple'].append(float(np.dot(np.cross(v2, v3), v4)))

    out: dict[str, torch.Tensor] = {}
    for key, values in rows_3d.items():
        out[key] = torch.tensor(np.array(values), dtype=torch.float32, device=device)
    for key, values in rows_1d.items():
        out[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return out
