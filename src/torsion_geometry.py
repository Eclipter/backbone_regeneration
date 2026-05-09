"""Torsion definitions, sugar puckering (Altona–Sundaralingam / MDAnalysis-compatible), and angle wrapping."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

N_TORSIONS = 8
# 0:α 1:β 2:γ 3:δ 4:ε 5:ζ 6:χ 7:P

# True = circular (S¹); used for circular-aware diffusion
TORSION_IS_CIRCULAR = torch.tensor(
    [True, True, True, True, True, True, True, True],
    dtype=torch.bool,
)  # α β γ δ ε ζ χ P
TOR_ALPHA = 0
TOR_BETA = 1
TOR_GAMMA = 2
TOR_DELTA = 3
TOR_EPS = 4
TOR_ZETA = 5
TOR_CHI = 6
TOR_PUCKER_P = 7


def wrap_angle_rad(x):
    """Map angles to (-π, π]."""
    return np.arctan2(np.sin(x), np.cos(x))


def nerf_place(a, b, c, r, theta, psi):
    """Place D given prior atoms A–B–C, bond C–D length r, interior angle ∠(B–C–D)=theta, dihedral(A,B,C,D)=psi (rad)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    c = np.asarray(c, dtype=np.float64).reshape(3)
    ba = a - b
    bc = c - b
    bc_u = bc / (np.linalg.norm(bc) + 1e-12)
    n = np.cross(ba, bc_u)
    nn = np.linalg.norm(n)
    if nn < 1e-10:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / nn
    m = np.cross(n, bc_u)
    return c + r * (
        np.cos(np.pi - theta) * bc_u
        + np.sin(np.pi - theta) * (np.cos(psi) * n + np.sin(psi) * m)
    )


def dihedral_rad(p0, p1, p2, p3):
    """Signed dihedral angle (radians) for points p0–p3 (column vectors in R^3)."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_u = n1 / (np.linalg.norm(n1) + 1e-12)
    n2_u = n2 / (np.linalg.norm(n2) + 1e-12)
    m1 = np.cross(n1_u, b2 / (np.linalg.norm(b2) + 1e-12))
    x = np.dot(n1_u, n2_u)
    y = np.dot(m1, n2_u)
    return float(np.arctan2(y, x))


_RING_TORSION_DEFS = [
    ("C1'", "C2'", "C3'", "C4'"),
    ("C2'", "C3'", "C4'", "O4'"),
    ("C3'", "C4'", "O4'", "C1'"),
    ("C4'", "O4'", "C1'", "C2'"),
    ("O4'", "C1'", "C2'", "C3'"),
]
_RING_ANGLES = 2.0 * 2.0 * np.pi * np.arange(5, dtype=np.float64) / 5.0


def pseudorotation_P_rad_from_nus(nu_deg):
    """MDAnalysis `phase_as` phase angle in radians from five endocyclic torsions (degrees)."""
    nu = np.asarray(nu_deg, dtype=np.float64)
    B = np.dot(nu, np.sin(_RING_ANGLES)) * (-2.0 / 5.0)
    A = np.dot(nu, np.cos(_RING_ANGLES)) * (2.0 / 5.0)
    return float(np.arctan2(B, A))


def pucker_amplitude_rad(nu_deg, P_rad):
    """Amplitude τ_m (rad): LS fit ν_i ≈ τ_m cos(P + 2π i/5) with P fixed; five ν in degree (MDAnalysis order)."""
    nu = np.deg2rad(np.asarray(nu_deg, dtype=np.float64))
    idx = np.arange(5, dtype=np.float64)
    phases = P_rad + (2.0 * np.pi * idx / 5.0)
    c = np.cos(phases)
    den = float(np.dot(c, c)) + 1e-12
    tau = float(np.dot(nu, c) / np.sqrt(den))
    return abs(tau)


def _chi_quads(base_one_letter):
    if base_one_letter in ('A', 'G'):
        return ("O4'", "C1'", "N9", "C4")
    return ("O4'", "C1'", "N1", "C2")


def nucleotide_torsions_numpy(
    xyz_by_name_cur,
    xyz_by_name_prev,
    xyz_by_name_next,
    base_one_letter,
):
    """Return (torsions [N_TORSIONS], mask [N_TORSIONS], tau_m [rad], tau_m_valid).

    tau_m_valid is False when sugar ring ν torsions were incomplete.
    """

    def g(d, name):
        if d is None or name not in d:
            return None
        return np.asarray(d[name], dtype=np.float64).reshape(3)

    t = np.zeros(N_TORSIONS, dtype=np.float64)
    m = np.zeros(N_TORSIONS, dtype=bool)

    o3_prev = g(xyz_by_name_prev, "O3'")
    p_c = g(xyz_by_name_cur, 'P')
    o5 = g(xyz_by_name_cur, "O5'")
    c5 = g(xyz_by_name_cur, "C5'")
    c4 = g(xyz_by_name_cur, "C4'")
    c3 = g(xyz_by_name_cur, "C3'")
    o3_c = g(xyz_by_name_cur, "O3'")
    p_next = g(xyz_by_name_next, 'P')
    o5_next = g(xyz_by_name_next, "O5'")

    # α O3'(i-1)–P–O5'–C5'
    if all(x is not None for x in (o3_prev, p_c, o5, c5)):
        t[TOR_ALPHA] = dihedral_rad(o3_prev, p_c, o5, c5)
        m[TOR_ALPHA] = True
    # β P–O5'–C5'–C4'
    if all(x is not None for x in (p_c, o5, c5, c4)):
        t[TOR_BETA] = dihedral_rad(p_c, o5, c5, c4)
        m[TOR_BETA] = True
    # γ O5'–C5'–C4'–C3'
    if all(x is not None for x in (o5, c5, c4, c3)):
        t[TOR_GAMMA] = dihedral_rad(o5, c5, c4, c3)
        m[TOR_GAMMA] = True
    # δ C5'–C4'–C3'–O3'
    if all(x is not None for x in (c5, c4, c3, o3_c)):
        t[TOR_DELTA] = dihedral_rad(c5, c4, c3, o3_c)
        m[TOR_DELTA] = True
    # ε C4'–C3'–O3'–P(i+1)
    if all(x is not None for x in (c4, c3, o3_c, p_next)):
        t[TOR_EPS] = dihedral_rad(c4, c3, o3_c, p_next)
        m[TOR_EPS] = True
    # ζ C3'–O3'–P(i+1)–O5'(i+1)
    if all(x is not None for x in (c3, o3_c, p_next, o5_next)):
        t[TOR_ZETA] = dihedral_rad(c3, o3_c, p_next, o5_next)
        m[TOR_ZETA] = True

    o4 = g(xyz_by_name_cur, "O4'")
    c1 = g(xyz_by_name_cur, "C1'")
    a0, a1, a2, a3 = _chi_quads(base_one_letter)
    ap2 = g(xyz_by_name_cur, a2)
    ap3 = g(xyz_by_name_cur, a3)
    if all(x is not None for x in (o4, c1, ap2, ap3)):
        t[TOR_CHI] = dihedral_rad(o4, c1, ap2, ap3)
        m[TOR_CHI] = True

    nu_deg = []
    for a0n, a1n, a2n, a3n in _RING_TORSION_DEFS:
        pts = [g(xyz_by_name_cur, a0n), g(xyz_by_name_cur, a1n), g(xyz_by_name_cur, a2n), g(xyz_by_name_cur, a3n)]
        if not all(x is not None for x in pts):
            nu_deg = None
            break
        nu_deg.append(float(np.degrees(dihedral_rad(pts[0], pts[1], pts[2], pts[3]))))
    tau_m_val = 0.0
    if nu_deg is not None:
        nu_arr = np.asarray(nu_deg, dtype=np.float64)
        P_rad = pseudorotation_P_rad_from_nus(nu_arr)
        t[TOR_PUCKER_P] = P_rad
        m[TOR_PUCKER_P] = True
        tau_m_val = float(pucker_amplitude_rad(nu_arr, P_rad))
    tau_m_valid = nu_deg is not None
    return t, m, tau_m_val, tau_m_valid


# Cache template atom dicts per restype to avoid repeated pynamod calls.
_template_cache: dict = {}


_template_tau_m: dict = {}  # per-restype canonical ring amplitude τm_AS (radians)


def _get_template(restype: str) -> 'dict[str, np.ndarray]':
    """Return heavy-atom positions (local frame, Å) for the canonical nucleotide."""
    if restype in _template_cache:
        return _template_cache[restype]
    from pynamod.atomic_analysis.nucleotides_parser import \
        get_base_u  # lazy import
    _RENAMES = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}
    tpl: dict = {}
    for atom in get_base_u(restype):  # type: ignore[union-attr]
        if 'H' in atom.name or getattr(atom, 'element', None) in {'H', 'D'}:
            continue
        nm = _RENAMES.get(atom.name, atom.name.rstrip('AB'))
        tpl[nm] = np.asarray(atom.position, dtype=np.float64)
    # Compute canonical ring amplitude τm_AS = sqrt(2/5 · Σ νₖ²) from template.
    # This is consistent with the A-S inverse formula νₖ = τm cos(P + offsets[k]).
    nu_rad = np.array([
        dihedral_rad(tpl[q[0]], tpl[q[1]], tpl[q[2]], tpl[q[3]])
        for q in _RING_TORSION_DEFS
    ])
    _template_tau_m[restype] = float(np.sqrt(0.4 * float(np.dot(nu_rad, nu_rad))))
    _template_cache[restype] = tpl
    return tpl


def build_backbone_from_torsions(
    torsions: np.ndarray,
    restype: str,
    o3_prev_local: Optional[np.ndarray] = None,
    base_atoms_local: Optional[Dict[str, np.ndarray]] = None,
    tau_m: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Build backbone atom positions in the local nucleotide frame from 8 torsions.

    Parameters
    ----------
    torsions : np.ndarray, shape [N_TORSIONS]
        Order: [α, β, γ, δ, ε, ζ, χ, P], radians.
    restype : str
        One-letter nucleotide code: 'A' | 'C' | 'G' | 'T'.
    o3_prev_local : optional (3,)
        Previous nucleotide's O3' in this residue's local frame; enables α-based P placement.
    base_atoms_local : optional name -> (3,)
        Base heavy atoms (e.g. N9/C4 purines, N1/C2 pyrimidines) in local frame for χ refinement.
    tau_m : optional float
        Predicted puckering amplitude (rad); fallback: template amplitude.

    Returns
    -------
    dict[str, np.ndarray]
        atom_name -> xyz (Å) in the local nucleotide frame.
    """
    tpl = _get_template(restype)

    def _blen(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _bangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba, bc = a - b, c - b
        cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
        return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    # Extract relevant torsion angles
    alpha = float(torsions[TOR_ALPHA])
    beta = float(torsions[TOR_BETA])
    gamma = float(torsions[TOR_GAMMA])
    delta = float(torsions[TOR_DELTA])
    chi = float(torsions[TOR_CHI])
    P_rad = float(torsions[TOR_PUCKER_P])
    if tau_m is not None:
        _TAU_DNA = float(tau_m)
    else:
        _TAU_DNA = float(_template_tau_m.get(restype, 0.611))

    # _RING_TORSION_DEFS starts at standard ν₂ (C1'-C2'-C3'-C4').
    # Mapping: our k → A-S index: 0→ν₂, 1→ν₃, 2→ν₄, 3→ν₀, 4→ν₁.
    _NU_OFFSETS = np.array([0.0, 4.0, 8.0, 2.0, 6.0]) * np.pi / 5.0
    nus = _TAU_DNA * np.cos(P_rad + _NU_OFFSETS)
    nu0, _nu1, _nu2, nu3, nu4 = nus

    # Anchor atoms — fixed in local frame from canonical template
    c1 = tpl["C1'"].copy()
    o4 = tpl["O4'"].copy()
    c4_ref = tpl["C4'"].copy()

    # nerf_place(A, B, C, r, θ, ψ) satisfies dihedral(A,B,C,D) = ψ + π/2
    _HP = np.pi / 2.0

    # ── Sugar ring ──────────────────────────────────────────────────────────
    c2 = nerf_place(
        c4_ref, o4, c1,
        _blen(tpl["C2'"], c1),
        _bangle(o4, c1, tpl["C2'"]),
        nu3 - _HP,
    )
    c3 = nerf_place(
        o4, c1, c2,
        _blen(tpl["C3'"], tpl["C2'"]),
        _bangle(c1, tpl["C2'"], tpl["C3'"]),
        nu4 - _HP,
    )
    c4 = nerf_place(
        c1, c2, c3,
        _blen(tpl["C4'"], tpl["C3'"]),
        _bangle(tpl["C2'"], tpl["C3'"], tpl["C4'"]),
        nu0 - _HP,
    )

    psi_c5 = dihedral_rad(o4, tpl["C3'"], tpl["C4'"], tpl["C5'"])
    c5 = nerf_place(
        o4, c3, c4,
        _blen(tpl["C5'"], tpl["C4'"]),
        _bangle(tpl["C3'"], tpl["C4'"], tpl["C5'"]),
        psi_c5 - _HP,
    )

    o5_gamma = nerf_place(
        c3, c4, c5,
        _blen(tpl["O5'"], tpl["C5'"]),
        _bangle(tpl["C4'"], tpl["C5'"], tpl["O5'"]),
        gamma - _HP,
    )

    o3 = nerf_place(
        c5, c4, c3,
        _blen(tpl["O3'"], tpl["C3'"]),
        _bangle(tpl["C4'"], tpl["C3'"], tpl["O3'"]),
        delta - _HP,
    )

    out: dict[str, np.ndarray] = {
        "C1'": c1.copy(),
        "C2'": c2,
        "C3'": c3,
        "C4'": c4,
        "C5'": c5,
        "O4'": o4.copy(),
        "O3'": o3,
    }

    p_built: Optional[np.ndarray] = None
    o3p: Optional[np.ndarray] = None
    r_po3 = 0.0
    o5 = o5_gamma.copy()

    if o3_prev_local is not None:
        o3p = np.asarray(o3_prev_local, dtype=np.float64).reshape(3)
        r_po3 = _blen(tpl['P'], tpl["O3'"])
        theta_po3 = _bangle(tpl["O5'"], tpl["O3'"], tpl['P'])
        # ψ = −α aligns dihedral(O3′, P, O5′, C5′) with α given nerf_place's +π/2 offset.
        p_built = nerf_place(
            c5, o5_gamma, o3p, r_po3, theta_po3, -alpha - _HP,
        )
        r_o5p = _blen(tpl["O5'"], tpl['P'])
        ang_at_p = _bangle(tpl["C5'"], tpl['P'], tpl["O5'"])
        o5_beta = nerf_place(c4, c5, p_built, r_o5p, ang_at_p, beta - _HP)
        o5 = 0.5 * (o5_gamma + o5_beta)

    out["O5'"] = o5

    if base_atoms_local is not None:
        n_atom = base_atoms_local.get('N9')
        if n_atom is None:
            n_atom = base_atoms_local.get('N1')
        c_atom = base_atoms_local.get('C4')
        if c_atom is None:
            c_atom = base_atoms_local.get('C2')
        if n_atom is not None and c_atom is not None:
            n_atom = np.asarray(n_atom, dtype=np.float64).reshape(3)
            c_atom = np.asarray(c_atom, dtype=np.float64).reshape(3)
            n_key = 'N9' if restype in ('A', 'G') else 'N1'
            r_c1o4 = _blen(tpl["C1'"], tpl["O4'"])
            ang_c1 = _bangle(tpl[n_key], tpl["O4'"], tpl["C1'"])
            c1_chi = nerf_place(c_atom, n_atom, o4, r_c1o4, ang_c1, chi - _HP)
            out["C1'"] = 0.5 * (c1 + c1_chi)

    if p_built is not None:
        o5_f = out["O5'"]
        u_op2 = tpl["OP2"] - tpl['P']
        u_op1 = tpl["OP1"] - tpl['P']
        c_op2_ref = p_built + u_op2
        c_op1_ref = p_built + u_op1
        r_p_op1 = _blen(tpl['P'], tpl["OP1"])
        r_p_op2 = _blen(tpl['P'], tpl["OP2"])
        ang_op1 = _bangle(tpl["O5'"], tpl['P'], tpl["OP1"])
        ang_op2 = _bangle(tpl["O5'"], tpl['P'], tpl["OP2"])
        psi_op1 = dihedral_rad(tpl["O5'"], tpl['P'], tpl["OP2"], tpl["OP1"])
        psi_op2 = dihedral_rad(tpl["O5'"], tpl['P'], tpl["OP1"], tpl["OP2"])
        op1 = nerf_place(o5_f, p_built, c_op2_ref, r_p_op1, ang_op1, psi_op1 - _HP)
        op2 = nerf_place(o5_f, p_built, c_op1_ref, r_p_op2, ang_op2, psi_op2 - _HP)
        out['P'] = p_built
        out['OP1'] = op1
        out['OP2'] = op2
    else:
        for nm in ('P', 'OP1', 'OP2'):
            if nm in tpl:
                out[nm] = tpl[nm].copy()

    return out
