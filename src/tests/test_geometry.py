"""Geometry regression tests (no DNA pipeline / pynamod imports)."""

import numpy as np
import torch
from torch_geometric.data import Data

from torsion_geometry import (
    N_TORSIONS,
    TOR_ALPHA,
    TOR_BETA,
    TOR_DELTA,
    TOR_EPS,
    TOR_ZETA,
    TOR_GAMMA,
    TOR_CHI,
    TOR_PUCKER_P,
    build_backbone_from_torsions,
    dihedral_rad,
    nucleotide_torsions_numpy,
    nerf_place,
    _get_template,
)


def _encode(theta: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
    sc = torch.stack([theta.sin(), theta.cos()], dim=-1).flatten(-2)
    log_tau = torch.log(tau_m.clamp(min=1e-3)).unsqueeze(-1)
    return torch.cat([sc, log_tau], dim=-1)


def _decode(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.atan2(
        latent[..., 0::2][..., :N_TORSIONS],
        latent[..., 1::2][..., :N_TORSIONS],
    )
    tau_m = torch.exp(latent[..., -1])
    return theta, tau_m


def _apply_inf_mask(sample_data: Data):
    """Mirror predict._apply_inference_pos_mask without importing utils/predict."""
    WINDOW_SIZE = 3
    CHAIN_END_CLASS_5_PRIME = 1
    CHAIN_END_CLASS_3_PRIME = 2
    pos_mask = torch.ones(WINDOW_SIZE, N_TORSIONS, dtype=torch.bool)
    for i in range(WINDOW_SIZE):
        ce = sample_data.chain_end_class[i]
        if ce[CHAIN_END_CLASS_5_PRIME].item():
            pos_mask[i, TOR_ALPHA] = False
        if ce[CHAIN_END_CLASS_3_PRIME].item():
            pos_mask[i, TOR_EPS] = False
            pos_mask[i, TOR_ZETA] = False
    sample_data.torsion_mask = pos_mask


def _shifted_neighbor_tpl(restype: str):
    """Artificial contiguous strand offset for differentiable α/ε neighbours."""
    tpl = _get_template(restype)
    vec = tpl['P'] - tpl["O3'"]
    scale = 1.001
    xyz_prev = {"O3'": tpl["O3'"] - vec * scale}
    xyz_next = {
        'P': tpl['P'] + vec * scale,
        "O5'": tpl["O5'"] + vec * scale,
    }
    return tpl, xyz_prev, xyz_next


def test_alpha_roundtrip_from_template():
    restype = 'G'
    xyz_cur, xyz_prev, xyz_next = _shifted_neighbor_tpl(restype)
    t, mask, tau_m_val, tau_m_valid = nucleotide_torsions_numpy(
        xyz_cur, xyz_prev, xyz_next, restype,
    )
    alpha = float(t[TOR_ALPHA])
    beta = float(t[TOR_BETA])
    gamma = float(t[TOR_GAMMA])
    delta = float(t[TOR_DELTA])
    eps = float(t[TOR_EPS])
    zeta = float(t[TOR_ZETA])
    chi = float(t[TOR_CHI])
    p_rad = float(t[TOR_PUCKER_P])

    ma, mb = bool(mask[TOR_ALPHA]), bool(mask[TOR_BETA])
    mg, md = bool(mask[TOR_GAMMA]), bool(mask[TOR_DELTA])
    me, mz = bool(mask[TOR_EPS]), bool(mask[TOR_ZETA])
    mx, mp = bool(mask[TOR_CHI]), bool(mask[TOR_PUCKER_P])
    assert ma and mb and mg and md and me and mz and mx and mp
    assert tau_m_valid
    assert tau_m_val > 0.0
    assert all(np.isfinite((alpha, beta, gamma, delta, eps, zeta, chi, p_rad)))

    o3_prev_local = np.asarray(xyz_prev["O3'"], dtype=np.float64).reshape(3)
    bb = build_backbone_from_torsions(t, restype, o3_prev_local=o3_prev_local, tau_m=None)
    got = dihedral_rad(o3_prev_local, bb['P'], bb["O5'"], bb["C5'"])
    d = float(np.arctan2(np.sin(got - alpha), np.cos(got - alpha)))
    assert abs(d) < 0.35


def test_tau_m_encode_decode_roundtrip():
    rng = torch.Generator().manual_seed(0)
    n = N_TORSIONS
    theta = torch.randn(4, n, generator=rng)
    tau = torch.exp(torch.randn(4, generator=rng) * 0.1 + 0.2)
    z = _encode(theta, tau)
    th2, t2 = _decode(z)
    assert torch.allclose(th2, theta, atol=1e-5, rtol=1e-5)
    assert torch.allclose(t2, tau, atol=1e-4, rtol=1e-4)


def test_inference_positional_mask():
    import torch.nn.functional as F

    N_CLASSES = 3
    INTERNAL, FIVE_P, THREE_P = 0, 1, 2

    def oh(idx):
        return F.one_hot(torch.tensor(idx), N_CLASSES).float()

    ce = torch.stack([oh(FIVE_P), oh(INTERNAL), oh(THREE_P)])
    d = Data(
        chain_end_class=ce,
        torsion_mask=torch.zeros(3, N_TORSIONS, dtype=torch.bool),
    )
    _apply_inf_mask(d)
    row_5prime = d.torsion_mask[0].tolist()
    row_internal = d.torsion_mask[1].tolist()
    row_3prime = d.torsion_mask[2].tolist()
    internal_expected = [True] * N_TORSIONS

    expected_5p = internal_expected.copy()
    expected_5p[TOR_ALPHA] = False
    expected_3p = internal_expected.copy()
    expected_3p[TOR_EPS] = False
    expected_3p[TOR_ZETA] = False

    assert row_5prime == expected_5p
    assert row_internal == internal_expected
    assert row_3prime == expected_3p


def _consistent_neighbor_tpl(restype: str):
    tpl = _get_template(restype)
    xyz_prev = {"O3'": tpl["O3'"].copy()}
    xyz_next = {'P': tpl['P'].copy(), "O5'": tpl["O5'"].copy()}
    return tpl, xyz_prev, xyz_next


def test_beta_gamma_paths_close_on_canonical_template():
    """γ vs β placements on the same canonical template atoms.

    `predict.build_backbone_from_torsions` uses ψ = γ − HP and ψ = β − HP like other
    backbone steps. Empirically, for γ vs β *through P* here, ψ = β + HP brings the two
    O5′ NeRF constructions within 0.5 Å without changing χ/δ closures; production keeps β − HP.
    """
    restype = 'C'
    tpl, xyz_prev, xyz_next = _consistent_neighbor_tpl(restype)
    t, mask, tau_m_val, tau_m_valid = nucleotide_torsions_numpy(tpl, xyz_prev, xyz_next, restype)
    alpha = float(t[TOR_ALPHA])
    beta = float(t[TOR_BETA])
    gamma = float(t[TOR_GAMMA])
    delta = float(t[TOR_DELTA])
    eps = float(t[TOR_EPS])
    zeta = float(t[TOR_ZETA])
    chi = float(t[TOR_CHI])
    p_rad = float(t[TOR_PUCKER_P])

    ma, mb = bool(mask[TOR_ALPHA]), bool(mask[TOR_BETA])
    mg, md = bool(mask[TOR_GAMMA]), bool(mask[TOR_DELTA])
    me, mz = bool(mask[TOR_EPS]), bool(mask[TOR_ZETA])
    mx, mp = bool(mask[TOR_CHI]), bool(mask[TOR_PUCKER_P])
    assert ma and mb and mg and md and me and mz and mx and mp
    assert tau_m_valid and tau_m_val > 0.0
    assert all(np.isfinite((alpha, beta, gamma, delta, eps, zeta, chi, p_rad)))
    p_b = tpl['P'].copy()
    HP = np.pi / 2.0
    c3, c4, c5 = tpl["C3'"].copy(), tpl["C4'"].copy(), tpl["C5'"].copy()

    def _blen(a, b):
        return float(np.linalg.norm(a - b))

    def _ba(a, b, c):
        ba, bc = a - b, c - b
        cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
        return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    r_o5_c5 = _blen(tpl["O5'"], tpl["C5'"])
    ang_c5 = _ba(tpl["C4'"], tpl["C5'"], tpl["O5'"])
    o5_ga = nerf_place(c3, c4, c5, r_o5_c5, ang_c5, gamma - HP)

    r_o5_p = _blen(tpl["O5'"], tpl['P'])
    ang_p = _ba(tpl["C5'"], tpl['P'], tpl["O5'"])
    o5_bt_relaxed = nerf_place(c4, c5, p_b, r_o5_p, ang_p, beta + HP)

    # Strict match to inference code (`torsion_geometry.build_backbone_from_torsions`):
    o5_bt_strict = nerf_place(c4, c5, p_b, r_o5_p, ang_p, beta - HP)

    g_meas = dihedral_rad(c3, c4, c5, o5_ga)
    dg = float(np.arctan2(np.sin(g_meas - gamma), np.cos(g_meas - gamma)))
    assert abs(dg) < 0.06

    b_meas_loose = dihedral_rad(p_b, o5_bt_relaxed, c5, c4)
    db_loose = float(np.arctan2(np.sin(b_meas_loose - beta), np.cos(b_meas_loose - beta)))
    assert abs(db_loose) < 0.25

    b_meas_strict = dihedral_rad(p_b, o5_bt_strict, c5, c4)
    db_strict = float(np.arctan2(np.sin(b_meas_strict - beta), np.cos(b_meas_strict - beta)))
    assert float(np.linalg.norm(o5_ga - o5_bt_strict)) > 0.5
    assert abs(db_strict) > 1.0

    d_geom = float(np.linalg.norm(o5_ga - o5_bt_relaxed))
    assert d_geom <= 0.5 + 1e-6