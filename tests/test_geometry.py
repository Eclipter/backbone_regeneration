"""Geometry regression tests (no DNA pipeline / pynamod imports)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from base2backbone.geometry.primitives import dihedral_rad_coords, nerf_place_coords
from base2backbone.torsion_constants import TAU_M_MAX, TAU_M_MIN
from base2backbone.geometry.backbone import (
    N_TORSIONS,
    TOR_BRIDGE_PHASE,
    TOR_GAMMA,
    TOR_CHI,
    TOR_PSEUDOROTATION_PHASE,
    build_backbone_local,
    local_to_world_points,
    nucleotide_torsions,
    _get_template,
    world_to_local_points,
)
from base2backbone.geometry import bridge_phase_from_points
from base2backbone.score_diffusion import decode_torsions, encode_torsions, wrap_angle


def _encode(theta: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
    return encode_torsions(theta, tau_m)


def _decode(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return decode_torsions(latent)


def _apply_inf_mask(sample_data: Data):
    """Apply the same chain-end torsion mask as inference uses on window batches."""
    WINDOW_SIZE = 3
    CHAIN_END_CLASS_5_PRIME = 1
    CHAIN_END_CLASS_3_PRIME = 2
    pos_mask = torch.ones(WINDOW_SIZE, N_TORSIONS, dtype=torch.bool)
    for i in range(WINDOW_SIZE):
        ce = sample_data.chain_end_class[i]
        if ce[CHAIN_END_CLASS_5_PRIME].item():
            pos_mask[i, TOR_BRIDGE_PHASE] = False
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


def test_bridge_phase_roundtrip_from_template():
    restype = 'G'
    xyz_cur, xyz_prev, xyz_next = _shifted_neighbor_tpl(restype)
    t, mask, tau_m_val, tau_m_valid = nucleotide_torsions(
        xyz_cur, xyz_prev, xyz_next, restype,
    )
    bridge_phase = float(t[TOR_BRIDGE_PHASE])
    gamma = float(t[TOR_GAMMA])
    chi = float(t[TOR_CHI])
    p_rad = float(t[TOR_PSEUDOROTATION_PHASE])

    mb = bool(mask[TOR_BRIDGE_PHASE])
    mg = bool(mask[TOR_GAMMA])
    mx, mp = bool(mask[TOR_CHI]), bool(mask[TOR_PSEUDOROTATION_PHASE])
    assert mb and mg and mx and mp
    assert tau_m_valid
    assert tau_m_val > 0.0
    assert all(np.isfinite((bridge_phase, gamma, chi, p_rad)))

    o3_prev_local = np.asarray(xyz_prev["O3'"], dtype=np.float64).reshape(3)
    bb = build_backbone_local(t, restype, o3_prev_local=o3_prev_local, tau_m=None)
    r_bridge = float(np.linalg.norm(xyz_cur['P'] - xyz_cur["O5'"]))
    got = bridge_phase_from_points(
        o3_prev_local,
        bb["O5'"],
        bb['P'],
        r_bridge,
        r_bridge,
    )
    d = float(np.arctan2(np.sin(got - bridge_phase), np.cos(got - bridge_phase)))
    assert abs(d) < 0.2


def test_tau_m_encode_decode_roundtrip():
    rng = torch.Generator().manual_seed(0)
    n = N_TORSIONS
    theta = torch.randn(4, n, generator=rng)
    tau = torch.exp(torch.randn(4, generator=rng) * 0.1 + 0.2).clamp(TAU_M_MIN, TAU_M_MAX)
    z = _encode(theta, tau)
    th2, t2 = _decode(z)
    assert torch.allclose(th2, wrap_angle(theta), atol=1e-5, rtol=1e-5)
    assert torch.allclose(t2, tau, atol=1e-4, rtol=1e-4)


def test_inference_positional_mask():
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
    expected_5p[TOR_BRIDGE_PHASE] = False
    expected_3p = internal_expected.copy()

    assert row_5prime == expected_5p
    assert row_internal == internal_expected
    assert row_3prime == expected_3p


def _consistent_neighbor_tpl(restype: str):
    tpl = _get_template(restype)
    xyz_prev = {"O3'": tpl["O3'"].copy()}
    xyz_next = {'P': tpl['P'].copy(), "O5'": tpl["O5'"].copy()}
    return tpl, xyz_prev, xyz_next


def test_gamma_path_matches_template_on_canonical_template():
    restype = 'C'
    tpl, xyz_prev, xyz_next = _consistent_neighbor_tpl(restype)
    t, mask, tau_m_val, tau_m_valid = nucleotide_torsions(tpl, xyz_prev, xyz_next, restype)
    gamma = float(t[TOR_GAMMA])
    chi = float(t[TOR_CHI])
    p_rad = float(t[TOR_PSEUDOROTATION_PHASE])

    mb = bool(mask[TOR_BRIDGE_PHASE])
    mg = bool(mask[TOR_GAMMA])
    mx, mp = bool(mask[TOR_CHI]), bool(mask[TOR_PSEUDOROTATION_PHASE])
    assert mb and mg and mx and mp
    assert tau_m_valid and tau_m_val > 0.0
    assert all(np.isfinite((gamma, chi, p_rad)))
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
    o5_ga = nerf_place_coords(c3, c4, c5, r_o5_c5, ang_c5, gamma - HP).cpu().numpy()

    g_meas = float(dihedral_rad_coords(c3, c4, c5, o5_ga).item())
    dg = float(np.arctan2(np.sin(g_meas - gamma), np.cos(g_meas - gamma)))
    assert abs(dg) < 0.06
    d_geom = float(np.linalg.norm(o5_ga - tpl["O5'"]))
    assert d_geom <= 0.1 + 1e-6


def test_world_local_roundtrip():
    torch.manual_seed(0)
    origin = torch.randn(2, 3)
    a = torch.randn(2, 3, 3)
    frame, _ = torch.linalg.qr(a)
    x_local = torch.randn(2, 3)
    x_w = local_to_world_points(x_local, origin, frame)
    x2 = world_to_local_points(x_w, origin, frame)
    assert (x2 - x_local).abs().max() < 1e-5


def test_window_builder_uses_same_frame_convention_as_utils():
    torch.manual_seed(1)
    o3w = torch.randn(5, 3)
    ok = torch.randn(5, 3)
    frame = torch.randn(5, 3, 3)
    frame, _ = torch.linalg.qr(frame)
    u = torch.einsum('bi,bij->bj', o3w - ok, frame)
    v = world_to_local_points(o3w, ok, frame)
    assert torch.allclose(u, v, atol=1e-5, rtol=1e-5)


def test_chi_changes_sugar_coordinates():
    tpl, xyz_prev, xyz_next = _shifted_neighbor_tpl('A')
    t, mask, tau_m_val, ok = nucleotide_torsions(tpl, xyz_prev, xyz_next, 'A')
    assert ok
    o3_prev = np.asarray(xyz_prev["O3'"], dtype=np.float64)
    t_lo = np.array(t, copy=True)
    t_hi = np.array(t, copy=True)
    t_lo[TOR_CHI] = -1.2
    t_hi[TOR_CHI] = 1.2
    b_lo = build_backbone_local(t_lo, 'A', o3_prev_local=o3_prev, tau_m=tau_m_val)
    b_hi = build_backbone_local(t_hi, 'A', o3_prev_local=o3_prev, tau_m=tau_m_val)
    d = float(np.linalg.norm(b_lo["C4'"] - b_hi["C4'"]))
    assert d > 1e-3


def test_measured_chi_purine_matches_input():
    restype = 'G'
    tpl, xyz_prev, xyz_next = _shifted_neighbor_tpl(restype)
    t, mask, tau_m_val, ok = nucleotide_torsions(tpl, xyz_prev, xyz_next, restype)
    assert ok
    o3_prev = np.asarray(xyz_prev["O3'"], dtype=np.float64)
    bb = build_backbone_local(t, restype, o3_prev_local=o3_prev, tau_m=tau_m_val)
    tpl_b = _get_template(restype)
    m = float(dihedral_rad_coords(bb["O4'"], bb["C1'"], tpl_b['N9'], tpl_b['C4']).item())
    d = float(np.arctan2(np.sin(m - t[TOR_CHI]), np.cos(m - t[TOR_CHI])))
    assert abs(d) < 0.05


def test_measured_chi_pyrimidine_matches_input():
    restype = 'C'
    tpl, xyz_prev, xyz_next = _shifted_neighbor_tpl(restype)
    t, mask, tau_m_val, ok = nucleotide_torsions(tpl, xyz_prev, xyz_next, restype)
    assert ok
    o3_prev = np.asarray(xyz_prev["O3'"], dtype=np.float64)
    bb = build_backbone_local(t, restype, o3_prev_local=o3_prev, tau_m=tau_m_val)
    tpl_b = _get_template(restype)
    m = float(dihedral_rad_coords(bb["O4'"], bb["C1'"], tpl_b['N1'], tpl_b['C2']).item())
    d = float(np.arctan2(np.sin(m - t[TOR_CHI]), np.cos(m - t[TOR_CHI])))
    assert abs(d) < 0.05


def test_world_local_matches_project_helpers():
    """Same row-vector convention as builder/inference: local = (world - origin) @ R."""
    torch.manual_seed(2)
    b = 5
    origin = torch.randn(b, 3)
    q, _ = torch.linalg.qr(torch.randn(b, 3, 3))
    world = torch.randn(b, 3)
    local = torch.einsum('bi,bij->bj', world - origin, q)
    assert torch.allclose(local, world_to_local_points(world, origin, q), atol=1e-5, rtol=1e-5)
    assert torch.allclose(world, local_to_world_points(local, origin, q), atol=1e-5, rtol=1e-5)


def test_world_local_world_roundtrip_explicit_matmul():
    """``local = (world - origin) @ R`` matches `torsion_geometry.local_to_world_points` round-trip."""
    torch.manual_seed(0)
    origin = torch.randn(4, 3)
    frame, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    x_w = torch.randn(4, 3)
    local = (x_w - origin) @ frame
    x_back = local @ frame.transpose(-1, -2) + origin
    assert torch.allclose(x_w, x_back, atol=1e-5, rtol=1e-5)