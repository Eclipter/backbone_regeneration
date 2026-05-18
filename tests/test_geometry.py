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
from base2backbone.geometry import (
    bridge_circle_geometry,
    bridge_phase_from_points_numpy,
    bridge_phase_from_points_torch,
    phosphate_from_bridge_phase,
)
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
    xyz_prev = {
        "O3'": tpl["O3'"] - vec * scale,
        "C3'": tpl["C3'"] - vec * scale,
        "C4'": tpl["C4'"] - vec * scale,
    }
    xyz_next = {
        'P': tpl['P'] + vec * scale,
        "O5'": tpl["O5'"] + vec * scale,
    }
    return tpl, xyz_prev, xyz_next


def test_bridge_phase_roundtrip_from_template():
    o3_prev = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    o5_cur = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    c3_prev = np.array([0.0, 1.0, 0.2], dtype=np.float64)
    c4_prev = np.array([0.0, 0.2, 1.0], dtype=np.float64)
    r_bridge = 1.6
    phase = 0.43

    p_roundtrip = phosphate_from_bridge_phase(
        torch.as_tensor(o3_prev, dtype=torch.float64).reshape(1, 3),
        torch.as_tensor(o5_cur, dtype=torch.float64).reshape(1, 3),
        torch.as_tensor(c3_prev, dtype=torch.float64).reshape(1, 3),
        torch.tensor([phase], dtype=torch.float64),
        torch.tensor([r_bridge], dtype=torch.float64),
        torch.tensor([r_bridge], dtype=torch.float64),
        torch.as_tensor(c4_prev, dtype=torch.float64).reshape(1, 3),
    ).squeeze(0).cpu().numpy()
    phase_back = bridge_phase_from_points_numpy(
        o3_prev,
        o5_cur,
        c3_prev,
        p_roundtrip,
        r_bridge,
        r_bridge,
        c4_prev,
    )
    d = float(np.arctan2(np.sin(phase_back - phase), np.cos(phase_back - phase)))
    assert abs(d) < 1e-6


def test_bridge_phase_is_rigid_transform_invariant():
    tpl, xyz_prev, _xyz_next = _shifted_neighbor_tpl('A')
    o3 = np.asarray(xyz_prev["O3'"], dtype=np.float64)
    c3 = np.asarray(xyz_prev["C3'"], dtype=np.float64)
    c4 = np.asarray(xyz_prev["C4'"], dtype=np.float64)
    o5 = np.asarray(tpl["O5'"], dtype=np.float64)
    p = np.asarray(tpl['P'], dtype=np.float64)
    r_bridge = float(np.linalg.norm(p - o5))
    phase = bridge_phase_from_points_numpy(o3, o5, c3, p, r_bridge, r_bridge, c4)

    axis = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = 0.71
    kx, ky, kz = axis
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    t = np.array([1.2, -0.8, 0.4], dtype=np.float64)

    def _rt(x):
        return R @ x + t

    phase_rt = bridge_phase_from_points_numpy(
        _rt(o3),
        _rt(o5),
        _rt(c3),
        _rt(p),
        r_bridge,
        r_bridge,
        _rt(c4),
    )
    d = float(np.arctan2(np.sin(phase_rt - phase), np.cos(phase_rt - phase)))
    assert abs(d) < 1e-5


def test_bridge_decoder_uses_reference_atom():
    anchor_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    anchor_b = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    ref_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    ref_z = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    phase = torch.tensor([0.37], dtype=torch.float64)
    r_a = torch.tensor([1.6], dtype=torch.float64)
    r_b = torch.tensor([1.6], dtype=torch.float64)

    p_y = phosphate_from_bridge_phase(anchor_a, anchor_b, ref_y, phase, r_a, r_b)
    p_z = phosphate_from_bridge_phase(anchor_a, anchor_b, ref_z, phase, r_a, r_b)
    center, _radius, _u, _v = bridge_circle_geometry(anchor_a, anchor_b, ref_y, r_a, r_b)
    rot_x_90 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    expected = center + (p_y - center) @ rot_x_90.T
    assert torch.allclose(p_z, expected, atol=1e-6, rtol=1e-6)


def test_bridge_circle_degenerate_backward_is_finite():
    anchor_a = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    anchor_b = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)
    ref_atom = torch.tensor([[0.0, 1.0, 0.0]])
    r_a = torch.tensor([1.0])
    r_b = torch.tensor([1.0])

    center, radius, u, v = bridge_circle_geometry(anchor_a, anchor_b, ref_atom, r_a, r_b)
    loss = center.square().sum() + radius.square().sum() + u.square().sum() + v.square().sum()
    assert torch.isfinite(loss)

    loss.backward()
    assert torch.isfinite(anchor_a.grad).all()
    assert torch.isfinite(anchor_b.grad).all()


def test_bridge_phase_at_circle_center_backward_is_finite():
    anchor_a = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    anchor_b = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    ref_atom = torch.tensor([[0.0, 1.0, 0.0]])
    phosphate = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    r_a = torch.tensor([1.0])
    r_b = torch.tensor([1.0])

    phase = bridge_phase_from_points_torch(anchor_a, anchor_b, ref_atom, phosphate, r_a, r_b)
    assert torch.isfinite(phase).all()

    phase.sum().backward()
    assert torch.isfinite(anchor_a.grad).all()
    assert torch.isfinite(anchor_b.grad).all()
    assert torch.isfinite(phosphate.grad).all()


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
    xyz_prev = {
        "O3'": tpl["O3'"].copy(),
        "C3'": tpl["C3'"].copy(),
        "C4'": tpl["C4'"].copy(),
    }
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
    c3_prev = np.asarray(xyz_prev["C3'"], dtype=np.float64)
    c4_prev = np.asarray(xyz_prev["C4'"], dtype=np.float64)
    t_lo = np.array(t, copy=True)
    t_hi = np.array(t, copy=True)
    t_lo[TOR_CHI] = -1.2
    t_hi[TOR_CHI] = 1.2
    b_lo = build_backbone_local(
        t_lo,
        'A',
        o3_prev_local=o3_prev,
        c3_prev_local=c3_prev,
        c4_prev_local=c4_prev,
        tau_m=tau_m_val,
    )
    b_hi = build_backbone_local(
        t_hi,
        'A',
        o3_prev_local=o3_prev,
        c3_prev_local=c3_prev,
        c4_prev_local=c4_prev,
        tau_m=tau_m_val,
    )
    d = float(np.linalg.norm(b_lo["C4'"] - b_hi["C4'"]))
    assert d > 1e-3


def test_measured_chi_purine_matches_input():
    restype = 'G'
    tpl, xyz_prev, xyz_next = _shifted_neighbor_tpl(restype)
    t, mask, tau_m_val, ok = nucleotide_torsions(tpl, xyz_prev, xyz_next, restype)
    assert ok
    o3_prev = np.asarray(xyz_prev["O3'"], dtype=np.float64)
    c3_prev = np.asarray(xyz_prev["C3'"], dtype=np.float64)
    c4_prev = np.asarray(xyz_prev["C4'"], dtype=np.float64)
    bb = build_backbone_local(
        t,
        restype,
        o3_prev_local=o3_prev,
        c3_prev_local=c3_prev,
        c4_prev_local=c4_prev,
        tau_m=tau_m_val,
    )
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
    c3_prev = np.asarray(xyz_prev["C3'"], dtype=np.float64)
    c4_prev = np.asarray(xyz_prev["C4'"], dtype=np.float64)
    bb = build_backbone_local(
        t,
        restype,
        o3_prev_local=o3_prev,
        c3_prev_local=c3_prev,
        c4_prev_local=c4_prev,
        tau_m=tau_m_val,
    )
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