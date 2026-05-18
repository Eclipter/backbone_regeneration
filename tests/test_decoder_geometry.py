"""Tests for analytic sugar ring closure, stereo exocyclic atoms, and multi-torsion phosphate decoder."""

from pathlib import Path

import numpy as np
import pytest
import torch

from base2backbone.data import BACKBONE_ATOMS
from base2backbone.bridge_closure import canonical_two_residue_bridge_positions
from base2backbone.geometry.backbone import (
    _RING_TORSION_DEFS, N_TORSIONS, _get_template, _get_template_tensors,
    add_exocyclic_sugar_atoms, add_o5_from_gamma,
    build_backbone_from_torsions, build_sugar_ring_closed_form,
    build_batch_window_backbone_from_torsions, build_window_backbone_from_torsions,
    close_phosphate_bridge_multi, dihedral_rad,
    nucleotide_torsions, nus_rad_from_phase_and_amplitude,
    wrap_dihedral_diff)
from base2backbone.geometry import bridge_phase_from_points_torch
from base2backbone.torsion_constants import TOR_BRIDGE_PHASE


def _assert_window_bb_finiteness_design(bb: torch.Tensor) -> None:
    """Row 0: no upstream bridge ⇒ P / OPs may stay NaN; all other backbone rows/columns finite."""
    ph = frozenset({
        BACKBONE_ATOMS.index('P'),
        BACKBONE_ATOMS.index('OP1'),
        BACKBONE_ATOMS.index('OP2'),
    })
    assert bb.dim() == 4 and bb.shape[-1] == 3
    for j in range(bb.shape[2]):
        if j in ph:
            assert torch.all(torch.isfinite(bb[:, 1:, j, :])), (
                'Phosphate/inter-bridge atoms must be finite for k≥1.'
            )
        else:
            assert torch.all(torch.isfinite(bb[:, :, j, :])), (
                f'Non-terminal-phosphate backbone column {BACKBONE_ATOMS[j]} includes NaNs.'
            )


def _template_tc(dev):
    return _get_template_tensors(str(dev))


def _template_delta(dev, ri: torch.Tensor) -> torch.Tensor:
    return _template_tc(dev)['psi_o3'][ri.long()]


@pytest.fixture
def device():
    return torch.device('cpu')


def test_chi_rotation_preserves_sugar_ring_internal_torsions(device):
    """Same P, τ_m, restype; different χ: endocyclic ν₀…ν₄ match; ring atom coords change."""
    ri = torch.tensor([2], device=device, dtype=torch.long)
    P = torch.tensor([0.17], device=device)
    tau_m = torch.tensor([0.33], device=device)
    chi_a = torch.tensor([-0.55], device=device)
    chi_b = torch.tensor([0.78], device=device)
    ring_a = build_sugar_ring_closed_form(chi_a, P, tau_m, ri)
    ring_b = build_sugar_ring_closed_form(chi_b, P, tau_m, ri)

    def _ring_nus(ring: dict) -> torch.Tensor:
        return torch.stack(
            [
                dihedral_rad(
                    ring[a0n].reshape(1, 3),
                    ring[a1n].reshape(1, 3),
                    ring[a2n].reshape(1, 3),
                    ring[a3n].reshape(1, 3),
                ).reshape(1)
                for a0n, a1n, a2n, a3n in _RING_TORSION_DEFS
            ],
            dim=-1,
        )

    nu_a = _ring_nus(ring_a)
    nu_b = _ring_nus(ring_b)
    max_nu = wrap_dihedral_diff(nu_a, nu_b).abs().max().item()
    assert max_nu < 1e-3

    for nm in ("O4'", "C2'", "C3'", "C4'"):
        assert not torch.allclose(
            ring_a[nm].reshape(3),
            ring_b[nm].reshape(3),
            atol=1e-4,
            rtol=1e-4,
        )


def test_chi_scan_matches_measured_dihedral_preserves_pucker_and_sugar_metric(device):
    """Same P and τ_m; varying χ: measured χ matches input; ν/Puncher from ring atoms unchanged."""
    ri = torch.tensor([2], device=device, dtype=torch.long)
    P = torch.tensor([0.18], device=device)
    tau_m = torch.tensor([0.31], device=device)
    chis = torch.linspace(-0.95, 0.95, 5, device=device)
    tgt_nu = nus_rad_from_phase_and_amplitude(P, tau_m)
    tc = _get_template_tensors(str(device))
    n_atom = tc['chi_n'][ri].reshape(1, 3)
    c_atom = tc['chi_c'][ri].reshape(1, 3)
    dist_ref = None
    for chi in chis:
        ring = build_sugar_ring_closed_form(chi.unsqueeze(0), P, tau_m, ri)
        atoms = add_exocyclic_sugar_atoms(ring, _template_delta(device, ri), restype_indices=ri)
        mchi = dihedral_rad(
            atoms["O4'"].reshape(1, 3),
            atoms["C1'"].reshape(1, 3),
            n_atom,
            c_atom,
        )
        assert wrap_dihedral_diff(mchi, chi.unsqueeze(0)).abs().item() < 0.02
        nu_meas = []
        for a0n, a1n, a2n, a3n in _RING_TORSION_DEFS:
            nu_meas.append(
                dihedral_rad(
                    atoms[a0n].reshape(1, 3),
                    atoms[a1n].reshape(1, 3),
                    atoms[a2n].reshape(1, 3),
                    atoms[a3n].reshape(1, 3),
                ),
            )
        nu_st = torch.cat(nu_meas, dim=-1)
        err_nu = wrap_dihedral_diff(nu_st, tgt_nu).abs().max().item()
        assert err_nu < 0.12
        sugar_names = ["O4'", "C2'", "C3'", "C4'", "C5'", "O3'"]
        pts = torch.stack([atoms[nm].reshape(3) for nm in sugar_names], dim=0)
        dmat = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
        if dist_ref is None:
            dist_ref = dmat
        else:
            assert torch.allclose(dmat, dist_ref, atol=5e-4, rtol=2e-3)


def test_pseudorotation_torsions_consistent(device):
    restype_idx = torch.tensor([0, 1, 2, 3], device=device)
    P = torch.linspace(-1.0, 1.0, 4, device=device)
    tau = torch.full((4,), 0.35, device=device)
    chi = torch.zeros(4, device=device)
    ring = build_sugar_ring_closed_form(chi, P, tau, restype_idx)
    tgt = nus_rad_from_phase_and_amplitude(P, tau)
    for i, (a0, a1, a2, a3) in enumerate(
        [
            ("C1'", "C2'", "C3'", "C4'"),
            ("C2'", "C3'", "C4'", "O4'"),
            ("C3'", "C4'", "O4'", "C1'"),
            ("C4'", "O4'", "C1'", "C2'"),
            ("O4'", "C1'", "C2'", "C3'"),
        ],
    ):
        m = dihedral_rad(
            ring[a0], ring[a1], ring[a2], ring[a3],
        )
        err = wrap_dihedral_diff(m, tgt[:, i]).abs().max().item()
        assert err < 0.12, i


def test_sugar_ring_is_closed(device):
    torch.manual_seed(1)
    for _ in range(8):
        restype_idx = torch.randint(0, 4, (16,), device=device)
        P = (torch.rand(16, device=device) * 2.0 - 1.0) * torch.pi
        tau = torch.rand(16, device=device) * 0.2 + 0.28
        chi = (torch.rand(16, device=device) - 0.5) * torch.pi
        ring = build_sugar_ring_closed_form(chi, P, tau, restype_idx)
        tc = _template_tc(device)
        bl_o4 = tc['bl_o4_c4'][restype_idx.long()]
        dst = (ring["C4'"] - ring["O4'"]).norm(dim=-1)
        dlen = (dst - bl_o4).abs().mean().item()
        # C4′–O4′ length vs template: see planar sugar builder (ν-first).
        assert dlen < 0.25
        assert torch.isfinite(ring["C1'"]).all()
        ri_l = restype_idx.long()
        bl_o4_c1 = (tc['c1'][ri_l] - tc['o4'][ri_l]).norm(dim=-1)
        v = ring["C1'"] - ring["O4'"]
        err = (v.norm(dim=-1) - bl_o4_c1).abs().mean().item()
        assert err < 0.04

        d44 = (ring["C4'"] - ring["O4'"]).norm(dim=-1) - bl_o4
        assert (d44.abs() < 0.12).all()


def test_exocyclic_stereochemistry(device):
    ri = torch.tensor([0], device=device)
    P = torch.tensor([0.2], device=device)
    tau = torch.tensor([0.34], device=device)
    chi = torch.zeros(1, device=device)
    ring = build_sugar_ring_closed_form(chi, P, tau, ri)
    atoms = add_exocyclic_sugar_atoms(ring, _template_delta(device, ri), restype_indices=ri)
    tc = _template_tc(device)
    l_o3 = tc['bl_o3_c3'][ri]
    l_c5 = tc['bl_c5_c4'][ri]
    assert ((atoms["O3'"] - atoms["C3'"]).norm(dim=-1) - l_o3).abs().item() < 0.02
    assert ((atoms["C5'"] - atoms["C4'"]).norm(dim=-1) - l_c5).abs().item() < 0.02

    ch_geo = (torch.linalg.cross(
        atoms["C4'"] - atoms["C3'"],
        atoms["C5'"] - atoms["C4'"],
        dim=-1,
    ) * (atoms["O3'"] - atoms["C3'"])).sum(dim=-1)
    assert ch_geo.item() * tc['ring_chiral_triple'][ri].item() > 0


def test_o3_follows_predicted_delta_not_template_o3_prior(device):
    ri = torch.tensor([0], device=device)
    P = torch.tensor([0.2], device=device)
    tau = torch.tensor([0.34], device=device)
    chi = torch.tensor([0.1], device=device)
    ring = build_sugar_ring_closed_form(chi, P, tau, ri)
    tc = _template_tc(device)
    delta = tc['psi_o3'][ri]
    base_atoms = add_exocyclic_sugar_atoms(
        ring, delta, restype_indices=ri, geometry={'template_tensors': tc},
    )
    tc_shifted = {key: value.clone() for key, value in tc.items()}
    tc_shifted['psi_o3_ring'][ri] = tc_shifted['psi_o3_ring'][ri] + 0.75
    shifted_atoms = add_exocyclic_sugar_atoms(
        ring, delta, restype_indices=ri, geometry={'template_tensors': tc_shifted},
    )
    assert torch.allclose(
        shifted_atoms["O3'"],
        base_atoms["O3'"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_o3_measured_delta_matches_input(device):
    ri = torch.tensor([0], device=device)
    P = torch.tensor([0.2], device=device)
    tau = torch.tensor([0.34], device=device)
    chi = torch.tensor([0.1], device=device)
    ring = build_sugar_ring_closed_form(chi, P, tau, ri)
    delta = torch.tensor([0.7], device=device)
    atoms = add_exocyclic_sugar_atoms(ring, delta, restype_indices=ri)
    delta_meas = dihedral_rad(
        atoms["C5'"],
        atoms["C4'"],
        atoms["C3'"],
        atoms["O3'"],
    )
    assert wrap_dihedral_diff(delta_meas, delta).abs().item() < 0.05


def test_gamma_places_o5(device):
    ri = torch.tensor([2], device=device)
    P = torch.tensor([-0.1], device=device)
    tau = torch.tensor([0.32], device=device)
    chi = torch.zeros(1, device=device)
    ring = build_sugar_ring_closed_form(chi, P, tau, ri)
    atoms = add_exocyclic_sugar_atoms(ring, _template_delta(device, ri), restype_indices=ri)
    gam = torch.tensor([0.55], device=device)
    atoms = add_o5_from_gamma(atoms, gam, restype_indices=ri)
    g_meas = dihedral_rad(
        atoms["O5'"], atoms["C5'"], atoms["C4'"], atoms["C3'"],
    )
    err = wrap_dihedral_diff(g_meas, gam).abs().item()
    assert err < 0.05


def test_phosphate_bridge_uses_predicted_bridge_phase(device):
    # Use a contiguous C–C pair so the inter-residue chord is chemically feasible for bridge placement.
    rest = 'C'
    prev_np, next_np = canonical_two_residue_bridge_positions(rest, rest)

    dev = device
    ri = torch.tensor([{'A': 0, 'C': 1, 'G': 2, 'T': 3}[rest]], dtype=torch.long, device=dev)

    def _pv(d: dict, name: str) -> torch.Tensor:
        return torch.tensor(d[name], dtype=torch.float32, device=dev).unsqueeze(0)

    o3p = _pv(prev_np, "O3'")
    c3p = _pv(prev_np, "C3'")
    c4p = _pv(prev_np, "C4'")
    o5n = _pv(next_np, "O5'")
    c5n = _pv(next_np, "C5'")
    c4n = _pv(next_np, "C4'")
    p_gt = _pv(next_np, 'P')

    tc = _template_tc(dev)
    phase_t = bridge_phase_from_points_torch(
        o3p,
        o5n,
        c3p,
        p_gt,
        tc['bond_p_o3_inter'][ri],
        tc['bond_p_o5'][ri],
        c4p,
    )

    prev_atoms = {"O3'": o3p, "C3'": c3p, "C4'": c4p}
    next_atoms = {"O5'": o5n, "C5'": c5n, "C4'": c4n, '_ri': ri}
    out = close_phosphate_bridge_multi(
        prev_atoms,
        next_atoms,
        phase_t,
        geometry={'restype_indices_next': ri},
    )
    p = out['P'].squeeze(0)
    target_len = tc['bond_p_o3_inter'][ri].item()
    d_po3 = float(torch.linalg.vector_norm(p - o3p.squeeze(0)).item())
    assert abs(d_po3 - target_len) < 0.12

    phase_m = bridge_phase_from_points_torch(
        o3p,
        o5n,
        c3p,
        p.unsqueeze(0),
        tc['bond_p_o3_inter'][ri],
        tc['bond_p_o5'][ri],
        c4p,
    )
    err = abs(float(wrap_dihedral_diff(phase_m, phase_t).item()))
    assert err < 1e-4


def test_ground_truth_torsions_from_disk():
    root = Path(__file__).resolve().parents[2]
    found = list(root.glob('*.pt'))
    if not found:
        pytest.skip('no *.pt at repo root')
    pth = found[0]
    obj = torch.load(pth, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'torsions' in obj:
        t = obj['torsions'].float()
        assert int(t.shape[-1]) == N_TORSIONS
        n = min(int(t.shape[0]), 4)
        tm = obj.get('tau_m', torch.full((n,), 0.5)).float().view(-1)[:n]
        if tm.numel() < n:
            tm = tm.expand(n)
        ri = torch.zeros(n, dtype=torch.long)
        bb = build_window_backbone_from_torsions(
            t[:n], tm[:n], ri,
            torch.zeros(n, 3),
            torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(n, 3, 3).clone(),
        )
        assert torch.isfinite(bb).all()


def test_build_backbone_requires_c3_prev_for_chemical_bridge(device):
    theta = torch.zeros(1, N_TORSIONS, device=device)
    tau = torch.tensor([0.35], device=device)
    ri = torch.tensor([3], dtype=torch.long, device=device)
    o3 = torch.randn(1, 3, device=device)
    with pytest.raises(ValueError, match='c3_prev_local is required'):
        build_backbone_from_torsions(theta, tau, ri, o3, None)


def test_bridge_phase_is_rigid_transform_invariant(device):
    rest = 'C'
    prev_np, next_np = canonical_two_residue_bridge_positions(rest, rest)

    def t(d, name):
        return torch.tensor(d[name], dtype=torch.float64, device=device).reshape(1, 3)

    o3 = t(prev_np, "O3'")
    c3 = t(prev_np, "C3'")
    c4 = t(prev_np, "C4'")
    o5 = t(next_np, "O5'")
    p = t(next_np, 'P')

    ri = torch.tensor([1], dtype=torch.long, device=device)
    tc = _template_tc(device)

    phase0 = bridge_phase_from_points_torch(
        o3,
        o5,
        c3,
        p,
        tc['bond_p_o3_inter'][ri].double(),
        tc['bond_p_o5'][ri].double(),
        c4,
    )

    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, device=device))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    shift = torch.randn(1, 3, dtype=torch.float64, device=device)

    def rt(x):
        return x @ q.T + shift

    phase1 = bridge_phase_from_points_torch(
        rt(o3),
        rt(o5),
        rt(c3),
        rt(p),
        tc['bond_p_o3_inter'][ri].double(),
        tc['bond_p_o5'][ri].double(),
        rt(c4),
    )

    assert wrap_dihedral_diff(phase1, phase0).abs().item() < 1e-6


def test_ground_truth_torsions_rmsd_loose(device):
    tpl = _get_template('T')
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
    t_row, mask, tau_mv, tau_ok = nucleotide_torsions(tpl, xyz_prev, xyz_next, 'T')
    assert tau_ok
    assert bool(mask[TOR_BRIDGE_PHASE])
    tau_t = float(tau_mv)
    theta = torch.tensor(t_row, dtype=torch.float32, device=device).unsqueeze(0)
    tau = torch.tensor([tau_t], device=device)
    ri = torch.tensor([3], dtype=torch.long, device=device)

    bb_sugar = build_backbone_from_torsions(theta, tau, ri, None)
    sugar_atoms = ("C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'")
    diffs = []
    for nm in sugar_atoms:
        gt = torch.tensor(tpl[nm], dtype=torch.float32, device=device)
        diffs.append((bb_sugar[nm].squeeze(0) - gt).norm().item())
    rms = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else 0.0
    # Tight bound: with the column-stacked frame, the template-derived y-axis
    # and the stereo-pinned C2' branch, the only residual is the Altona-Sundaralingam
    # ν fit error (~0.03 rad) which propagates to ~0.05 Å per ring atom.
    assert rms < 0.10, f'round-trip RMSD too high: {rms:.4f} A'

    o3_prev = torch.tensor(xyz_prev["O3'"], dtype=torch.float32, device=device).unsqueeze(0)
    c3_prev = torch.tensor(xyz_prev["C3'"], dtype=torch.float32, device=device).unsqueeze(0)
    c4_prev = torch.tensor(xyz_prev["C4'"], dtype=torch.float32, device=device).unsqueeze(0)
    bb = build_backbone_from_torsions(theta, tau, ri, o3_prev, c3_prev, c4_prev)
    assert 'P' in bb
    assert torch.isfinite(bb['P']).all()


def test_window_builder_batch_size_gt_one_finite(device):
    B, W = 3, 4
    theta = torch.randn(B, W, N_TORSIONS, device=device, dtype=torch.float32) * 0.1
    tau = torch.full((B, W), 0.35, device=device, dtype=torch.float32)
    ri = torch.randint(0, 4, (B, W), device=device, dtype=torch.long)
    origins = torch.randn(B, W, 3, device=device, dtype=torch.float32)
    frm = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, W, 3, 3).contiguous()
    mask = torch.ones(B, W, N_TORSIONS, dtype=torch.bool, device=device)
    bb = build_batch_window_backbone_from_torsions(theta, tau, ri, origins, frm, mask)
    assert bb.shape[0] == B and bb.shape[1] == W
    _assert_window_bb_finiteness_design(bb)
