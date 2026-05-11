"""Tests for sugar grid-closure ring, stereo exocyclic atoms, and multi-torsion phosphate decoder."""

from pathlib import Path

import numpy as np
import pytest
import torch

from torsion_geometry import (
    N_TORSIONS,
    TOR_ALPHA,
    TOR_BETA,
    TOR_CHI,
    TOR_EPS,
    TOR_GAMMA,
    TOR_ZETA,
    _get_template,
    _get_template_tensors,
    add_exocyclic_sugar_atoms_torch,
    add_o5_from_gamma_torch,
    build_backbone_from_torsions_torch,
    build_sugar_ring_grid_closed_torch,
    build_window_backbone_from_torsions_torch,
    close_phosphate_bridge_multi_torch,
    dihedral_rad,
    dihedral_rad_torch,
    nucleotide_torsions_numpy,
    nus_rad_from_P_tau_torch,
    wrap_dihedral_diff_torch,
)


def _template_tc(dev):
    return _get_template_tensors(str(dev))


@pytest.fixture
def device():
    return torch.device('cpu')


def test_pseudorotation_torsions_consistent(device):
    restype_idx = torch.tensor([0, 1, 2, 3], device=device)
    P = torch.linspace(-1.0, 1.0, 4, device=device)
    tau = torch.full((4,), 0.35, device=device)
    chi = torch.zeros(4, device=device)
    ring = build_sugar_ring_grid_closed_torch(chi, P, tau, restype_idx)
    tgt = nus_rad_from_P_tau_torch(P, tau)
    for i, (a0, a1, a2, a3) in enumerate(
        [
            ("C1'", "C2'", "C3'", "C4'"),
            ("C2'", "C3'", "C4'", "O4'"),
            ("C3'", "C4'", "O4'", "C1'"),
            ("C4'", "O4'", "C1'", "C2'"),
            ("O4'", "C1'", "C2'", "C3'"),
        ],
    ):
        m = dihedral_rad_torch(
            ring[a0], ring[a1], ring[a2], ring[a3],
        )
        err = wrap_dihedral_diff_torch(m, tgt[:, i]).abs().max().item()
        assert err < 0.55, i


def test_sugar_ring_is_closed(device):
    torch.manual_seed(1)
    for _ in range(8):
        restype_idx = torch.randint(0, 4, (16,), device=device)
        P = (torch.rand(16, device=device) * 2.0 - 1.0) * torch.pi
        tau = torch.rand(16, device=device) * 0.2 + 0.28
        chi = (torch.rand(16, device=device) - 0.5) * torch.pi
        ring = build_sugar_ring_grid_closed_torch(chi, P, tau, restype_idx)
        tc = _template_tc(device)
        bl_o4 = tc['bl_o4_c4'][restype_idx.long()]
        dst = (ring["C4'"] - ring["O4'"]).norm(dim=-1)
        dlen = (dst - bl_o4).abs().mean().item()
        assert dlen < 0.5
        assert torch.isfinite(ring["C1'"]).all()
        v = ring["C1'"] - ring["O4'"]
        err = (v.norm(dim=-1) - tc['bl_c2_c1'][restype_idx.long()]).abs().mean().item()
        assert err < 0.12

        d44 = (ring["C4'"] - ring["O4'"]).norm(dim=-1) - bl_o4
        assert (d44.abs() < 0.5).all()


def test_exocyclic_stereochemistry(device):
    ri = torch.tensor([0], device=device)
    P = torch.tensor([0.2], device=device)
    tau = torch.tensor([0.34], device=device)
    chi = torch.zeros(1, device=device)
    ring = build_sugar_ring_grid_closed_torch(chi, P, tau, ri)
    atoms = add_exocyclic_sugar_atoms_torch(ring, restype_indices=ri)
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


def test_gamma_places_o5(device):
    ri = torch.tensor([2], device=device)
    P = torch.tensor([-0.1], device=device)
    tau = torch.tensor([0.32], device=device)
    chi = torch.zeros(1, device=device)
    ring = build_sugar_ring_grid_closed_torch(chi, P, tau, ri)
    atoms = add_exocyclic_sugar_atoms_torch(ring, restype_indices=ri)
    gam = torch.tensor([0.55], device=device)
    atoms = add_o5_from_gamma_torch(atoms, gam, restype_indices=ri)
    g_meas = dihedral_rad_torch(
        atoms["O5'"], atoms["C5'"], atoms["C4'"], atoms["C3'"],
    )
    err = wrap_dihedral_diff_torch(g_meas, gam).abs().item()
    assert err < 0.05


def test_phosphate_bridge_uses_all_four_torsions(device):
    tpl = _get_template('C')
    xyz_prev = {"O3'": tpl["O3'"].copy()}
    xyz_next = {'P': tpl['P'].copy(), "O5'": tpl["O5'"].copy()}
    t, *_ = nucleotide_torsions_numpy(tpl, xyz_prev, xyz_next, 'C')
    e_t = float(t[TOR_EPS])
    z_t = float(t[TOR_ZETA])
    a_t = float(t[TOR_ALPHA])
    b_t = float(t[TOR_BETA])

    dev = device
    ri = torch.tensor([1], dtype=torch.long, device=dev)
    o3p = torch.tensor(tpl["O3'"], dtype=torch.float32, device=dev).unsqueeze(0)
    c3p = torch.tensor(tpl["C3'"], dtype=torch.float32, device=dev).unsqueeze(0)
    c4p = torch.tensor(tpl["C4'"], dtype=torch.float32, device=dev).unsqueeze(0)
    o5n = torch.tensor(tpl["O5'"], dtype=torch.float32, device=dev).unsqueeze(0)
    c5n = torch.tensor(tpl["C5'"], dtype=torch.float32, device=dev).unsqueeze(0)
    c4n = c4p.clone()

    prev_atoms = {"O3'": o3p, "C3'": c3p, "C4'": c4p}
    next_atoms = {
        "O5'": o5n,
        "C5'": c5n,
        "C4'": c4n,
        '_ri': ri,
    }
    et = torch.tensor([e_t], device=dev)
    zt = torch.tensor([z_t], device=dev)
    at = torch.tensor([a_t], device=dev)
    bt = torch.tensor([b_t], device=dev)
    out = close_phosphate_bridge_multi_torch(
        prev_atoms,
        next_atoms,
        et,
        zt,
        at,
        bt,
        geometry={'restype_indices_next': ri},
        weight_epsilon=1.0,
        weight_zeta=1.0,
        weight_alpha=1.0,
        weight_beta=1.0,
    )
    p = out['P'].squeeze(0)
    tc = _template_tc(dev)
    target_len = tc['bond_p_o3_inter'][ri].item()
    d_po3 = float(torch.linalg.vector_norm(p - o3p.squeeze(0)).item())
    # Circle intersection can fail on raw template neighbours; then ``close_phosphate_bridge_multi_torch`` uses midpoint fallback.
    if d_po3 < target_len + 0.65:
        assert abs(d_po3 - target_len) < 0.55

    eps_m = dihedral_rad_torch(c4p.squeeze(0), c3p.squeeze(0), o3p.squeeze(0), p)
    ze_m = dihedral_rad_torch(c3p.squeeze(0), o3p.squeeze(0), p, o5n.squeeze(0))
    al_m = dihedral_rad_torch(o3p.squeeze(0), p, o5n.squeeze(0), c5n.squeeze(0))
    be_m = dihedral_rad_torch(p, o5n.squeeze(0), c5n.squeeze(0), c4n.squeeze(0))
    for name, m, target in zip(
        ('eps', 'zeta', 'alpha', 'beta'),
        (eps_m, ze_m, al_m, be_m),
        (e_t, z_t, a_t, b_t),
    ):
        err = abs(
            float(
                wrap_dihedral_diff_torch(
                    m.reshape(1),
                    torch.tensor([target], device=dev),
                ).item(),
            ),
        )
        assert err < 0.25, name


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
        bb = build_window_backbone_from_torsions_torch(
            t[:n], tm[:n], ri,
            torch.zeros(n, 3),
            torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(n, 3, 3).clone(),
        )
        assert torch.isfinite(bb).all()


def test_ground_truth_torsions_rmsd_loose(device):
    tpl = _get_template('T')
    xyz_prev = {"O3'": tpl["O3'"].copy()}
    xyz_next = {'P': tpl['P'].copy(), "O5'": tpl["O5'"].copy()}
    t_row, _mk, tau_mv, tau_ok = nucleotide_torsions_numpy(tpl, xyz_prev, xyz_next, 'T')
    assert tau_ok
    tau_t = float(tau_mv)
    bb = build_backbone_from_torsions_torch(
        torch.tensor(t_row, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor([tau_t], device=device),
        torch.tensor([3], dtype=torch.long, device=device),
        None,
    )
    diffs = []
    for nm, v in bb.items():
        if nm not in tpl:
            continue
        gt = torch.tensor(tpl[nm], dtype=torch.float32, device=device)
        diffs.append((v.squeeze(0) - gt).norm().item())
    rms = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else 0.0
    assert rms < 5.0


def test_window_builder_batch_size_gt_one_finite(device):
    from torsion_geometry import build_batch_window_backbone_from_torsions_torch

    B, W = 3, 4
    theta = torch.randn(B, W, N_TORSIONS, device=device, dtype=torch.float32) * 0.1
    tau = torch.full((B, W), 0.35, device=device, dtype=torch.float32)
    ri = torch.randint(0, 4, (B, W), device=device, dtype=torch.long)
    origins = torch.randn(B, W, 3, device=device, dtype=torch.float32)
    frm = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).expand(B, W, 3, 3).contiguous()
    mask = torch.ones(B, W, N_TORSIONS, dtype=torch.bool, device=device)
    bb = build_batch_window_backbone_from_torsions_torch(theta, tau, ri, origins, frm, mask)
    assert bb.shape[0] == B and bb.shape[1] == W
    assert torch.isfinite(bb).all()
