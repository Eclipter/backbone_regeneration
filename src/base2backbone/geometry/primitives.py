"""Shared geometry primitives used across decoding and closure metrics."""

import math

import numpy as np
import torch

GEO_EPS = 1e-8
GEO_EPS_NP = 1e-12
# Backward-stable epsilons used in safe-norm and acos-clamp tricks.
# `GEO_EPS_SQ` keeps `((v*v).sum + eps2).sqrt()` differentiable at v=0;
# `ACOS_CLAMP_EPS` must be >> FP32 epsilon (~1.19e-7) so `1 - eps != 1` in float32.
GEO_EPS_SQ = 1e-12
ACOS_CLAMP_EPS = 1e-6


def _preferred_geometry_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def device() -> torch.device:
    """Prefer CUDA for numeric geometry (CPU fallback when CUDA is unavailable)."""
    return _preferred_geometry_device()


def expand_point_origin(origin: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while origin.ndim < points.ndim:
        origin = origin.unsqueeze(-2)
    return origin


def expand_point_frame(frame: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while frame.ndim < points.ndim + 1:
        frame = frame.unsqueeze(-3)
    return frame


def world_to_local_points(
    x_world: torch.Tensor,
    origin: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """Row-vector convention: ``local = (world - origin) @ frame``."""
    delta = x_world - expand_point_origin(origin, x_world)
    if frame.dim() == 2:
        return torch.matmul(delta, frame)
    return torch.matmul(delta.unsqueeze(-2), expand_point_frame(frame, delta)).squeeze(-2)


def local_to_world_points(
    x_local: torch.Tensor,
    origin: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """Row-vector convention: ``world = local @ frame.T + origin``."""
    rotation = frame.transpose(-2, -1)
    origin_expanded = expand_point_origin(origin, x_local)
    if frame.dim() == 2:
        return torch.matmul(x_local, rotation) + origin_expanded
    return (
        torch.matmul(x_local.unsqueeze(-2), expand_point_frame(rotation, x_local)).squeeze(-2)
        + origin_expanded
    )


def wrap_angle_rad(x):
    """Map angles to ``(-pi, pi]``."""
    return np.arctan2(np.sin(x), np.cos(x))


def _dihedral_rad(p0, p1, p2, p3) -> float:
    b1 = p1 - p0; b2 = p2 - p1; b3 = p3 - p2
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1u = n1 / (np.linalg.norm(n1) + GEO_EPS_NP)
    n2u = n2 / (np.linalg.norm(n2) + GEO_EPS_NP)
    m1 = np.cross(n1u, b2 / (np.linalg.norm(b2) + GEO_EPS_NP))
    return float(np.arctan2(np.dot(m1, n2u), np.dot(n1u, n2u)))


def _bond_angle(a, b, c) -> float:
    ba = a - b; bc = c - b
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + GEO_EPS_NP)
    return float(np.arccos(np.clip(cos_t, -1.0 + 1e-9, 1.0 - 1e-9)))


def dihedral_rad(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Batched signed dihedral in radians; shapes broadcast to ``[..., 3]``."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    # `(v*v).sum + eps2`.sqrt() avoids NaN backward of `||v||` at v=0 (`v/||v||` -> 0/0).
    b2u = b2 / ((b2 * b2).sum(dim=-1, keepdim=True) + GEO_EPS_SQ).sqrt()
    n1u = n1 / ((n1 * n1).sum(dim=-1, keepdim=True) + GEO_EPS_SQ).sqrt()
    n2u = n2 / ((n2 * n2).sum(dim=-1, keepdim=True) + GEO_EPS_SQ).sqrt()
    m1 = torch.linalg.cross(n1u, b2u)
    x = (n1u * n2u).sum(dim=-1)
    y = (m1 * n2u).sum(dim=-1)
    # Substitute (x, y) -> (1, 0) where both vanish: avoids `atan2(0, 0)` backward = 0/0 = NaN
    # for collinear-bond samples; gradient on degenerate rows then flows only into the constant.
    ok = (x * x + y * y) > GEO_EPS_SQ
    y = torch.where(ok, y, torch.zeros_like(y))
    x = torch.where(ok, x, torch.ones_like(x))
    return torch.atan2(y, x)


def dihedral_rad_coords(p0, p1, p2, p3, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Dihedral for array-like points; auto-selects CUDA when available."""
    device = _preferred_geometry_device()
    return dihedral_rad(
        torch.as_tensor(p0, dtype=dtype, device=device),
        torch.as_tensor(p1, dtype=dtype, device=device),
        torch.as_tensor(p2, dtype=dtype, device=device),
        torch.as_tensor(p3, dtype=dtype, device=device),
    )


def nerf_place(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
) -> torch.Tensor:
    """Batched differentiable NERF."""
    ba = a - b
    bc = c - b
    bc_u = bc / (bc.norm(dim=-1, keepdim=True) + GEO_EPS_NP)
    normal = torch.linalg.cross(ba, bc_u)
    normal_norm = normal.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(normal)
    fallback[..., 0] = 1.0
    normal = torch.where(normal_norm < 1e-10, fallback, normal / (normal_norm + GEO_EPS_NP))
    binormal = torch.linalg.cross(normal, bc_u)
    radius = r.unsqueeze(-1)
    return c + radius * (
        torch.cos(theta.new_tensor(math.pi) - theta).unsqueeze(-1) * bc_u
        + torch.sin(theta.new_tensor(math.pi) - theta).unsqueeze(-1)
        * (
            torch.cos(psi).unsqueeze(-1) * normal
            + torch.sin(psi).unsqueeze(-1) * binormal
        )
    )


def nerf_place_coords(a, b, c, r, theta, psi, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Single-point NeRF; returns ``[3]``, auto-selects CUDA when available."""
    device = _preferred_geometry_device()
    return nerf_place(
        torch.as_tensor(a, dtype=dtype, device=device).reshape(1, 3),
        torch.as_tensor(b, dtype=dtype, device=device).reshape(1, 3),
        torch.as_tensor(c, dtype=dtype, device=device).reshape(1, 3),
        torch.as_tensor(r, dtype=dtype, device=device).reshape(1),
        torch.as_tensor(theta, dtype=dtype, device=device).reshape(1),
        torch.as_tensor(psi, dtype=dtype, device=device).reshape(1),
    ).reshape(3)


def bond_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float = GEO_EPS,
) -> torch.Tensor:
    """Interior angle ``∠(a-b-c)`` in radians; batched last dim 3."""
    ba = a - b
    bc = c - b
    # `(v*v).sum + eps2`.sqrt() avoids NaN backward of `||v||` at v=0 (`v/||v||` -> 0/0).
    ba_n = ((ba * ba).sum(dim=-1) + GEO_EPS_SQ).sqrt()
    bc_n = ((bc * bc).sum(dim=-1) + GEO_EPS_SQ).sqrt()
    denom = ba_n * bc_n + eps
    cos_t = (ba * bc).sum(dim=-1) / denom
    # `1 - 1e-8` collapses to 1.0 in float32 (FP32 eps ~1.19e-7) -> acos backward = -inf.
    # Use `ACOS_CLAMP_EPS=1e-6` (float32-effective); on the boundary clamp zeroes the grad.
    cos_t = cos_t.clamp(-1.0 + ACOS_CLAMP_EPS, 1.0 - ACOS_CLAMP_EPS)
    return torch.acos(cos_t)


def bond_angle_coords(a, b, c, *, dtype: torch.dtype = torch.float64, eps: float = GEO_EPS) -> torch.Tensor:
    """Bond angle for array-like vertices; auto-selects CUDA when available."""
    device = _preferred_geometry_device()
    return bond_angle(
        torch.as_tensor(a, dtype=dtype, device=device),
        torch.as_tensor(b, dtype=dtype, device=device),
        torch.as_tensor(c, dtype=dtype, device=device),
        eps,
    )


def wrap_dihedral_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + torch.pi) % (2.0 * torch.pi) - torch.pi


def safe_normalize(v: torch.Tensor, eps: float = GEO_EPS) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def orthonormal_basis_from_axis(
    axis: torch.Tensor,
    eps: float = GEO_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit axis and two perpendicular unit vectors."""
    unit_axis = safe_normalize(axis, eps=eps)
    tmp = torch.zeros_like(unit_axis)
    tmp[..., 0] = 1.0
    use_alt = unit_axis[..., 0].abs() >= 0.9
    tmp[use_alt, 0] = 0.0
    tmp[use_alt, 1] = 1.0
    e1 = torch.linalg.cross(tmp, unit_axis, dim=-1)
    e1 = safe_normalize(e1, eps=eps)
    e2 = torch.linalg.cross(unit_axis, e1, dim=-1)
    return unit_axis, e1, e2


def signed_tetra_volume(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Signed volume proportional to ``(b-a)·((c-a)x(d-a))``."""
    return (torch.linalg.cross(b - a, c - a, dim=-1) * (d - a)).sum(dim=-1)


def rodrigues_rotate_point(
    v: torch.Tensor,
    k: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Rotate vectors around unit axes by ``theta`` radians."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cross = torch.linalg.cross(k, v, dim=-1)
    dot = (k * v).sum(dim=-1, keepdim=True)
    return (
        v * cos_t.unsqueeze(-1)
        + cross * sin_t.unsqueeze(-1)
        + k * dot * (1.0 - cos_t.unsqueeze(-1))
    )
