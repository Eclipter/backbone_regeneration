"""Wrapped score matching on T^k for torsion angles and Gaussian score for log τ_m."""

import math

import torch

from .torsion_constants import (
    LOG_TAU_M_MAX,
    LOG_TAU_M_MIN,
    N_LATENT,
    N_TORSIONS,
    TAU_M_MAX,
    TAU_M_MIN,
)


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Map angles to (-pi, pi]. Differentiable almost everywhere."""
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def wrapped_angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return wrapped difference a - b in (-pi, pi]."""
    return wrap_angle(a - b)


def sigma_schedule(
    t: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Exponential sigma schedule: sigma(t) = sigma_min ** (1 - t) * sigma_max ** t, t in [0, 1]."""
    lo = math.log(float(sigma_min))
    hi = math.log(float(sigma_max))
    log_sigma = lo + t * (hi - lo)
    return torch.exp(log_sigma)


def wrapped_normal_score(
    x: torch.Tensor,
    sigma: torch.Tensor,
    num_wraps: int = 5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Score d/dx log WrappedNormal(x; 0, sigma^2) with x = wrapped angular increment."""
    x = wrap_angle(x)
    sigma = sigma.clamp_min(eps)
    while sigma.ndim < x.ndim:
        sigma = sigma.unsqueeze(-1)

    ks = torch.arange(
        -num_wraps,
        num_wraps + 1,
        device=x.device,
        dtype=x.dtype,
    )
    view_shape = [1] * x.ndim + [2 * num_wraps + 1]
    ks = ks.view(*view_shape)

    xk = x.unsqueeze(-1) + 2.0 * math.pi * ks
    inv_var = 1.0 / (sigma.unsqueeze(-1) ** 2)

    log_w = -0.5 * xk * xk * inv_var
    log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
    w = torch.exp(log_w)

    score = -torch.sum(xk * inv_var * w, dim=-1)
    return score


def gaussian_score(
    y_t: torch.Tensor,
    y_0: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Score ∂/∂y_t log N(y_t; y_0, sigma^2) = -(y_t - y_0) / sigma^2."""
    sigma = sigma.clamp_min(eps)
    while sigma.ndim < y_t.ndim:
        sigma = sigma.unsqueeze(-1)
    return -(y_t - y_0) / (sigma ** 2)


def encode_torsions(theta: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
    """Concatenate wrapped θ and log τ_m → latent [..., N_LATENT]."""
    theta_w = wrap_angle(theta)
    log_tau = torch.log(tau_m.clamp(min=TAU_M_MIN, max=TAU_M_MAX))
    if log_tau.ndim == theta_w.ndim - 1:
        log_tau = log_tau.unsqueeze(-1)
    log_tau = log_tau.clamp(LOG_TAU_M_MIN, LOG_TAU_M_MAX)
    return torch.cat([theta_w, log_tau], dim=-1)


def decode_torsions(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split latent [..., N_LATENT] into θ and τ_m."""
    theta = wrap_angle(x[..., :N_TORSIONS])
    log_tau = x[..., N_TORSIONS].clamp(LOG_TAU_M_MIN, LOG_TAU_M_MAX)
    tau_m = torch.exp(log_tau).clamp(min=TAU_M_MIN, max=TAU_M_MAX)
    return theta, tau_m


def perturb_torsions(
    theta_0: torch.Tensor,
    log_tau_0: torch.Tensor,
    t: torch.Tensor,
    angular_sigma_min: float,
    angular_sigma_max: float,
    tau_sigma_min: float,
    tau_sigma_max: float,
) -> dict[str, torch.Tensor]:
    """Forward perturbation and closed-form score targets for wrapped-normal angles + Gaussian log τ."""
    if theta_0.shape[-1] != N_TORSIONS:
        raise ValueError(
            f'theta_0 last dimension must be {N_TORSIONS}, got {theta_0.shape[-1]}',
        )
    sigma_theta = sigma_schedule(t, angular_sigma_min, angular_sigma_max)
    sigma_tau = sigma_schedule(t, tau_sigma_min, tau_sigma_max)

    while sigma_theta.ndim < theta_0.ndim:
        sigma_theta = sigma_theta.unsqueeze(-1)
    while sigma_tau.ndim < log_tau_0.ndim:
        sigma_tau = sigma_tau.unsqueeze(-1)

    eps_theta = torch.randn_like(theta_0)
    theta_t = wrap_angle(theta_0 + sigma_theta * eps_theta)

    eps_tau = torch.randn_like(log_tau_0)
    log_tau_t = log_tau_0 + sigma_tau * eps_tau

    angular_score_target = wrapped_normal_score(
        wrapped_angle_diff(theta_t, theta_0),
        sigma_theta,
    )
    tau_score_target = gaussian_score(log_tau_t, log_tau_0, sigma_tau)

    return {
        'theta_t': theta_t,
        'log_tau_t': log_tau_t,
        'angular_score_target': angular_score_target,
        'tau_score_target': tau_score_target,
        'sigma_theta': sigma_theta,
        'sigma_tau': sigma_tau,
        'angular_noise': eps_theta,
        'tau_noise': eps_tau,
    }


def estimate_latent_from_ve_score(
    theta_t: torch.Tensor,
    log_tau_t: torch.Tensor,
    score_pred: torch.Tensor,
    sigma_theta: torch.Tensor,
    sigma_tau: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Tweedie-style x̂_0 from VE score prediction (for auxiliary closure / diagnostics)."""
    sig_th = sigma_theta.clamp_min(eps)
    sig_tau = sigma_tau.clamp_min(eps)
    while sig_th.ndim < theta_t.ndim:
        sig_th = sig_th.unsqueeze(-1)
    while sig_tau.ndim < log_tau_t.ndim:
        sig_tau = sig_tau.unsqueeze(-1)
    theta_hat = wrap_angle(theta_t + (sig_th ** 2) * score_pred[..., :N_TORSIONS])
    log_hat = (log_tau_t + (sig_tau ** 2) * score_pred[..., N_TORSIONS:N_LATENT]).clamp(
        LOG_TAU_M_MIN,
        LOG_TAU_M_MAX,
    )
    return torch.cat([theta_hat, log_hat], dim=-1)


def weighted_score_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sigma_sq_weight: torch.Tensor | None,
    *,
    weighting: str,
) -> torch.Tensor:
    """Masked MSE with optional σ² scaling per score matching practice."""
    if weighting not in ('none', 'sigma2'):
        raise ValueError(f'unknown score_loss_weighting: {weighting!r}')
    diff_sq = (pred - target) ** 2
    if weighting == 'sigma2':
        assert sigma_sq_weight is not None
        w = sigma_sq_weight
        while w.ndim < diff_sq.ndim:
            w = w.unsqueeze(-1)
        diff_sq = diff_sq * w
    m = mask
    while m.ndim < diff_sq.ndim:
        m = m.unsqueeze(-1)
    num = (diff_sq * m).sum()
    den = m.sum().clamp(min=1.0)
    return num / den


def ve_sigma_grid(
    sigma_max: float,
    sigma_min: float,
    num_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """σ_i from high to low (inclusive endpoints on log grid)."""
    if num_steps < 2:
        raise ValueError('num_steps must be >= 2 for a sigma trajectory')
    lo = math.log(float(sigma_min))
    hi = math.log(float(sigma_max))
    log_sig = torch.linspace(hi, lo, num_steps, device=device, dtype=dtype)
    return torch.exp(log_sig)


def reverse_ve_score_step(
    theta: torch.Tensor,
    log_tau: torch.Tensor,
    score_pred: torch.Tensor,
    sigma_cur: torch.Tensor,
    sigma_next: torch.Tensor,
    sigma_tau_cur: torch.Tensor,
    sigma_tau_next: torch.Tensor,
    *,
    rng_theta: torch.Tensor | None = None,
    rng_tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ancestral VE reverse step toward lower σ using predicted scores."""
    eps_safe = 1e-12
    d2_theta = (sigma_cur ** 2 - sigma_next ** 2).clamp(min=0.0)
    d2_tau = (sigma_tau_cur ** 2 - sigma_tau_next ** 2).clamp(min=0.0)

    while d2_theta.ndim < theta.ndim:
        d2_theta = d2_theta.unsqueeze(-1)
    while d2_tau.ndim < log_tau.ndim:
        d2_tau = d2_tau.unsqueeze(-1)

    z_th = torch.randn_like(theta) if rng_theta is None else rng_theta
    z_tau = torch.randn_like(log_tau) if rng_tau is None else rng_tau

    step_noise_theta = torch.zeros_like(theta)
    if float(sigma_next.max()) > eps_safe:
        step_noise_theta = torch.sqrt(d2_theta.clamp(min=eps_safe)) * z_th
    step_noise_tau = torch.zeros_like(log_tau)
    if float(sigma_tau_next.max()) > eps_safe:
        step_noise_tau = torch.sqrt(d2_tau.clamp(min=eps_safe)) * z_tau

    theta_next = wrap_angle(theta + d2_theta * score_pred[..., :N_TORSIONS] + step_noise_theta)
    log_next = (
        log_tau + d2_tau * score_pred[..., N_TORSIONS:N_LATENT] + step_noise_tau
    ).clamp(LOG_TAU_M_MIN, LOG_TAU_M_MAX)
    return theta_next, log_next
