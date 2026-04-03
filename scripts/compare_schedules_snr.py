"""SNR-matched schedule comparison.

For a pair of beta schedules (default: linear vs cosine), prints a table showing
which timestep t_B in schedule B gives the same alpha_bar (and thus the same SNR)
as a reference timestep t_A in schedule A.

Usage:
    NUM_TIMESTEPS=200 python scripts/compare_schedules_snr.py
    SCHEDULE_A=linear SCHEDULE_B=cosine REF_SCHEDULE=A NUM_TIMESTEPS=200 \
        python scripts/compare_schedules_snr.py

ENV vars:
    NUM_TIMESTEPS   int, default 200
    SCHEDULE_A      str, default 'linear'
    SCHEDULE_B      str, default 'cosine'
    REF_SCHEDULE    'A' or 'B' — which schedule provides the reference t values (default 'A')
    REF_T_POINTS    comma-separated int list (default: 10 evenly-spaced + boundary points)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from model import get_beta_schedule


def build_alphas_cumprod(schedule: str, num_timesteps: int) -> torch.Tensor:
    betas = get_beta_schedule(schedule, num_timesteps)
    return torch.cumprod(1.0 - betas, dim=0)


def log_snr(ac: torch.Tensor) -> torch.Tensor:
    """log-SNR = log(alpha_bar / (1 - alpha_bar))."""
    return torch.log(ac / (1.0 - ac).clamp(min=1e-12))


def find_matched_t(ac_target: torch.Tensor, ref_value: float) -> int:
    """Return t in target schedule with closest alpha_bar to ref_value."""
    return int((ac_target - ref_value).abs().argmin().item())


def default_ref_ts(num_timesteps: int, n: int = 12) -> list[int]:
    step = max(1, num_timesteps // n)
    pts = list(range(0, num_timesteps, step))
    if (num_timesteps - 1) not in pts:
        pts.append(num_timesteps - 1)
    return pts


def main():
    num_timesteps = int(os.getenv('NUM_TIMESTEPS', '200'))
    sched_a = os.getenv('SCHEDULE_A', 'linear')
    sched_b = os.getenv('SCHEDULE_B', 'cosine')
    ref_side = os.getenv('REF_SCHEDULE', 'A').strip().upper()

    raw_ts = os.getenv('REF_T_POINTS', '').strip()
    if raw_ts:
        ref_ts = [int(x.strip()) for x in raw_ts.split(',') if x.strip()]
    else:
        ref_ts = default_ref_ts(num_timesteps)

    ac_a = build_alphas_cumprod(sched_a, num_timesteps)
    ac_b = build_alphas_cumprod(sched_b, num_timesteps)
    lsnr_a = log_snr(ac_a)
    lsnr_b = log_snr(ac_b)

    if ref_side == 'A':
        ac_ref, ac_match = ac_a, ac_b
        lsnr_ref, lsnr_match = lsnr_a, lsnr_b
        name_ref, name_match = sched_a, sched_b
    else:
        ac_ref, ac_match = ac_b, ac_a
        lsnr_ref, lsnr_match = lsnr_b, lsnr_a
        name_ref, name_match = sched_b, sched_a

    col = 14
    hdr = (
        f'{"t_" + name_ref:>{col}} | {"alpha_bar":>{col}} | {"log_SNR":>{col}} | '
        f'{"t_" + name_match + " (matched)":>{col+4}} | {"alpha_bar":>{col}} | '
        f'{"log_SNR":>{col}} | {"delta_alpha_bar":>{col}}'
    )
    sep = '-' * len(hdr)

    print(f'\nNUM_TIMESTEPS={num_timesteps}  |  {name_ref}  →  {name_match}\n')
    print(hdr)
    print(sep)

    for t_ref in ref_ts:
        ab_ref = ac_ref[t_ref].item()
        ls_ref = lsnr_ref[t_ref].item()
        t_match = find_matched_t(ac_match, ab_ref)
        ab_match = ac_match[t_match].item()
        ls_match = lsnr_match[t_match].item()
        delta = ab_match - ab_ref
        print(
            f'{t_ref:>{col}} | {ab_ref:>{col}.6f} | {ls_ref:>{col}.3f} | '
            f'{t_match:>{col+4}} | {ab_match:>{col}.6f} | '
            f'{ls_match:>{col}.3f} | {delta:>{col}.2e}'
        )

    print()
    # Summary: at what t does each schedule reach SNR ≈ 0 (alpha_bar ≈ 0.5)?
    t_half_a = find_matched_t(ac_a, 0.5)
    t_half_b = find_matched_t(ac_b, 0.5)
    print(
        f'alpha_bar ≈ 0.5 (SNR=0) reached at:  '
        f't_{sched_a}={t_half_a}  t_{sched_b}={t_half_b}'
    )
    t_noise_a = find_matched_t(ac_a, 0.01)
    t_noise_b = find_matched_t(ac_b, 0.01)
    print(
        f'alpha_bar ≈ 0.01 (nearly pure noise) reached at:  '
        f't_{sched_a}={t_noise_a}  t_{sched_b}={t_noise_b}'
    )


if __name__ == '__main__':
    main()
