"""Single source for torsion order and latent width (angles + log τ_m channel)."""

from __future__ import annotations

import torch

TORSION_NAMES = ('alpha', 'beta', 'gamma', 'epsilon', 'zeta', 'chi', 'P')
N_TORSIONS = len(TORSION_NAMES)
N_LATENT = N_TORSIONS + 1
N_TORSIONS_LATENT = N_LATENT

TOR_ALPHA = TORSION_NAMES.index('alpha')
TOR_BETA = TORSION_NAMES.index('beta')
TOR_GAMMA = TORSION_NAMES.index('gamma')
TOR_EPS = TORSION_NAMES.index('epsilon')
TOR_ZETA = TORSION_NAMES.index('zeta')
TOR_CHI = TORSION_NAMES.index('chi')
TOR_PUCKER_P = TORSION_NAMES.index('P')

TORSION_IS_CIRCULAR = torch.tensor(
    [True] * N_TORSIONS,
    dtype=torch.bool,
)
