"""Single source for torsion order and latent width (angles + log τ_m channel)."""

import math

import torch

# Pseudorotational puckering amplitude τ_m (radians; see ``nucleotide_torsions`` / sugar fit).
# Typical DNA ~0.35–0.55; canonical templates ~0.61. Wide bounds are conservative; refine from dataset stats.
TAU_M_MIN = 0.05
TAU_M_MAX = 1.5
LOG_TAU_M_MIN = math.log(TAU_M_MIN)
LOG_TAU_M_MAX = math.log(TAU_M_MAX)

TORSION_NAMES = (
    'alpha',
    'beta',
    'gamma',
    'delta',
    'epsilon',
    'zeta',
    'chi',
    'pseudorotation_phase',
    'eta_p',
)
N_TORSIONS = len(TORSION_NAMES)
N_LATENT = N_TORSIONS + 1
N_TORSIONS_LATENT = N_LATENT

TOR_ALPHA = TORSION_NAMES.index('alpha')
TOR_BETA = TORSION_NAMES.index('beta')
TOR_GAMMA = TORSION_NAMES.index('gamma')
TOR_DELTA = TORSION_NAMES.index('delta')
TOR_EPS = TORSION_NAMES.index('epsilon')
TOR_ZETA = TORSION_NAMES.index('zeta')
TOR_CHI = TORSION_NAMES.index('chi')
TOR_PSEUDOROTATION_PHASE = TORSION_NAMES.index('pseudorotation_phase')
# eta_p = dihedral(C5', O5', P, OP1): orientation of the non-bridging PO2 group
TOR_ETA_P = TORSION_NAMES.index('eta_p')

TORSION_IS_CIRCULAR = torch.tensor(
    [True] * N_TORSIONS,
    dtype=torch.bool,
)


def assert_torsion_layout():
    """Debug helper: latent is nine wrapped angles plus log τ_m."""
    assert len(TORSION_NAMES) == N_TORSIONS == 9
    assert N_LATENT == N_TORSIONS + 1 == 10
    assert 'delta' in TORSION_NAMES
