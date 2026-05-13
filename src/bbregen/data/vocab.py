"""Shared domain vocabulary for nucleotide features and backbone layout."""

BACKBONE_ATOMS = (
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
    'OP1',
    'OP2',
    'P',
    "O3'",
    "O4'",
    "O5'",
)

NUCLEIC_ACID_ATOMS = (
    'N1',
    'N2',
    'N3',
    'N4',
    'N6',
    'N7',
    'N9',
    'C2',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'O2',
    'O4',
    'O6',
)

NUCLEOTIDE_ATOMS = NUCLEIC_ACID_ATOMS + BACKBONE_ATOMS
ATOM_TO_INDEX = {atom: idx for idx, atom in enumerate(NUCLEOTIDE_ATOMS)}

BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_BASE = {idx: base for base, idx in BASE_TO_INDEX.items()}

CHAIN_END_CLASS_INTERNAL = 0
CHAIN_END_CLASS_5_PRIME = 1
CHAIN_END_CLASS_3_PRIME = 2
N_CHAIN_END_CLASSES = 3

FIVE_PRIME_PHOSPHATE_ATOMS = frozenset({'P', 'OP1', 'OP2'})
