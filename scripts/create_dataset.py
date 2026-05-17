import sys
from pathlib import Path

from config import DATASET_MANIFEST, SEED

from base2backbone.dataset import prepare_dataset

# Allow `from config import ...` when running as a script (same pattern as train.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def main():
    prepare_dataset(SEED, DATASET_MANIFEST)


if __name__ == '__main__':
    main()
