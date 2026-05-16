# Base2Backbone

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Eclipter/base2backbone.git
cd base2backbone
```

1. Install Conda. See [Conda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Download PyNAMod:

```bash
git clone https://github.com/intbio/PyNAMod.git pynamod
cd pynamod
```

1. Create a new environment with PyNAMod dependencies:

```bash
/opt/miniconda/bin/conda env create \
  --name base2backbone \
  --file pynamod/environment.yml
```

1. Install the project dependencies:

```bash
/opt/miniconda/bin/conda env update \
  --name base2backbone \
  --file environment.yml
/opt/miniconda/bin/conda run \
  -n base2backbone \
  pip install \
    -e ./pynamod \
    -e .[test] \
    --no-deps
```

## Training

1. Make sure to set up the environment. See [Setup](#setup)
2. Train and test the model:

```bash
python scripts/train.py
```

1. Analyze the results if necessary

```bash
python scripts/analyze.py
```

1. Export the best model to ONNX:

```bash
python scripts/export.py --run-id torsions/1/baseline
```

## Usage

1. Make sure to set up the environment. See [Setup](#setup)
2. Predict the backbone (alternatives are listed below):

  - CLI (for a single structure):

    ```bash
    base2backbone \
      --input input.pdb \
      --output output.pdb
    ```

    Input format may be PDB or mmCIF. Output may be PDB or mmCIF independently. Both are inferred from the filename extension.

    By default 5'-terminal phosphate atoms (`P`, `OP1`, `OP2`) are not predicted. Pass `--generate-5-prime-phosphate` to include them.

  - CLI (for a trajectory):
    - Single multi-model file output:

      ```bash
      base2backbone \
        --input topology.pdb \
        --trajectory traj.xtc \
        --output output_traj.pdb
      ```

      Input format may be PDB or mmCIF. Output may be PDB or mmCIF independently. Both are inferred from the filename extension.

    - Example (one PDB per frame):

      ```bash
      base2backbone \
        --input topology.pdb \
        --trajectory traj.xtc \
        --output-dir frames \
        --output-format .cif
      ```

      Input format may be PDB or mmCIF. Output may be PDB or mmCIF independently. Output format is set by the `--output-format` argument.

  - Python API:
    The library always returns an `MDAnalysis.Universe`.

      ```python
      import MDAnalysis as mda
      from base2backbone import predict_backbone, predict_backbone_trajectory

      single = predict_backbone('input.cif')
      traj_in = mda.Universe('topology.pdb', 'traj.xtc')
      traj_out = predict_backbone_trajectory(traj_in)
      ```
