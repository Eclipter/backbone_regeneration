# Backbone Regeneration

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Eclipter/backbone-regeneration.git
cd backbone-regeneration
```

2. Install Conda. See [Conda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

3. Create a new environment with pynamod dependencies:

```bash
conda env create --name backbone_regen --file pynamod/environment.yml
```

4. Install the project requirements:

```bash
conda env update --name backbone_regen --file environment.yml
```

## Training

1. Make sure to set up the environment. See [Environment Setup](#environment-setup)

2. Train and test the model

```bash
python src/train.py
```

3. Visualize the results if necessary

```bash
python src/visualize.py
```

4. Export the best model to ONNX. A single unified checkpoint regenerates both central and chain-edge nucleotides via the per-atom `is_target` flag, so `--run-dir` points to one concrete run directory.

```bash
python src/export_to_onnx.py --run-dir logs/fixed_swa/baseline
```

## Usage

1. Make sure to set up the environment. See [Environment Setup](#environment-setup)

2. Predict the backbone. `--run-dir` points to one concrete run directory with a trained checkpoint; the unified model is called once per window for the central nucleotide and, additionally, once per edge window for the chain-edge nucleotide. Input and output may be PDB or mmCIF independently (e.g. PDB in, mmCIF out).

```bash
python src/predict.py \
    --run-dir models \
    --input input.pdb \
    --output output.pdb
```
