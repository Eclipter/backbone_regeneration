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

4. Export the best model to ONNX. Pass `--run-dir` as a path template that still contains the literal `{target_mode}` segment (do not substitute it yourself). One invocation exports every mode in `config.PER_MODE`, so each expanded run directory must exist and include a trained checkpoint.

```bash
python src/export_to_onnx.py --run-dir logs/fixed_swa/{target_mode}/baseline
```

## Usage

1. Make sure to set up the environment. See [Environment Setup](#environment-setup)

2. Predict the backbone. Pass `--run-dir` as a path template that still contains the literal `{target_mode}` segment (do not substitute it yourself). One invocation runs inference for every entry in `config.PER_MODE`, so each expanded run directory must exist and include a trained checkpoint. Input and output may be PDB or mmCIF independently (e.g. PDB in, mmCIF out).

```bash
python src/predict.py \
    --run-dir models/{target_mode} \
    --input input.pdb \
    --output output.pdb
```
