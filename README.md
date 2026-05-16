# Base2Backbone

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Eclipter/base2backbone.git
cd base2backbone
```

2. Install Conda. See [Conda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

3. Download PyNAMod:

```bash
git clone https://github.com/intbio/PyNAMod.git pynamod
cd pynamod
```

4. Create a new environment with PyNAMod dependencies:

```bash
/opt/miniconda/bin/conda env create \
  --name base2backbone \
  --file pynamod/environment.yml
```

5. Install the project dependencies:

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

3. Analyze the results if necessary

```bash
python scripts/analyze.py
```

4. Export the best model to ONNX:

```bash
python scripts/export.py --run-id torsions/1/baseline
```

## Usage

1. Make sure to set up the environment. See [Setup](#setup)

2. Predict the backbone

```bash
base2backbone \
    --input input.pdb \
    --output output.pdb
```

Input and output may be PDB or mmCIF independently (e.g. PDB in, mmCIF out). By default **5'-terminal phosphate atoms** (`P`, `OP1`, `OP2`) are **not** predicted. Pass `--generate-5-prime-phosphate` to include them.

A tradeoff between speed and quality can be chosen by changing the number of ODE steps using the `--num-timesteps` argument. Default is 50.
