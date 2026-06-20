# Credit Risk Prediction in P2P Lending

A statistical learning project based on the Bondora public loan dataset.

This project studies credit-risk prediction in peer-to-peer lending. Rather
than treating model comparison as the only objective, the analysis examines
several reasons why a conventional credit-risk pipeline can give misleading
results:

- default is not represented by a single unambiguous label;
- discrimination and probability calibration measure different properties;
- predicted probabilities depend on the target horizon;
- recent loans may be right-censored before their final outcome is observed;
- model performance may change across countries and origination periods.

The analysis compares logistic regression, LightGBM, TabPFN-3, and Bondora's
published Probability of Default (PoD) score. Split-conformal prediction is
used as an additional uncertainty-quantification layer.

## Research questions

1. How sensitive are credit-risk results to the definition of default?
2. Why can a model have acceptable AUC but poor probability calibration?
3. How does the prediction horizon affect the interpretation of default
   probability?
4. Can conformal prediction provide a more transparent uncertainty framework
   under distribution shift?

## Dataset

The source data are available from the Kaggle dataset
[Bondora P2P Loans](https://www.kaggle.com/datasets/marcobeyer/bondora-p2p-loans/data),
published by Marco Beyer. Kaggle identifies this dataset as CC0 1.0.

The downloaded source file is not included in this repository. Users should
download it from Kaggle and review the current dataset terms before use.

### Analysis sample

The figures below describe the filtered sample used in this project, not the
complete Kaggle table:

- period covered by the project snapshot: 2009–2024 H2;
- analysis sample: 264,369 closed loans;
- included countries: Estonia (EE), Finland (FI), and Spain (ES);
- excluded countries: the Netherlands (NL) and Slovakia (SK), due to small
  sample sizes;
- unit of observation: one loan-level record.

Sample counts can change if Kaggle updates the dataset or if the filtering
rules are changed. See [DATA.md](DATA.md) for data-handling details.

## Methods

The main pipeline:

1. parses dates and constructs alternative default labels;
2. distinguishes one-year and lifetime prediction targets;
3. removes post-origination and outcome-derived fields;
4. audits candidate features for unusually strong univariate association with
   the target;
5. creates country-specific, time-ordered train/calibration/test splits;
6. evaluates logistic regression and a selected gradient-boosting backend;
7. reports AUC, Brier score, calibration error, and calibration diagnostics;
8. evaluates cross-country conformal coverage.

The default tree backend is LightGBM. XGBoost and sklearn's histogram gradient
boosting implementation can be selected explicitly for comparison.

## Repository contents

- `bondora_reanalysis.py`: data preparation, label construction, leakage
  audit, baseline models, calibration analysis, and conformal evaluation.
- `make_figures.py`: creates the main analysis figures.
- `run_tabpfn_remote.py`: optional experiment using the hosted Prior Labs
  TabPFN API.
- `conformal_core.py`: reusable split-conformal utilities for binary
  prediction sets and conformal selection.
- `prepare_conformal_data.py`: builds the conformal master file from the main
  pipeline outputs and runs data-readiness checks.
- `run_conformal_experiments.py`: runs the extended conformal-inference
  experiments, including group-conditional coverage, distribution shift,
  online adaptation, label robustness, selection, and TabPFN wrapping.
- `DATA.md`: data source, expected directory layout, and publication rules.
- `requirements.txt`: dependencies for the local analysis.
- `requirements-tabpfn.txt`: additional dependency for the hosted TabPFN run.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Place the downloaded CSV at:

```text
data/LoanData.csv
```

## Run the analysis

```bash
python bondora_reanalysis.py \
  --csv data/LoanData.csv \
  --data-dir data \
  --fig-dir figures/pipeline \
  --gbdt-backend lightgbm
```

Available tree backends are `lightgbm`, `xgboost`, `sklearn`, and `auto`.
Reported experiments should use an explicit backend because `auto` depends on
which optional packages are installed.

Create the final figures with:

```bash
python make_figures.py
```

## Extended conformal experiments

After running the main pipeline, build the derived file used by the conformal
experiments:

```bash
python prepare_conformal_data.py build-master --data-dir data
```

Optional readiness checks:

```bash
python prepare_conformal_data.py check-tabpfn --data-dir data
python prepare_conformal_data.py diagnose-window --data-dir data
```

Run all conformal-inference experiments:

```bash
python run_conformal_experiments.py all \
  --master data/conformal_master.csv \
  --outdir figures/conformal
```

Individual experiments can also be run:

```bash
python run_conformal_experiments.py four-score --master data/conformal_master.csv
python run_conformal_experiments.py mondrian --master data/conformal_master.csv
python run_conformal_experiments.py shift-break --master data/conformal_master.csv
python run_conformal_experiments.py online-adapt --master data/conformal_master.csv
python run_conformal_experiments.py label-robust --master data/conformal_master.csv
python run_conformal_experiments.py selection --master data/conformal_master.csv
python run_conformal_experiments.py tabpfn-native --master data/conformal_master.csv
```

## Optional TabPFN experiment

`run_tabpfn_remote.py` sends preprocessed feature matrices to the hosted Prior
Labs API. This is external data processing and is disabled by default.

First inspect the selected features without making an API request:

```bash
python -m pip install -r requirements-tabpfn.txt
python run_tabpfn_remote.py --dry-run-features
```

After confirming that remote processing is permitted, set the token in an
environment variable and opt in explicitly:

```powershell
$env:PRIORLABS_API_KEY="..."
python run_tabpfn_remote.py --allow-remote-processing
```

API tokens must not be committed or supplied as command-line arguments.

## Reproducibility

- Train, calibration, and test subsets are ordered by origination date within
  each country.
- Sampling uses fixed random seeds unless otherwise stated.
- Prediction columns identify the estimator that generated them.
- Model metrics are evaluated on held-out test data.
- Raw data and loan-level generated outputs are excluded by `.gitignore`.

For a reported result, record the Kaggle download date, source-file checksum,
Python version, installed package versions, and the exact command used.

## Data and software licensing

The Kaggle dataset and this repository's source code are separate works. The
dataset is not redistributed here and is governed by the terms shown on its
Kaggle page. A software license should be added to this repository before
granting reuse rights for the code.

