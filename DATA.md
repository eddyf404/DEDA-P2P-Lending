# Data Handling

This repository does not redistribute the Bondora loan-level dataset or any
generated loan-level outputs.

## Source Data

Download the Bondora P2P Loans dataset from Kaggle:

https://www.kaggle.com/datasets/marcobeyer/bondora-p2p-loans/data

Place the downloaded source file at:

```text
data/LoanData.csv
```

The Kaggle dataset and this repository's source code are separate works.
Before using or redistributing derived results, review the current Kaggle page
and dataset terms.

## Local Directory Layout

The analysis scripts expect this local layout:

```text
data/
  LoanData.csv
  closed_loans.csv
  loans_clean.csv
  step5_predictions.csv
  conformal_master.csv

figures/
  pipeline/
  conformal/
```

Only source code and documentation should be committed. The `.gitignore` file
excludes raw data, generated CSV files, generated figures, slide decks, PDFs,
and large archives.

## Generated Files

`bondora_reanalysis.py` creates the core pipeline outputs, including:

- `data/closed_loans.csv`
- `data/loans_clean.csv`
- `data/step5_predictions.csv`
- `data/step5_model_horizon_metrics.csv`
- `data/step7_conformal_transfer.csv`

`prepare_conformal_data.py` merges the relevant pipeline outputs into:

- `data/conformal_master.csv`

`run_conformal_experiments.py` writes summary tables and figures to:

- `figures/conformal/`

## Optional Remote Processing

`run_tabpfn_remote.py` uses the hosted Prior Labs TabPFN API. This is optional
and must be explicitly enabled with `--allow-remote-processing`.

API tokens should be supplied through environment variables only. Do not commit
tokens, API responses containing private data, or loan-level generated outputs.

## Reproducibility Notes

For reported results, record:

- Kaggle download date and source-file checksum;
- Python version;
- installed package versions;
- exact commands used;
- whether hosted TabPFN predictions were generated;
- the Git commit of this repository.
