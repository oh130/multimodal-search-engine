# Data Pipeline

This directory contains the preprocessing scripts used to build recommendation training data from the raw H&M dataset.

The pipeline starts from three raw CSV files:
- `data/raw/customers.csv`
- `data/raw/articles.csv`
- `data/raw/transactions_train.csv`

It produces the processed files used by search and recommendation experiments.

## Files

- `build_customer_features.py`
  - builds `data/processed/customer_features.csv`
- `build_article_features.py`
  - builds `data/processed/articles_feature.csv`
- `build_item_features.py`
  - builds item-level popularity and freshness features
- `build_ranking_train_data.py`
  - builds the ranking training dataset with positive and sampled negative rows
- `run_data_pipeline.py`
  - runs the full pipeline in the correct order

## Recommended Entry Point

If you only have the three raw dataset files and want the full pipeline to run in order, use:

```bash
python data_pipeline/run_data_pipeline.py
```

## Mode Guide

The scripts support two runtime modes:
- `test`
- `production`

The mode is set at the top of `run_data_pipeline.py`:

```python
# MODE = "production"
MODE = "test"
```

`run_data_pipeline.py` passes the selected mode to:
- `build_item_features.py`
- `build_ranking_train_data.py`

`build_customer_features.py` and `build_article_features.py` always generate their standard processed outputs.

## Output Files

Common outputs:
- `data/processed/customer_features.csv`
- `data/processed/articles_feature.csv`

Test mode outputs:
- `data/processed/item_features_test.csv`
- `data/processed/train_data_test.csv`

Production mode outputs:
- `data/processed/item_features.csv`
- `data/processed/train_data_production.csv`

## Manual Execution Order

If you want to run each script manually, use this order:

```bash
python data_pipeline/build_customer_features.py
python data_pipeline/build_article_features.py
python data_pipeline/build_item_features.py
python data_pipeline/build_ranking_train_data.py
```

## Notes

- Run commands from the repository root.
- Keep raw CSV files local only; do not commit them.
- `build_ranking_train_data.py` expects `customer_features.csv` and `articles_feature.csv` to exist first.
- The ranking dataset is purchase-based and uses sampled negatives rather than impression logs.
