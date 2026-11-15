#  PAMPA prediction and scaffold generalization

## Project status

This package contains a frozen dataset, six saved data splits, trained model artifacts and predictions for 18 model runs, downstream analyses, and publication-figure scripts.

- Frozen dataset: `data/processed/clean.csv`
- Rows: 6,889
- Regression target: `PAMPA_clipped`
- Split designs: Random and Murcko Scaffold
- Seeds: 0, 1, 2
- Split ratio: 80:10:10
- Models: Random Forest, XGBoost, Chemprop D-MPNN
- Completed runs represented in `results/models/`: 2 split designs × 3 seeds × 3 models = 18

The original data, split indices, model artifacts, prediction files, and existing analysis outputs have been retained. This reorganization does not retrain models or change the scientific data.

## Directory guide

- `data/`: frozen model input and data-source documentation.
- `splits/`: saved train/validation/test indices and split metadata.
- `configs/`: model configuration files.
- `results/models/`: trained model artifacts, logs, metrics, and predictions.
- `results/analysis/`: derived analysis tables.
- `src/`: current scripts recommended for the main workflow.

Run commands from the project root (`cycpep/`).

### 1. Verify the frozen data and splits

```bash
python src/Data preparation and splits/03_verify_splits.py
python src/Data preparation and splits/04_full_split_integrity.py
```

`03_verify_splits.py` checks the saved split files. `04_full_split_integrity.py` performs the full train/validation/test integrity checks.

### 2. Model training (already completed)

The 18 existing runs are stored under `results/models/`. **Do not rerun training just to reproduce the packaged analysis.** The training entry points are retained:

- `src/Train/train_rf.py`
- `src/Train/train_xgboost.py`
- `src/Train/train_chemprop.py`
- shared utilities: `src/Train/train_common.py`

A new run should only be started deliberately, after checking its configuration and output path. Existing run directories are protected against accidental overwrite by the training code.

### 3. Rebuild analysis tables

The following scripts are the recommended analysis entry points. Run them from the project root and in dependency order:

```bash
python src/Analysis/01_scaffold_performance.py
python src/Analysis/02_final_model_comparison.py
python src/Analysis/03_scaffold_macro_micro.py
python src/Analysis/04_scaffold_bootstrap.py
python src/Analysis/05_scaffold_factor_analysis.py
python src/Analysis/06_cross_seed_similarity_analysis.py
python src/Analysis/07_outlier_sensitivity_analysis.py
python src/Analysis/08_scaffold_generalization_analysis.py
```

## Data and scientific interpretation

The target `PAMPA_clipped` is the clipped PAMPA endpoint defined during dataset preparation. Do not silently change the clipping bounds, stereochemical SMILES handling, frozen row order, or saved split indices when reproducing the packaged results.

Random Split and Scaffold Split answer different questions. Random Split estimates performance when test molecules may have close analogues in training; Scaffold Split provides a more demanding assessment of generalization to held-out Murcko scaffolds. Report both and avoid treating either as a substitute for the other.

Scaffold-level factor and outlier analyses are observational/exploratory. Associations do not establish causality. For scaffold-level summaries, always report the number of test molecules (`n_test`); single-molecule scaffolds should not be interpreted as stable estimates of scaffold-wide difficulty.

## Environment

See `requirements_notes.txt` for the environment guidance supplied with this package. Exact package versions should be captured from the environment used for a new run (for example, `python --version` and `python -m pip freeze`). GPU-compatible PyTorch/Chemprop installation depends on the target server and CUDA stack; follow the official Chemprop installation instructions rather than assuming a portable environment.

