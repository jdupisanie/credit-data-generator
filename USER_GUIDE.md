# User Guide

This guide explains how to use the project day-to-day, especially the two interactive entry points:

- `run_pipeline.py` (run the full workflow)
- `input_parameters/edit_variables.py` (edit simulation inputs safely)

---

## 1) First-time setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2) Main workflow (`run_pipeline.py`)

Run:

```powershell
python run_pipeline.py
```

You will see a menu:

1. `Data creation`
2. `Model training + comparison`
3. `Archive outputs`
4. `Exit`

The menu stays open after each run, so you can do `1` then `2` in one session.

### Option 1: Data creation

Runs all data-prep stages:

- Generate simulated dataset (`main.py`)
- Generate data dictionary (`generate_data_dictionary.py`)
- Generate variable charts (`variable_band_charts.py`)
- Split train/test (`split_train_test.py`)
- Run variable selection (`variable_selection.py`)
- Build WOE datasets (`woe_encode.py`)

### Option 2: Model training + comparison

You are asked whether to use WOE data (`y/n`, default `y`), then it runs:

- Logistic regression training
- Neural network training
- Cox hazard model training
- Cross-model comparison (AUC, ROC, confusion matrix, etc.)

### Option 3: Archive outputs

Moves generated outputs into a timestamped folder under `archive/` so your workspace is clean but recoverable.

---

## 3) Input editor (`input_parameters/edit_variables.py`)

Run:

```powershell
python input_parameters/edit_variables.py
```

Use this when you want to change:

- variable `distribution_pct`
- variable `bad_rate_ratio`
- global parameters like:
  - `simulation_population`
  - `global_bad_rate_pct`
  - `train_set_pct`
  - `test_set_pct`

### Safety checks before save

The editor validates:

- each variable band distribution sums to `100`
- no invalid/negative distributions
- `bad_rate_ratio > 0`
- train/test percentages are valid and sum to `100`
- basic global parameter sanity

If validation fails, save is blocked and errors are shown.

### Save behavior

- A backup is created before writing:
  - `input_parameters/variables.json.bak`
  - `input_parameters/global_parameters.json.bak`
- Exit label is dynamic:
  - `Exit` if no unsaved changes
  - `Exit without saving` if there are unsaved changes

---

## 4) Where outputs go

- Raw simulated data: `analytics/data_analysis/artifacts/01_datasets/`
- Analysis/model outputs: `analytics/data_analysis/artifacts/`
  - `00_documentation/`
  - `01_datasets/`
  - `02_feature_selection/`
  - `03_preprocessing/`
  - `04_models/`
  - `05_model_comparison/`
  - `06_visualizations/`

---

## 5) Typical usage pattern

1. Edit inputs: `python input_parameters/edit_variables.py`
2. Run pipeline option `1` (data creation)
3. Run pipeline option `2` (train + compare)
4. Review outputs in `analytics/data_analysis/artifacts/05_model_comparison/`
5. Optionally run pipeline option `3` to archive old outputs

---

## 6) Troubleshooting

- **`No module named ...`**
  - Reinstall dependencies: `pip install -r requirements.txt`
- **No PDF dictionary**
  - Ensure `reportlab` is installed (included in requirements)
- **Model comparison empty**
  - Make sure model metadata/artifacts exist under `analytics/data_analysis/artifacts/04_models/`
- **Validation fails in editor**
  - Fix reported fields (especially distribution totals and train/test percentages)
