# Data Analysis Workflow

This folder is organized as a simple pipeline so each stage writes to a dedicated subfolder.

## Pipeline order

0. Data dictionary
   - Script: `analytics/data_analysis/generate_data_dictionary.py`
   - Output: `analytics/data_analysis/artifacts/00_documentation/`

1. Split data
   - Script: `analytics/data_analysis/split_train_test.py`
   - Output: `analytics/data_analysis/artifacts/01_datasets/`

2. Variable selection
   - Script: `analytics/data_analysis/variable_selection.py`
   - Output: `analytics/data_analysis/artifacts/02_feature_selection/`

3. WOE preprocessing
   - Script: `analytics/data_analysis/woe_encode.py`
   - Output: `analytics/data_analysis/artifacts/03_preprocessing/`

4. Model training
   - Logistic: `analytics/data_analysis/train_logistic_model.py`
   - Neural Network: `analytics/data_analysis/train_neural_network_model.py`
   - Cox: `analytics/data_analysis/train_cox_model.py`
   - Output: `analytics/data_analysis/artifacts/04_models/`

5. Model comparison
   - Script: `analytics/data_analysis/compare_models.py`
   - Output: `analytics/data_analysis/artifacts/05_model_comparison/`

6. Visual diagnostics
   - Script: `analytics/data_analysis/variable_band_charts.py`
   - Output: `analytics/data_analysis/artifacts/06_visualizations/`

## Notes

- Simulation output remains in `outputs/simulator/`.
- Training scripts append model runs to:
  - `analytics/data_analysis/artifacts/05_model_comparison/model_registry.csv`
- Defaults in scripts now point to this staged structure.
