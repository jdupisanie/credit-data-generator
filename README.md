# Credit Risk Modeling

## Overview
Synthetic credit-risk data generation and downstream modeling workflow (logistic + neural network + Cox), with consistent model comparison outputs.

## Project structure
- `input_parameters/`: simulation configuration (`variables.json`, `global_parameters.json`)
- `src/`: simulation core logic
- `outputs/simulator/`: generated synthetic dataset + metadata
- `analytics/data_analysis/`: analysis scripts
- `analytics/data_analysis/artifacts/`: staged analysis/model outputs

## Quick run order
1. Run interactive orchestrator: `python run_pipeline.py`
2. Choose:
   - `1` for data creation pipeline (includes auto-generated data dictionary)
   - `2` for model training + comparison pipeline
   - `3` for archiving outputs into a timestamped folder

For detailed stage-by-stage paths, see `analytics/data_analysis/README.md`.
For a full walkthrough, see `USER_GUIDE.md`.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
