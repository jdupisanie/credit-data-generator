# Credit Risk Modeling

![GitHub release](https://img.shields.io/github/v/release/jdupisanie/credit-data-generator)

## Overview
Synthetic credit-risk data generation and downstream modeling workflow (logistic + neural network + Cox), with consistent model comparison outputs.

## Academic Background

This project implements and operationalizes the simulation methodology introduced in:

**du Pisanie, J., Allison, J. S., & Visagie, J. (2023). _A Proposed Simulation Technique for Population Stability Testing in Credit Risk Scorecards_. Mathematics, 11(2), 492.**  
📄 https://doi.org/10.3390/math11020492

The published article provides the theoretical foundation and motivation for the synthetic data generation techniques used in this software. Researchers using this tool in academic work should cite both the repository and the original article.


## Project structure
- `input_parameters/`: simulation configuration (`variables.json`, `global_parameters.json`)
- `src/`: simulation core logic
- `analytics/data_analysis/artifacts/01_datasets/`: generated synthetic dataset + metadata
- `analytics/data_analysis/`: analysis scripts
- `analytics/data_analysis/artifacts/`: staged analysis/model outputs

## Quick run order
1. Run interactive orchestrator: `python run_pipeline.py`
2. Choose:
   - `1` for data creation pipeline (includes auto-generated data dictionary)
   - `2` for model training + comparison pipeline
   - `3` for archiving outputs into a timestamped folder

## Frontend UI
1. Install dependencies (from project root): `pip install -r requirements.txt`
2. Start the FastAPI UI server: `python -m uvicorn frontend.main:app --reload`
3. Open the UI in your browser: `http://127.0.0.1:8000`

For detailed stage-by-stage paths, see `analytics/data_analysis/README.md`.
For a full walkthrough, see `USER_GUIDE.md`.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in a publication or academic project, please cite both:

**Software**

du Pisanie, J. (2026). *Credit Data Generator (Version 1.0.0)* [Computer software]. GitHub. https://github.com/jdupisanie/credit-data-generator

**Methodology article:**

du Pisanie J, Allison JS, Visagie J. A Proposed Simulation Technique for Population Stability Testing in Credit Risk Scorecards. Mathematics. 2023; 11(2):492. https://doi.org/10.3390/math11020492