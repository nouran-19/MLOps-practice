# MLOps Pipeline Project

<a target="_blank" href="https://cookiecutter-data-science.drivendaily.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Complete MLOps Pipeline for ML Practitioners - an educational project that now trains a Titanic classifier end to end.

## Overview

This project demonstrates a complete end-to-end machine learning pipeline. It includes Titanic competition data download from Kaggle, feature engineering, preprocessing with scikit-learn transformers, model selection across multiple classifiers, and model/report saving.

## Prerequisites

- Python 3.11+
- uv (for dependency management)
- Kaggle API credentials configured at `~/.kaggle/kaggle.json` or through environment variables

Install dependencies:
```bash
uv sync
```

## Project Structure

```
ITI-MLOps/
├── conf/                     # Configuration files (placeholder)
├── data/                     # Data directory
│   ├── external/             # Third-party data
│   ├── interim/              # Intermediate transformed data
│   ├── processed/            # Final processed datasets
│   └── raw/                  # Original immutable data
├── docs/                     # MkDocs documentation
├── models/                   # Trained models and preprocessors
├── notebooks/                # Jupyter notebooks (W&B tutorial)
├── references/               # Data dictionaries and manuals
├── reports/                  # Generated analysis (HTML, PDF, etc.)
├── src/                      # Source code
│   ├── fake/                 # Custom estimators
│   ├── training/             # Training pipeline modules
│   │   ├── download_data.py  # Data download (Kaggle)
│   │   ├── process_data.py   # Data preprocessing & split
│   │   ├── train.py          # Model training & hyperopt
│   │   └── evaluate.py       # Model evaluation
│   └── logger.py             # Logging utilities
├── trainer.py                # Entry point for training pipeline
├── pyproject.toml            # Project configuration (uv)
├── uv.lock                   # Locked dependencies
└── Makefile                  # Convenience commands
```

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **W&B** | Experiment tracking & visualization (optional) |
| **scikit-learn** | Machine learning models |
| **Hyperopt** | Bayesian hyperparameter optimization |
| **kagglehub** | Kaggle dataset download |
| **skore** | Model evaluation & reporting |
| **uv** | Python dependency management |

## Quick Start

### 1. Data Preparation

The pipeline automatically downloads the Titanic competition files from Kaggle. Make sure your Kaggle account has accepted the competition rules and that your API token is configured.

### 2. Run the Pipeline

```bash
python trainer.py
```

This will:
1. **Download data** - Fetch Titanic competition files from Kaggle
2. **Engineer features** - Build Titanic-specific features inside a scikit-learn pipeline
3. **Train models** - Compare at least two scikit-learn classifiers and select the best one
4. **Evaluate and save** - Persist the trained pipeline and a JSON training report

### 3. View Logs

Logs are stored in `logs/` directory with timestamps.

## Pipeline Details

The pipeline is orchestrated by `trainer.py` and consists of four stages:

### 1. Data Download (`src/pipeline.py`)
- Downloads the Titanic competition archive using the Kaggle API
- Extracts the raw CSV files into `data/raw/titanic/`

### 2. Feature Engineering and Preprocessing (`src/pipeline.py`)
- Adds Titanic-specific features like family size, cabin deck, title, and ticket prefix
- Uses a `ColumnTransformer` with numeric imputation/scaling and categorical one-hot encoding

### 3. Model Training (`src/pipeline.py`)
- Compares logistic regression and random forest pipelines with cross-validation
- Selects the best model based on validation accuracy
- Fits the final pipeline on the training split
- Saves the trained pipeline to `models/titanic/titanic_pipeline.pkl`

### 4. Model Evaluation and Reporting (`src/pipeline.py`)
- Evaluates the selected model on a holdout split
- Saves metrics and cross-validation results to `reports/titanic/training_report.json`

## Using W&B for Experiment Tracking

The project can integrate with W&B for experiment tracking:
```bash
jupyter lab notebooks/wandb101.ipynb
```

## Available Make Commands

```bash
make help       # Show available commands
make requirements  # Install Python dependencies
make lint       # Run linting checks
make format     # Format code
make clean      # Clean compiled Python files
```

## Extending the Project

1. **Add new data sources**: Place in `data/external/`
2. **Add new features**: Modify `src/training/process_data.py`
3. **Add new models**: Add to `src/training/` or `src/fake/` directory
4. **Add experiments**: Use configs in `conf/`

## License

MIT License - See LICENSE file.
