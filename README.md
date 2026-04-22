# MLOps Pipeline Project

<a target="_blank" href="https://cookiecutter-data-science.drivendaily.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Complete MLOps Pipeline for ML Practitioners - An educational project for the ITI MLOps Course.

## Overview

This project demonstrates a complete end-to-end machine learning pipeline. It includes data download from Kaggle, data processing with train/test splits, hyperparameter optimization with Hyperopt, and model evaluation with skore.

## Prerequisites

- Python 3.11+
- uv (for dependency management)

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

The pipeline automatically downloads the Iris dataset from Kaggle. To use a different dataset:

```bash
cp your_data.csv data/raw/your_dataset.csv
```
Then update the dataset name in `trainer.py`.

### 2. Run the Pipeline

```bash
python trainer.py
```

This will:
1. **Download data** - Fetch Iris dataset from Kaggle
2. **Process data** - Split data into train/test sets
3. **Train model** - Run hyperparameter optimization with Hyperopt
4. **Evaluate** - Output model performance metrics

### 3. View Logs

Logs are stored in `logs/` directory with timestamps.

## Pipeline Details

The pipeline is orchestrated by `trainer.py` and consists of four stages:

### 1. Data Download (`src/training/download_data.py`)
- Downloads Iris dataset from Kaggle using `kagglehub`
- Saves raw data to `data/raw/Iris.csv`

### 2. Data Processing (`src/training/process_data.py`)
- Loads raw CSV data
- Splits data into train/test (85%/15%) with stratification
- Saves processed data as Parquet files in `data/processed/`

### 3. Model Training (`src/training/train.py`)
- Reads Parquet files and extracts features/target
- Creates label encoder/decoder for target variable
- Uses Hyperopt for Bayesian hyperparameter optimization
- Trains final model with optimal parameters
- Saves model to `models/fake/final_model.pkl`

### 4. Model Evaluation (`src/training/evaluate.py`)
- Loads trained model and test data
- Computes metrics (accuracy, precision, recall)
- Saves evaluation report as JSON in `reports/fake/evaluation_report.json`

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
