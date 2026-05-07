# MLOps Pipeline Project - Titanic Classifier with Hydra

<a target="_blank" href="https://cookiecutter-data-science.drivendaily.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Complete, configurable machine learning pipeline for the Titanic classification competition. This project demonstrates end-to-end ML workflows with **Hydra** configuration management, enabling reproducible experiments, hyperparameter sweeps, and easy experiment variants.

## Overview

This project trains a production-ready Titanic classifier through a fully automated pipeline that:
- Downloads data directly from the Kaggle Titanic competition API
- Engineers domain-specific features (family size, cabin deck, title parsing, etc.)
- Preprocesses with scikit-learn `ColumnTransformer` (imputation, scaling, one-hot encoding)
- Compares multiple classifiers (Logistic Regression, Random Forest) via cross-validation
- Selects the best model based on configurable metrics
- Saves trained pipelines and detailed training reports

**Key Innovation**: All pipeline parameters (data paths, feature lists, model hyperparameters, CV splits, scoring metrics) are configurable via Hydra YAML files, enabling reproducible experiments without code changes.

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
MLOps-practice2/
├── conf/                          # Hydra configuration files
│   ├── config.yaml                # Root config (defaults list)
│   └── pipeline/                  # Pipeline config group
│       ├── titanic_baseline.yaml  # Baseline experiment variant
│       └── titanic_alt.yaml       # Alternative experiment variant
├── data/                          # Data directory (auto-created)
│   ├── external/                  # Third-party data
│   ├── interim/                   # Intermediate transformed data
│   ├── processed/                 # Final processed datasets
│   └── raw/                       # Original immutable data (Titanic CSV)
├── docs/                          # Documentation
├── models/                        # Trained pipelines (auto-created)
├── notebooks/                     # Jupyter notebooks (W&B tutorial)
├── reports/                       # Generated training reports (auto-created)
├── src/                           # Source code
│   ├── pipeline.py                # Core training pipeline functions
│   ├── logger.py                  # Logging utilities
│   ├── fake/                      # Custom estimators
│   └── training/                  # Training modules (legacy structure)
├── trainer.py                     # Hydra entry point for training
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
├── Makefile                       # Convenience commands
└── README.md                      # This file
```

## Tools & Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| **Hydra** | 1.3.2+ | Configuration management and CLI argument parsing |
| **OmegaConf** | 2.3.0+ | Configuration objects and interpolation |
| **scikit-learn** | 1.6.0+ | ML models: LogisticRegression, RandomForestClassifier |
| **Kaggle API** | 1.7.4.5+ | Download Titanic competition data |
| **pandas** | 2.0.1+ | Data manipulation and feature engineering |
| **joblib** | Latest | Serialize/deserialize trained pipelines |
| **loguru** | 0.7.2+ | Structured logging |
| **uv** | Latest | Python dependency management (fast alternative to pip) |

## Quick Start

### 1. Run the Default Pipeline

```bash
python trainer.py
```

This executes the pipeline with the default configuration (`titanic_baseline`):
1. **Downloads Titanic data** from Kaggle API
2. **Engineers features** (family size, cabin deck, title, etc.)
3. **Trains two models** (Logistic Regression + Random Forest)
4. **Selects the best** model based on cross-validation accuracy
5. **Saves artifacts** to `models/titanic/titanic_pipeline.pkl` and `reports/titanic/training_report.json`

### 2. View Hydra Configuration

To see the resolved configuration before running:

```bash
python trainer.py --help
```

Output shows all configuration groups, parameters, and their values. Example:
```
Configuration groups:
  pipeline: titanic_alt, titanic_baseline (default: titanic_baseline)

Config:
  experiment_name: titanic_baseline
  data:
    competition_name: titanic
    raw_data_dir: data/raw/titanic
  ... (full resolved config)
```

### 3. Run with Different Configuration Group

Switch to the alternative experiment variant:

```bash
python trainer.py pipeline=titanic_alt
```

This uses settings from `conf/pipeline/titanic_alt.yaml` (e.g., different CV settings, selection metric).

---

## Hydra Configuration System

### Overview

**Hydra** is a framework for configuring complex applications. It eliminates hardcoded parameters and enables reproducible ML experiments through declarative YAML configurations.

Key benefits:
- **Reproducibility**: All parameters versioned in config files
- **CLI Overrides**: Override any parameter without code changes
- **Config Groups**: Manage experiment variants (baseline vs. alternative)
- **Variable Interpolation**: Reference values across config (DRY principle)
- **Multirun**: Run experiments with parameter sweeps in parallel

### Configuration Architecture

The configuration is organized hierarchically:

```
conf/
├── config.yaml                    # Root config (entry point)
└── pipeline/                      # Config group "pipeline"
    ├── titanic_baseline.yaml      # Option 1
    └── titanic_alt.yaml           # Option 2
```

**Root Config** (`conf/config.yaml`):
```yaml
defaults:
  - pipeline: titanic_baseline    # Use this pipeline config by default
  - _self_
```

This tells Hydra to:
1. Look for a `pipeline` config group
2. Select the `titanic_baseline` variant as default
3. Apply root-level overrides after the selected config (\_self\_)

### Config Groups Explained

A **config group** is a directory containing alternative configurations for a component.

**Pipeline Config Group** (`conf/pipeline/`):
- Each YAML file represents an experiment variant
- All files in the group are options accessible via `pipeline=<name>` CLI flag
- Allows easy experiment comparison without code duplication

**Available Pipeline Variants**:

#### `titanic_baseline.yaml`
- **Selection Metric**: Accuracy
- **CV Splits**: 5
- **Test Split**: 20%
- **Random State**: 42
- **RF Estimators**: 300
- **Purpose**: Production baseline

#### `titanic_alt.yaml`
- **Selection Metric**: ROC-AUC (better for imbalanced data)
- **CV Splits**: 5
- **Test Split**: 25%
- **Random State**: 123
- **RF Estimators**: 500
- **Purpose**: Alternative hyperparameter exploration

### Variable Interpolation

Hydra supports variable interpolation using `${path.to.value}` syntax. This enables:
- Single source of truth (change once, applies everywhere)
- Consistency across nested configs

**Example** (from `titanic_baseline.yaml`):
```yaml
training:
  random_state: 42           # Defined once

models:
  logistic_regression:
    random_state: ${pipeline.training.random_state}  # References training.random_state
  random_forest:
    random_state: ${pipeline.training.random_state}  # Same value, no duplication
```

When resolved, both models use `random_state: 42`. Change line 42 value once, and all models automatically use the new value.

### Object Instantiation via `_target_`

Hydra can dynamically instantiate Python objects from YAML using the `_target_` key. This enables:
- Configuration of class instantiation (which class? what arguments?)
- Parameter overrides via CLI without touching code

**Example** (from `titanic_baseline.yaml`):
```yaml
models:
  logistic_regression:
    _target_: sklearn.linear_model.LogisticRegression
    class_weight: balanced
    max_iter: 2000
    random_state: ${pipeline.training.random_state}
    solver: liblinear

  random_forest:
    _target_: sklearn.ensemble.RandomForestClassifier
    class_weight: balanced
    n_estimators: 300
    min_samples_leaf: 2
    max_depth: null
    n_jobs: -1
    random_state: ${pipeline.training.random_state}
```

**In Code** (`src/pipeline.py`):
```python
from hydra.utils import instantiate

def _build_model_candidates(cfg: DictConfig):
    """Build model instances from config using instantiate."""
    candidates = {}
    for model_name, model_cfg in cfg.models.items():
        # instantiate() uses _target_ to dynamically create the object
        candidates[model_name] = instantiate(model_cfg)
    return candidates
```

When `instantiate()` processes the config:
1. Reads `_target_: sklearn.linear_model.LogisticRegression`
2. Imports the class dynamically
3. Creates instance with remaining keys as kwargs: `LogisticRegression(class_weight='balanced', max_iter=2000, ...)`

**Benefit**: Change `n_estimators` from CLI without recompiling or editing code:
```bash
python trainer.py pipeline.models.random_forest.n_estimators=500
```

### CLI Overrides

Override any configuration parameter from the command line using dot notation:

```bash
# Change a single value
python trainer.py pipeline.training.random_state=100

# Change multiple values
python trainer.py pipeline.training.cv_splits=10 pipeline.training.test_size=0.15

# Override nested values
python trainer.py pipeline.models.random_forest.n_estimators=200 pipeline.models.random_forest.max_depth=10

# Override entire pipeline config
python trainer.py pipeline=titanic_alt
```

All overrides are logged to the output and saved to the output directory for reproducibility.

---

## Running Experiments

### Single Run (Default)

Execute with default or custom configuration:

```bash
# Default (titanic_baseline)
python trainer.py

# Alternative config
python trainer.py pipeline=titanic_alt

# Custom hyperparameter sweep (single run)
python trainer.py pipeline.training.cv_splits=10 pipeline.training.random_state=42
```

**Output Directory Structure**:
```
outputs/
└── <YYYY-MM-DD>/
    └── <HH-MM-SS>/
        ├── .hydra/
        │   ├── config.yaml            # Resolved config (for this run)
        │   ├── hydra.yaml             # Hydra runtime config
        │   └── overrides.yaml         # CLI overrides applied
        ├── logs/                      # Execution logs
        └── training_report.json       # Pipeline output (training metrics)
```

Each run gets a timestamped directory ensuring no overwrites.

### Multirun (Experiment Sweeps)

Run multiple configurations in parallel or series:

```bash
# Test both pipeline variants
python trainer.py --multirun pipeline=titanic_baseline,titanic_alt

# Sweep hyperparameters across both pipelines
python trainer.py --multirun \
  pipeline=titanic_baseline,titanic_alt \
  pipeline.training.random_state=42,100,200

# Sweep only the alternative pipeline
python trainer.py --multirun pipeline=titanic_alt \
  pipeline.training.cv_splits=5,10 \
  pipeline.models.random_forest.n_estimators=200,300,400
```

**Multirun Output Structure**:
```
multirun/
└── <YYYY-MM-DD>/
    └── <HH-MM-SS>/
        ├── 0/                         # First experiment run
        │   ├── .hydra/
        │   └── ...
        ├── 1/                         # Second experiment run
        │   ├── .hydra/
        │   └── ...
        ├── 2/                         # Third experiment run
        └── ...
```

Each run gets a numbered subdirectory. Inspect `.hydra/config.yaml` in each to see what parameters were used.

### Using Abbreviation

For convenience, `-m` is equivalent to `--multirun`:

```bash
python trainer.py -m pipeline=titanic_baseline,titanic_alt
```

---

## Understanding the Training Report

After training, `reports/titanic/training_report.json` contains:

```json
{
  "experiment_name": "titanic_baseline",
  "test_split": {
    "accuracy": 0.82,
    "f1": 0.79,
    "precision": 0.81,
    "recall": 0.77,
    "roc_auc": 0.88
  },
  "cross_validation": {
    "logistic_regression": {
      "accuracy": [0.81, 0.79, 0.80, 0.82, 0.78],
      "f1": [0.78, 0.76, 0.77, 0.79, 0.75],
      ...
    },
    "random_forest": {
      "accuracy": [0.83, 0.82, 0.81, 0.84, 0.80],
      ...
    }
  },
  "selected_model": "random_forest",
  "selection_metric": "accuracy"
}
```

**Fields**:
- `test_split`: Metrics on holdout test set
- `cross_validation`: Per-fold metrics for each candidate model
- `selected_model`: Which model won (based on `selection_metric`)
- `selection_metric`: The metric used to select the best model

---

## Extending the Configuration

### Add a New Experiment Variant

Create `conf/pipeline/titanic_experiment3.yaml` (copy and modify an existing one):

```yaml
experiment_name: titanic_experiment3

data:
  # ... (same as baseline)

preprocessing:
  numeric_features:
    # Add new features here
  categorical_features:
    # Modify feature lists

training:
  test_size: 0.15
  random_state: 999
  cv_splits: 7
  selection_metric: roc_auc

models:
  # Modify model parameters or add new models
```

Then run:
```bash
python trainer.py pipeline=titanic_experiment3
```

### Add a New Model to Compare

In the chosen config (e.g., `titanic_baseline.yaml`), add to `models` section:

```yaml
models:
  logistic_regression:
    _target_: sklearn.linear_model.LogisticRegression
    # ... (existing)
  
  random_forest:
    _target_: sklearn.ensemble.RandomForestClassifier
    # ... (existing)
  
  gradient_boosting:
    _target_: sklearn.ensemble.GradientBoostingClassifier
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5
    random_state: ${pipeline.training.random_state}
```

The pipeline will automatically compare all three models during cross-validation.

---

## Pipeline Implementation Details

### Architecture Overview

The training pipeline consists of these stages:

1. **Data Download** → Kaggle API retrieves Titanic competition files
2. **Feature Engineering** → Custom `TitanicFeatureEngineer` adds domain features
3. **Preprocessing** → scikit-learn `ColumnTransformer` (imputation, scaling, encoding)
4. **Model Training** → Multiple classifiers trained with cross-validation
5. **Model Selection** → Best model chosen by configurable metric
6. **Artifact Storage** → Trained pipeline + training report saved

### Code Organization

**Entry Point** (`trainer.py`):
- Decorated with `@hydra.main(config_path="conf", config_name="config", version_base=None)`
- Hydra resolves config from `conf/` and passes as `DictConfig`
- Logs resolved config before training starts
- Calls `train_titanic_pipeline(cfg.pipeline, logger)`

**Core Pipeline** (`src/pipeline.py`):
- Imports all functions parameterized by `DictConfig cfg`
- `TitanicFeatureEngineer`: Custom transformer for domain features
- `_build_preprocessor(cfg)`: Creates ColumnTransformer from feature lists in config
- `_build_model_candidates(cfg)`: Uses `instantiate()` to create models from `_target_` specs
- `download_titanic_competition_data(cfg, logger)`: Kaggle API integration
- `train_titanic_pipeline(cfg, logger)`: Main orchestrator function

### Feature Engineering

The `TitanicFeatureEngineer` class extracts Titanic-specific features:
- **FamilySize**: Sum of SibSp + Parch + 1 (passenger + family)
- **IsAlone**: Binary indicator for single passengers
- **HasCabin**: Binary indicator for cabin number presence
- **Title**: Extracted from Name, with rare titles grouped
- **Deck**: Extracted from Cabin letter (A, B, C, etc.)
- **TicketPrefix**: Prefix of ticket number

These features improve model performance by capturing domain knowledge.

### Model Comparison

The pipeline trains multiple candidate models in parallel using `StratifiedKFold` cross-validation:
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble model with configurable depth/estimators

For each fold, metrics are computed (accuracy, F1, precision, recall, ROC-AUC). The model with the best mean score in `selection_metric` is selected.

---

## Available Make Commands

```bash
make help       # Show all available commands
make format     # Format code with Black + isort
make lint       # Lint with Ruff
make clean      # Remove compiled artifacts
```

---

## Using W&B for Experiment Tracking (Optional)

The project includes a W&B tutorial notebook for tracking experiments:

```bash
jupyter lab notebooks/wandb101.ipynb
```

This enables:
- Versioned experiment history
- Hyperparameter tracking
- Metric visualization
- Artifact storage

Optional integration can be added to `trainer.py` or `src/pipeline.py` to auto-log metrics to W&B.

---

## Troubleshooting

### Kaggle API Credentials Not Found
**Error**: `OSError: Kaggle API credentials not found at ~/.kaggle/kaggle.json`

**Solution**: 
1. Go to https://kaggle.com/account/settings/account
2. Click "Create New API Token"
3. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Hydra Output Directory Permissions
If Hydra complains about write permissions in the `outputs/` directory, ensure it exists:
```bash
mkdir -p outputs
```

### Import Errors After Dependency Update
If you get import errors, rebuild the lock file:
```bash
uv sync --upgrade
```

---

## Common Use Cases

### Use Case 1: Baseline Pipeline
Run with default settings (titanic_baseline, 5-fold CV, accuracy metric):
```bash
python trainer.py
```

### Use Case 2: Compare Experiment Variants
Run baseline and alternative side-by-side:
```bash
python trainer.py -m pipeline=titanic_baseline,titanic_alt
```

Then compare `multirun/<date>/<time>/0/` and `multirun/<date>/<time>/1/` training reports.

### Use Case 3: Hyperparameter Sweep
Test multiple random states and CV splits:
```bash
python trainer.py -m \
  pipeline.training.random_state=42,100,200 \
  pipeline.training.cv_splits=5,10
```

This creates 6 experiment runs (3 × 2 combinations).

### Use Case 4: Production Optimization
Optimize Random Forest for the alternative pipeline:
```bash
python trainer.py -m pipeline=titanic_alt \
  pipeline.models.random_forest.n_estimators=100,200,300,400,500 \
  pipeline.models.random_forest.max_depth=5,10,15,20
```

### Use Case 5: Single Experiment with Custom Parameters
```bash
python trainer.py \
  pipeline.training.test_size=0.15 \
  pipeline.training.selection_metric=roc_auc \
  pipeline.models.random_forest.max_depth=8
```

---

## Key Takeaways

1. **Hydra** powers configuration management — no config files means no reproducibility
2. **Config Groups** enable experiment variants — swap entire configurations via CLI
3. **Interpolation** prevents duplication — define values once, reference everywhere
4. **Instantiation** keeps Python out of YAML — change model types/params without code
5. **Multirun** enables batch experiments — sweep parameters and compare results systematically
6. **CLI Overrides** provide flexibility — tweak parameters on-the-fly for quick experiments
7. **Timestamped Outputs** ensure reproducibility — every run is logged and versioned

---

## License

MIT License - See LICENSE file for details.
