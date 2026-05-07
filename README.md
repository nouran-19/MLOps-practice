# MLOps Pipeline Project - Titanic Training with DVC + DagsHub

<a target="_blank" href="https://cookiecutter-data-science.drivendaily.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project trains a Titanic classifier and versions data, models, and reports with DVC.
The remote storage is connected to DagsHub so anyone can clone the repo and pull the same artifacts.

## What This Project Solves

In ML projects, code is not enough.
You also need to version:
- training data
- trained model files
- evaluation reports

Git handles code well, but not large artifacts.
DVC handles large artifacts and keeps them reproducible.
DagsHub provides the remote storage and ML-friendly collaboration layer.

## Quick Concepts (Beginner Friendly)

- Git: versions code files.
- DVC: versions data/model artifacts and tracks pipeline outputs.
- DagsHub: remote backend where DVC files are stored and shared.

Typical flow:
1. Run pipeline locally.
2. DVC tracks outputs and saves hashes in Git.
3. DVC pushes real artifact files to DagsHub remote.
4. Another machine clones repo and runs `dvc pull` to get exact artifacts.

## Project Structure

```
MLOps-practice2/
├── .dvc/                             # DVC config and internal metadata
├── conf/                             # Training configuration files
├── data/raw/titanic/                 # Titanic CSV files (tracked by DVC)
│   ├── train.csv.dvc
│   ├── test.csv.dvc
│   └── gender_submission.csv.dvc
├── models/titanic/                   # Trained pipeline artifact (DVC output)
├── reports/titanic/                  # Training report artifact (DVC output)
├── src/                              # Pipeline source code
├── trainer.py                        # Training entry point
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # Locked stage dependency/output hashes
└── pyproject.toml                    # Dependencies and project config
```

## Prerequisites

- Python 3.11+
- uv
- Kaggle credentials for Titanic data access
- DagsHub account and token

Install dependencies:

```bash
uv sync
```

## DVC Pipeline in This Repo

The pipeline stage is defined in `dvc.yaml`.
It runs training and tracks outputs.

Main stage:
- stage name: `train_titanic`
- command: run trainer script using local virtual environment
- dependencies: training code, config files, and raw Titanic CSV files
- outputs: model file
- metrics: training report JSON

Why this matters:
- If dependencies do not change, DVC can skip reruns.
- If code/data/config changes, DVC can rerun only what is needed.
- `dvc.lock` captures exact reproducible state.

### Example Stage Chain

For learning, this repo now also shows how one DVC stage can feed the next stage.

- `train_titanic` creates `models/titanic/titanic_pipeline.pkl`
- `score_titanic` uses that `.pkl` file as an input dependency
- If the model changes, DVC reruns only the downstream scoring stage

This is the key idea behind multi-stage ML pipelines: each file can be either an output of one stage or an input to another stage.

## DagsHub Connection

This repo uses DagsHub as DVC remote storage.

Configured remote (in `.dvc/config`):
- default remote points to DagsHub S3 endpoint
- credentials are stored locally in `.dvc/config.local` (not committed)

Important:
- Never commit `.dvc/config.local`.
- If token is exposed, rotate it in DagsHub immediately.

## First-Time Setup (One Machine)

1. Clone repo.
2. Install dependencies.
3. Configure DVC auth for DagsHub.
4. Pull artifacts from remote.

```bash
git clone https://github.com/nouran-19/MLOps-practice.git
cd MLOps-practice2
uv sync

# if needed, set remote credentials locally
# dvc remote modify --local origin access_key_id <dagshub_token>
# dvc remote modify --local origin secret_access_key <dagshub_token>

dvc pull
```

After `dvc pull`, tracked data/model/report files are available locally.

## Daily Workflow

### 1) Reproduce training pipeline

```bash
dvc repro
```

This runs the training stage and updates `dvc.lock` when outputs change.

### 2) Check what changed

```bash
dvc status
git status
```

### 3) Push artifacts to DagsHub

```bash
dvc push
```

### 4) Commit and push code metadata

```bash
git add dvc.yaml dvc.lock .dvc/config *.dvc .gitignore
git commit -m "Update training artifacts and DVC metadata"
git push
```

## Verify Reproducibility (Lab Requirement)

To prove setup works from scratch:

1. Clone into a fresh folder.
2. Run `uv sync`.
3. Run `dvc pull`.
4. Confirm model and report exist.

```bash
git clone https://github.com/nouran-19/MLOps-practice.git MLOps-practice2-verify
cd MLOps-practice2-verify
uv sync
dvc pull
```

Expected files after pull:
- `data/raw/titanic/train.csv`
- `data/raw/titanic/test.csv`
- `models/titanic/titanic_pipeline.pkl`
- `reports/titanic/training_report.json`

## Minimal Note on Hydra

Hydra is still used internally for training configuration in `trainer.py` and `conf/`.
For Lab 2 workflow, focus on DVC commands (`repro`, `push`, `pull`) and DagsHub remote sync. See lab 1 branch for more details

## Common Issues

### `401 Unauthorized` on `dvc push`
Cause:
- Missing or invalid DagsHub token in local DVC config.

Fix:
- Update local credentials for remote and retry push.

### Outputs tracked by Git instead of DVC
Cause:
- Artifact file was committed before being moved to DVC tracking.

Fix:
```bash
git rm --cached <artifact_file>
git commit -m "Stop tracking artifact in Git"
```
Then rerun:
```bash
dvc commit <stage_name>
dvc push
```

### `dvc.lock` missing
Cause:
- Stage has not been committed/reproduced successfully.

Fix:
```bash
dvc repro
```
(or `dvc commit <stage_name>` if outputs already exist and should be attached).

## License

MIT License.
