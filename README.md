# MLOps Pipeline — Experiment Tracking with MLflow + DagsHub

<a target="_blank" href="https://cookiecutter-data-science.drivendaily.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project trains a Titanic survival classifier and tracks every experiment with **MLflow**, hosted on **DagsHub**.
Each training run automatically logs hyperparameters, cross-validation scores, validation metrics, and model artifacts — making runs comparable, searchable, and fully reproducible from the dashboard alone.

Data and model binaries are still versioned with **DVC** (remote on DagsHub), but the headline workflow is now **experiment tracking**.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      Your Machine                           │
│                                                             │
│  Hydra Config ──► trainer.py ──► pipeline.py                │
│       │                              │                      │
│       │       ┌──────────────────────┤                      │
│       │       │                      │                      │
│       ▼       ▼                      ▼                      │
│   mlflow_utils.py             DVC (dvc repro)               │
│       │                              │                      │
└───────┼──────────────────────────────┼──────────────────────┘
        │                              │
        ▼                              ▼
  ┌───────────────────────────────────────────┐
  │              DagsHub (Remote)             │
  │                                           │
  │  MLflow Tracking Server   DVC S3 Remote   │
  │  (params, metrics, tags)  (data, models)  │
  └───────────────────────────────────────────┘
```

**One `dvc repro` triggers everything**: the training pipeline runs, MLflow logs the experiment to DagsHub, and DVC snapshots the binary artifacts.

## Opinionated Modifications Explained

Every integration choice below is deliberate. This section explains **why**, not just **what**.

### 1. `dagshub.init()` over manual MLflow server setup

```python
# src/mlflow_utils.py
dagshub.init(repo_owner="nouran-19", repo_name="MLOps-practice", mlflow=True)
```

**Why**: DagsHub gives you a free hosted MLflow tracking server — zero infrastructure, no Docker, no database.
Calling `dagshub.init()` auto-configures `MLFLOW_TRACKING_URI` and injects auth from `DAGSHUB_USER_TOKEN`.
There is nothing else to set up.

**Trade-off**: You're coupled to DagsHub as the backend. If you need a self-hosted MLflow server later, swap the `init_dagshub_mlflow()` function in `mlflow_utils.py` — no other code changes required.

### 2. One MLflow run per pipeline execution

```python
# src/pipeline.py
with mlflow.start_run(run_name=cfg.experiment_name):
    # ... entire train → evaluate → save cycle
```

**Why**: The full train-evaluate-save cycle is one atomic unit of work. Logging it as one run means the DagsHub UI shows a single row per experiment config.
Switch from `titanic_baseline` to `titanic_alt` via Hydra, and each becomes a separate, directly comparable run.

**Trade-off**: If you need per-model-candidate child runs (e.g., logging Random Forest and Logistic Regression as sub-runs), MLflow supports nested runs. The current design keeps things flat and simple.

### 3. Centralised `src/mlflow_utils.py`

**Why**: All `mlflow.log_*()` calls live in one module. The training pipeline (`pipeline.py`) calls high-level helpers like `log_hydra_params(cfg)` instead of scattering tracking code everywhere.

Benefits:
- **Swap trackers**: Replace this one file to switch from MLflow to Weights & Biases, Neptune, etc.
- **Test in isolation**: Mock `mlflow_utils` in tests without touching training logic.
- **Read the pipeline**: `pipeline.py` stays focused on ML, not on serialising metrics to an API.

### 4. Hydra params are flattened and logged

```python
# Flattens nested Hydra config to:
# training.test_size = 0.2
# models.random_forest.n_estimators = 300
# ...
log_hydra_params(cfg)
```

**Why**: Every single hyperparameter is captured in MLflow. You can reproduce any historical run from the dashboard alone — no need to dig through Git history for the YAML file that was active at the time.

### 5. Model artifact logged to both DVC and MLflow

**Why**: Two different purposes:
- **DVC**: Heavyweight binary versioning with content-addressable storage. The `.pkl` file is pushed to the DagsHub S3 remote.
- **MLflow**: Lightweight run-attached snapshot for traceability. Click a run in the UI → download the exact model that produced those metrics.

Neither replaces the other. DVC is the source of truth for artifact versions; MLflow links runs to artifacts.

## Project Structure

```
MLOps-practice2/
├── .dvc/                             # DVC config and internal metadata
├── conf/                             # Hydra configuration files
│   ├── config.yaml                   # Default config (selects pipeline)
│   └── pipeline/
│       ├── titanic_baseline.yaml     # Baseline experiment config
│       └── titanic_alt.yaml          # Alternative experiment config
├── data/raw/titanic/                 # Titanic CSV files (tracked by DVC)
├── models/titanic/                   # Trained pipeline artifact (DVC output)
├── reports/titanic/                  # Training report artifact (DVC output)
├── src/
│   ├── mlflow_utils.py               # MLflow + DagsHub tracking helpers
│   ├── pipeline.py                   # Training pipeline (instrumented)
│   ├── logger.py                     # Loguru-based logger
│   └── training/                     # Evaluation & scoring modules
├── trainer.py                        # Entry point (Hydra + dotenv)
├── dockerfile.mlflow                 # Docker image for MLflow inference server
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # Locked stage dependency/output hashes
├── Makefile                          # Convenience targets (train, mlflow-ui)
└── pyproject.toml                    # Dependencies and project config
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Kaggle credentials for Titanic data access
- DagsHub account and [user token](https://dagshub.com/user/settings/tokens)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/nouran-19/MLOps-practice.git
cd MLOps-practice2
uv sync
```

### 2. Configure secrets

Copy the example env file and fill in your tokens:

```bash
cp .env.example .env
```

```dotenv
KAGGLE_USERNAME="your_kaggle_username"
KAGGLE_API_TOKEN="your_kaggle_key"
DAGSHUB_USER_TOKEN="your_dagshub_token"
```

### 3. Pull existing DVC artifacts (optional)

```bash
dvc pull
```

### 4. Run the training pipeline

```bash
# Option A: via DVC (recommended — also versions outputs)
dvc repro

# Option B: via Makefile shortcut
make train

# Option C: run directly (skips DVC output tracking)
.\.venv\Scripts\python.exe trainer.py
```

Each run automatically logs to the DagsHub MLflow server.

### 5. View experiments

```bash
make mlflow-ui
# → Opens: https://dagshub.com/nouran-19/MLOps-practice.mlflow
```

Or navigate directly to the URL above.

## Comparing Experiments

### Switch Hydra configs to create different runs

```bash
# Run the baseline config
dvc repro    # uses conf/pipeline/titanic_baseline.yaml by default

# Run the alternative config
.\.venv\Scripts\python.exe trainer.py pipeline=titanic_alt
```

Each config produces a separate MLflow run. On the DagsHub experiments page you can:

1. **Select runs** → side-by-side parameter + metric comparison
2. **Sort by metric** → find the best `val.roc_auc` or `val.f1`
3. **Download artifacts** → grab the `.pkl` model directly from the run

### What gets logged per run

| Category | Examples | Where |
|---|---|---|
| **Parameters** | `training.test_size`, `models.random_forest.n_estimators`, ... | MLflow Params tab |
| **CV Metrics** | `random_forest.cv.accuracy`, `logistic_regression.cv.f1`, ... | MLflow Metrics tab |
| **Validation Metrics** | `val.accuracy`, `val.precision`, `val.recall`, `val.f1`, `val.roc_auc` | MLflow Metrics tab |
| **Tags** | `best_model` | MLflow Tags |
| **Artifacts** | `titanic_pipeline.pkl`, `training_report.json` | MLflow Artifacts tab |

## DVC Pipeline (Reference)

The DVC pipeline still handles reproducibility and artifact versioning. Two stages are defined in `dvc.yaml`:

1. **`train_titanic`**: Runs `trainer.py` → outputs model + training report. Also logs to MLflow.
2. **`score_titanic`**: Scores the trained model on training data → outputs downstream predictions + report.

```bash
# Reproduce the pipeline (reruns only changed stages)
dvc repro

# Check what's changed
dvc status

# Push artifacts to DagsHub remote
dvc push
```

Pipeline DAG:

```text
raw Titanic CSVs
    │
    ▼
train_titanic ──► MLflow (params, metrics, artifacts)
    │
    ▼
models/titanic/titanic_pipeline.pkl
    │
    ▼
score_titanic
    │
    ▼
downstream report + predictions
```

## Daily Workflow

```bash
# 1. Make changes to config/code
# 2. Run pipeline (logs to MLflow automatically)
dvc repro

# 3. Compare runs on DagsHub
make mlflow-ui

# 4. Push artifacts + commit metadata
dvc push
git add dvc.yaml dvc.lock .dvc/config *.dvc .gitignore
git commit -m "Experiment: <what you changed>"
git push
```

## Common Issues

### `dagshub.init()` fails with auth error
**Cause**: Missing or invalid `DAGSHUB_USER_TOKEN` in `.env`.

**Fix**: Generate a new token at [DagsHub settings](https://dagshub.com/user/settings/tokens) and update `.env`.

### MLflow run not appearing on DagsHub
**Cause**: Running `trainer.py` directly without loading `.env` first.

**Fix**: The `trainer.py` entry point loads `.env` automatically via `python-dotenv`. If you're calling `pipeline.py` from a different entry point, ensure `DAGSHUB_USER_TOKEN` is set in your environment.

### `401 Unauthorized` on `dvc push`
**Cause**: Missing or invalid DagsHub token in local DVC config.

**Fix**: Update local credentials for the remote and retry:
```bash
dvc remote modify --local origin access_key_id <dagshub_token>
dvc remote modify --local origin secret_access_key <dagshub_token>
dvc push
```

### Outputs tracked by Git instead of DVC
**Cause**: Artifact file was committed before being moved to DVC tracking.

**Fix**:
```bash
git rm --cached <artifact_file>
git commit -m "Stop tracking artifact in Git"
dvc commit <stage_name>
dvc push
```

## Minimal Note on Hydra

Hydra manages training configuration in `trainer.py` and `conf/`.
Switch experiments by overriding the pipeline config:

```bash
# Default (titanic_baseline)
.\.venv\Scripts\python.exe trainer.py

# Alternative config
.\.venv\Scripts\python.exe trainer.py pipeline=titanic_alt
```

All Hydra parameters are automatically flattened and logged to MLflow — so the exact config is always traceable from any historical run.

## Model Registry and Deployment

### Creating the Inference Server Docker Image

Using the **MLflow Model Registry**, an inference server can be deployed by building a Docker image that pulls the registered model from DagsHub and serves it via `mlflow models serve`.

The Dockerfile is already provided at [dockerfile.mlflow](dockerfile.mlflow). Update the registry URI, model name, and model version placeholders to match your DagsHub registry entry. It:

1. Uses the `uv`-enabled Python 3.12 slim image
2. Installs project dependencies from the lockfile
3. Serves the model on container start using:
   ```
   mlflow models serve -m models:/<MODEL_NAME>/<MODEL_VERSION> -p 5000 --no-conda
   ```

#### Build the Image

```bash
docker build -t mlflow-dagshub-server -f dockerfile.mlflow .
```

#### Run Locally

Test the server locally before pushing. Pass your DagsHub credentials as environment variables:

```bash
docker run -p 5000:5000 \
  -e MLFLOW_TRACKING_USERNAME="<your_dagshub_username>" \
  -e MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>" \
  mlflow-dagshub-server
```

The inference endpoint will be available at `http://localhost:5000/invocations`.

#### Push to DockerHub

Once the server works locally, tag and push the image:

```bash
docker login
# enter your DockerHub username and password

# tag the image
docker tag mlflow-dagshub-server:latest <your_dockerhub_username>/mlflow-dagshub-server:latest

# push it
docker push <your_dockerhub_username>/mlflow-dagshub-server:latest
```

### Deploy to Lightning AI ⚡

With the image on DockerHub, deploy the inference server to [Lightning AI](https://lightning.ai/):

1. **Create a new Lightning Studio** or open an existing one.
2. **Pull the Docker image** from DockerHub inside the studio terminal:
   ```bash
   docker pull <your_dockerhub_username>/mlflow-dagshub-server:latest
   ```
3. **Run the container** with your DagsHub credentials:
   ```bash
   docker run -d -p 5000:5000 \
     -e MLFLOW_TRACKING_USERNAME="<your_dagshub_username>" \
     -e MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>" \
     <your_dockerhub_username>/mlflow-dagshub-server:latest
   ```
4. **Expose the port** — Lightning AI auto-generates a public URL for the forwarded port.
5. **Verify** — navigate to the public URL and confirm the server is responding.

---

## Lab 5 — Invoking the Deployed Endpoint

Once the inference server is running locally or on Lightning AI, you can send prediction requests to the `/invocations` endpoint.

### Quick validation script

A ready-made test script is included at [`test_endpoint.py`](test_endpoint.py):

```bash
# Against local Docker container
python test_endpoint.py

# Against Lightning AI (or any remote host)
python test_endpoint.py http://<lightning-ai-public-url>
```

It sends two sample passengers and prints the predictions.

### Using the service UI

If you put the model behind a FastAPI wrapper or another gateway that exposes docs, open the server URL in your browser and navigate to:

```
http://<host>:<port>/docs
```

From the interactive Swagger UI you can:

1. Expand the **POST `/invocations`** endpoint.
2. Click **Try it out**.
3. Paste a JSON payload and hit **Execute**.

If the deployment only exposes the MLflow scoring server, skip the docs page and call `/invocations` directly.

### Using `curl`

Send a prediction request from the command line:

```bash
curl -X POST http://<host>:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
      "data": [[3, "Braund, Mr. Owen Harris", "male", 22.0, 1, 0, "A/5 21171", 7.25, null, "S"]]
    }
  }'
```

> **Note**: The model's sklearn Pipeline includes `TitanicFeatureEngineer`, which derives features from `Name`, `Ticket`, and `Cabin`. All raw columns must be present in the request — even if some values are `null`.

### Using Python `requests`

```python
import requests

url = "http://<host>:5000/invocations"
payload = {
    "dataframe_split": {
        "columns": ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        "data": [[3, "Braund, Mr. Owen Harris", "male", 22.0, 1, 0, "A/5 21171", 7.25, None, "S"]],
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

> **Tip**: Replace `<host>` with `localhost` for local testing, or with the Lightning AI public URL for the deployed server.

## License

MIT License.
