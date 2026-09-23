# FineTuning BERT Tweets

This repository fine-tunes a BERT-based sentiment classifier for tweets and compares it against a TF-IDF + Logistic Regression baseline. The project is built around a full data-processing pipeline from raw Kaggle tweet CSV files to a Hugging Face model checkpoint, along with inference and evaluation scripts.

## Overview

The project performs binary sentiment classification on tweet text:

- Label 0: negative / non-positive sentiment
- Label 1: positive sentiment

The pipeline includes:

- ingesting the raw tweet CSV into a DuckDB database
- cleaning and deduplicating tweet text
- filtering to English-language tweets using Lingua
- creating a Hugging Face `DatasetDict` split for train/validation/test
- training a TF-IDF logistic regression baseline
- fine-tuning a pretrained BERT model
- evaluating both models on the test set
- running inference on individual texts or batches
- optionally uploading trained models to Hugging Face Hub

---

## Repository structure

```text
.
├── compose.yaml
├── Dockerfile.rocm
├── pyproject.toml
├── README.md
├── model_comparison.py
├── model_performance.py
├── data/
│   ├── tweets.csv
│   ├── tweets.duckdb
│   └── tweets_dataset/
├── saved_models/
├── saved_parameters/
├── src/
│   └── ftt/
│       ├── __init__.py
│       ├── config.py
│       ├── hf_model_upload.py
│       ├── data/
│       │   ├── convert_to_db.py
│       │   ├── preprocess.py
│       │   └── build_dataset.py
│       ├── inference/
│       │   ├── predictor_base.py
│       │   └── predictor_bert.py
│       ├── models/
│       │   ├── baseline.py
│       │   └── bert.py
│       └── training/
│           ├── train_baseline.py
│           └── train_bert.py
└── wandb/
```

---

## Tech stack

The project depends on:

- Python 3.12+
- PyTorch
- Hugging Face Transformers
- Datasets
- scikit-learn
- DuckDB
- Lingua language detector
- Optuna for baseline hyperparameter search
- Evaluate
- Weights & Biases (wandb)
- Hugging Face Hub

See the dependency list in [pyproject.toml](pyproject.toml) for the exact package requirements.

---

## Quick start

From the project root:

```bash
export PYTHONPATH=src
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tranining, use rocm environment through Docker:

```bash
docker compose build
docker compose run --rm ftt
```

The repo also expects a `.env` file for W&B credentials, for example:

```bash
WANDB_API_KEY=your_key_here
```

---

## Data pipeline

The raw dataset is expected at `data/tweets.csv` and is imported into DuckDB using [src/ftt/data/convert_to_db.py](src/ftt/data/convert_to_db.py).

### 1 Load CSV into DuckDB

```bash
export PYTHONPATH=src
python src/ftt/data/convert_to_db.py
```

This creates:

- `data/tweets.duckdb`
- a table named `kaggle_raw`

### 2 Preprocess and filter text

Run:

```bash
python src/ftt/data/preprocess.py
```

This file performs the following steps:

- deduplicates records by `ids`
- removes URLs and mentions from tweet text
- keeps only rows with non-empty text longer than 10 characters
- detects language using Lingua
- keeps only English tweets in `tweets_preprocessed`

### 3 Build Hugging Face dataset splits

Run:

```bash
python src/ftt/data/build_dataset.py
```

This does the following:

- exports DuckDB data to Parquet
- creates a `DatasetDict` split with train/validation/test
- renames column `target` to `label`
- remaps the original Kaggle labels so that `4 -> 1` and everything else -> `0`
- saves the final dataset to `data/tweets_dataset`

---

## Baseline model

The baseline is implemented in [src/ftt/models/baseline.py](src/ftt/models/baseline.py) and uses:

- TF-IDF vectorization
- Logistic Regression with `saga` solver
- Optuna hyperparameter optimization

### Hyperparameter search

Run:

```bash
export PYTHONPATH=src
python src/ftt/models/baseline.py
```

This will:

- train on a sample subset of the data
- search n-gram ranges, `min_df`, `max_features`, and `C`
- store the best results in `saved_parameters/tfidf_history.json`

### Train the final baseline model

Run:

```bash
python src/ftt/training/train_baseline.py
```

This loads the best Optuna parameters, trains on the full training split, evaluates on the test set, and saves the final model as:

```text
saved_models/tfidf_logreg_1.0.0.pkl
```

---

## BERT fine-tuning

The model code is in [src/ftt/models/bert.py](src/ftt/models/bert.py).

The training script in [src/ftt/training/train_bert.py](src/ftt/training/train_bert.py) loads the prepared dataset and fine-tunes a pretrained BERT model from Hugging Face:

- model name: `distilbert-base-uncased-finetuned-sst-2-english`
- learning rate: `5e-5`
- epochs: `3`
- batch size: `16`
- evaluation strategy: per epoch
- save strategy: per epoch
- W&B logging enabled

### Train the BERT model

```bash
export PYTHONPATH=src
python src/ftt/training/train_bert.py
```

This script:

- sets `CUDA_VISIBLE_DEVICES=0`
- checks for CUDA availability and falls back to CPU when needed
- initializes W&B with project `bert-finetuning`
- trains the model and saves checkpoints into `saved_models/bert-<version>/`

The training output is intended to be stored under:

```text
saved_models/bert-1.0.1/
```

---

## Inference

The inference code is under [src/ftt/inference](src/ftt/inference):

- [src/ftt/inference/predictor_base.py](src/ftt/inference/predictor_base.py): loads the TF-IDF + Logistic Regression model and returns class probabilities
- [src/ftt/inference/predictor_bert.py](src/ftt/inference/predictor_bert.py): loads a saved BERT checkpoint and predicts sentiment for single texts or batches

Example usage:

```python
from ftt.inference.predictor_bert import Predictor

bert = Predictor()
result = bert.predict_text("I love this product!")
print(result)
```

---

## Model comparison scripts

At the project root:

- [model_performance.py](model_performance.py): compares the baseline and BERT model F1 scores on the test split
- [model_comparison.py](model_comparison.py): prints side-by-side predictions for a small set of example sentences

Run them with:

```bash
python model_performance.py
python model_comparison.py
```

---

## Hugging Face upload

The script [src/ftt/hf_model_upload.py](src/ftt/hf_model_upload.py) uploads a trained model folder to Hugging Face Hub.

```bash
export PYTHONPATH=src
python src/ftt/hf_model_upload.py
```

It creates or reuses a repository named:

```text
sweetguma/bert-sentiment-model-v<version>
```

---

## Notes and conventions

- The raw Kaggle data uses a column set of: `target`, `ids`, `date`, `flag`, `user`, `text`.
- The repo converts the sentiment target so that the classification is binary instead of the original 0–4 sentiment scale.
- W&B logging is enabled in the BERT training flow, which is why a valid login token is expected.
- The project uses a GPU preference via `CUDA_VISIBLE_DEVICES=0` in the training scripts, but CPU fallbacks are included.
- The code assumes the repo root is the working directory when running scripts and that the `src` folder is on the Python path.

---

## Typical workflow

```bash
export PYTHONPATH=src

python src/ftt/data/convert_to_db.py
python src/ftt/data/preprocess.py
python src/ftt/data/build_dataset.py

python src/ftt/models/baseline.py
python src/ftt/training/train_baseline.py

python src/ftt/training/train_bert.py

python model_performance.py
```

This gives you a complete sentiment-analysis pipeline from raw tweet data to trained and evaluated models.
