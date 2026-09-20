# ~/src/ftt/models/baseline.py
import json
import os
import optuna  # https://optuna.readthedocs.io/en/stable/tutorial/index.html
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from datasets import load_from_disk

from ftt.config import baseConfig, Config, pathConfig
paths = pathConfig()
config = Config()
base = baseConfig()

dataset = load_from_disk(paths.DATA_DIR / "tweets_dataset")

X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_dev = dataset["validation"]["text"]
y_dev = dataset["validation"]["label"]

X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

def build_model(
    ngram_range,
    min_df,
    max_features,
    C,
):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=True,
            ),
        ),
        (
            "logreg",
            LogisticRegression(
                C=C,
                solver="saga",
                max_iter=base.MAX_ITER,
                random_state=config.SEED,
                verbose=True,
            ),
        ),
    ],
    verbose=True)


def run_optuna():

    X_train_sub, y_train_sub = (
        X_train[: base.SAMPLE_SIZE],
        y_train[: base.SAMPLE_SIZE],
    )

    X_dev_sub, y_dev_sub = X_dev[: base.SAMPLE_SIZE], y_dev[: base.SAMPLE_SIZE]

    ngram_dict = {"1-1": (1, 1), "1-2": (1, 2), "1-3": (1, 3)}

    def objective(trial):

        ngram_choice = trial.suggest_categorical(
            "ngram_range", 
            ["1-1", "1-2", "1-3"],
        )

        current_ngram = ngram_dict[ngram_choice]

        min_df = trial.suggest_int("min_df", *base.MIN_DF)

        max_features = trial.suggest_int(
            "max_features",
            *base.MAX_FEATURES
        )

        C = trial.suggest_float(
            "C",
            *base.C,
            log=True
        )

        pipe = build_model(
            ngram_range=current_ngram,
            min_df=min_df,
            max_features=max_features,
            C=C,
        )

        pipe.fit(X_train_sub, y_train_sub)
        predictions = pipe.predict(X_dev_sub)

        return f1_score(
            y_dev_sub,
            predictions,
            average=config.F1_AVG
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=base.N_TRIALS)

    best_params = study.best_params

    best_params["ngram_range"] = ngram_dict[study.best_params["ngram_range"]]

    # Generate a unique key based on experiment conditions

    new_data = {
        "best_params": best_params,
        "f1_score": study.best_value,
        "n_trials": base.N_TRIALS,
        "sample_size": base.SAMPLE_SIZE,
    }

    exp_key = f"trials{base.N_TRIALS}_sample{base.SAMPLE_SIZE}"

    # Load existing history if the file exists
    # Update the record (Overwrites if the key exists, adds if it doesn't)
    file_path = paths.EXP_DIR / "tfidf_history.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)

            except json.JSONDecodeError:
                history = {}

    else:
        history = {}

    history[exp_key] = new_data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"Best F1 Score: {study.best_value}")
    print(f"Best Params: {best_params}")

if __name__ == "__main__":
    run_optuna()