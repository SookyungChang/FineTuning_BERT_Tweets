# ~/src/ftt/training/train_baseline.py

import json
import pickle

from datasets import load_from_disk
from sklearn.metrics import f1_score

from ftt.models.baseline import build_model
from ftt.config import baseConfig, Config, pathConfig


paths = pathConfig()
config = Config()
base = baseConfig()


def train():

    # -------------------------
    # 1. Load dataset
    # -------------------------
    dataset = load_from_disk(
        paths.DATA_DIR / "tweets_dataset"
    )

    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]

    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    # -------------------------
    # 2. Load best parameters
    # -------------------------
    history_path = paths.EXP_DIR / "tfidf_history.json"

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    exp_key = f"trials{base.N_TRIALS}_sample{base.SAMPLE_SIZE}"

    best_params = history[exp_key]["best_params"]

    # tuple -> list
    best_params["ngram_range"] = tuple(
        best_params["ngram_range"]
    )

    print("Best parameters:")
    print(best_params)

    # -------------------------
    # 3. Build model
    # -------------------------
    model = build_model(**best_params)

    # -------------------------
    # 4. Train on full train set
    # -------------------------
    model.fit(X_train, y_train)

    # -------------------------
    # 5. Evaluate on test set
    # -------------------------
    preds = model.predict(X_test)

    f1 = f1_score(
        y_test,
        preds,
        average=config.F1_AVG,
    )

    print(f"Test F1 score: {f1:.4f}")

    # -------------------------
    # 6. Save model
    # -------------------------
    paths.SAVE_MODEL_PATH.mkdir(
        exist_ok=True,
        parents=True,
    )

    model_path = (
        paths.SAVE_MODEL_PATH
        / f"tfidf_logreg_{config.VERSION}.pkl"
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    train()