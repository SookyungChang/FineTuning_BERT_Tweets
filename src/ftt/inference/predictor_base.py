import pickle
from pathlib import Path
from ftt.config import pathConfig, baseConfig

paths = pathConfig()
bases = baseConfig()


def load_model():
    model_path = paths.SAVED_MODELS_PATH / f"tfidf_logreg_{bases.VERSION}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(text):
    model = load_model()
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    return {"text": text, "prediction": int(pred), "probs_0": probs[0], "probs_1": probs[1]}


if __name__ == "__main__":
    text = "I love this product!"
    print(predict(text))
