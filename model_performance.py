from datasets import load_from_disk
from sklearn.metrics import f1_score

from ftt.inference.predictor_base import load_model
from ftt.inference.predictor_bert import Predictor
from ftt.config import baseConfig, pathConfig, bertConfig

paths = pathConfig()
bases = baseConfig()
berts = bertConfig()

dataset = load_from_disk(
        paths.DATA_DIR / "tweets_dataset"
    )

def compare_performance():
    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    base = load_model()
    base_preds = base.predict(X_test)

    base_f1 = f1_score(
        y_test,
        base_preds,
        average=bases.F1_AVG,
    )

    bert = Predictor()
    bert_preds = bert.predict(X_test)

    bert_f1 = f1_score(
            y_test,
            bert_preds,
            average=berts.F1_AVG,
        )


    print(f"Test F1 score: Baseline {base_f1:.4f} | BERT {bert_f1:.4f}")

if __name__ == "__main__":
    y_train = dataset['train']['label'] # 1214111
    print(f"Number of tweets for training: {len(y_train)}")
    # compare_performance()
    # Test F1 score: Baseline 0.8167 | BERT 0.8649