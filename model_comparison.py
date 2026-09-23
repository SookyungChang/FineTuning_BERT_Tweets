import os
from ftt.inference import predictor_base, predictor_bert
from huggingface_hub import snapshot_download

from ftt.config import pathConfig, bertConfig
paths = pathConfig()
berts = bertConfig()


def compare_models():
    bert = predictor_bert.Predictor()

    texts = [
        "I love this product!",
        "This is the worst experience ever.",
        "It's okay, not great but not bad.",
        "I feel so happy today!",
        "I'm really disappointed and sad.",
    ]

    print("=" * 60)
    print(" BASELINE vs BERT COMPARISON")
    print("=" * 60)

    for text in texts:
        bert_result = bert.predict_text(text)
        base_result = predictor_base.predict(text)

        print(f"\n📝 Text: {text}")
        print("-" * 60)

        base_prediction = base_result['prediction']
        if base_result['probs_0'] > base_result['probs_1']:
            base_confidence = base_result['probs_0']
        else: 
            base_confidence = base_result['probs_1']

        print(
            f"Baseline → pred: {base_result['prediction']} | conf: {base_confidence:.4f}"
        )
        print(
            f"BERT     → pred: {bert_result['prediction']} | conf: {bert_result['confidence']:.4f}"
        )

        # disagreement
        if base_result["prediction"] != bert_result["prediction"]:
            print("⚠️  DISAGREEMENT!")

    print("\nDone.")


if __name__ == "__main__":
    compare_models()
