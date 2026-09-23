
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ftt.config import pathConfig, bertConfig
paths = pathConfig()
berts = bertConfig()

class Predictor:
    def __init__(self, model_path=berts.FINED_TUNED_MODEL_NAME):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.model.eval()

    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs).item()

        return {
            "text": text,
            "prediction": pred,
            "confidence": float(torch.max(probs)),
        }

    def predict(self, texts, batch_size=berts.BATCH_SIZE):
        predictions = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Predicting",
            unit="batch",
        ):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=berts.MAX_LENGTH,
                return_tensors="pt",
                return_token_type_ids=False,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = outputs.logits.argmax(dim=-1)

            predictions.extend(preds.cpu().tolist())

        return predictions

if __name__ == "__main__":
    text = "I love this product!"
    model_path = paths.SAVED_MODELS_PATH / f"bert-{berts.VERSION}" / "checkpoint-75882"
    bert = Predictor(model_path)
    print(bert.predict_text(text))