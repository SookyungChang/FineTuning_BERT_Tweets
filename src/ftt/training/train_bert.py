import os

# Only make GPU 0 visible to this process. This prevents the internal/bad GPU 1
# from being selected while still allowing a CPU fallback if CUDA is unavailable.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dotenv import load_dotenv
load_dotenv()

import torch
import wandb
from transformers import TrainingArguments
from ftt.models.bert import BERTfinetuning
from datasets import load_from_disk

from ftt.config import bertConfig, Config, pathConfig

paths = pathConfig()
config = Config()
bert_config = bertConfig()

def train(device=None, sample_size=None, log_name=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Requested device: {device}")
    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {torch.cuda.is_available()}")
    num_workers = max(1, os.cpu_count() // 4)
    
    if log_name is None:
        log_name = "bert" + "-" + bert_config.VERSION

    dataset = load_from_disk(
               paths.DATA_DIR / "tweets_dataset"
           )

    paths.SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    output_dir_path = os.path.join(paths.SAVED_MODELS_PATH, f"bert-{bert_config.VERSION}")
    wandb.init(project="bert-finetuning", name=log_name, config=bert_config.__dict__)
    # Training Pipeline: Define training arguments

    logging_steps = (
        50 if sample_size is None else max(1, sample_size // 1600)
    )  # Adjust logging steps based on sample size
    training_args = TrainingArguments(
        output_dir=output_dir_path,  # Directory for saving model checkpoints
        report_to="wandb",
        logging_steps=logging_steps,
        run_name=log_name,  # Name for WandB logging
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        # eval_steps=eval_steps,
        save_strategy="epoch",
        # save_steps=eval_steps,
        learning_rate=5e-5,  # *Start with a small learning rate
        per_device_train_batch_size=16,  # Batch size per device
        per_device_eval_batch_size=16,
        dataloader_num_workers=num_workers,  # Use CPU workers for data loading
        dataloader_pin_memory=use_gpu,  # Pin memory for faster GPU transfer
        num_train_epochs=bert_config.NUM_EPOCHS,  # Number of epochs
        weight_decay=0.01,  # Regularization
        save_total_limit=2,  # Limit checkpoints to save space
        load_best_model_at_end=True,  # Automatically load the best checkpoint
        fp16=use_gpu,  # Mixed precision only on GPU
        use_cpu=False,  # Explicitly allow CPU training if no GPU
        bf16=False,
    )
    bert = BERTfinetuning(dataset, training_args)
    print("Before train:", bert.test())
    bert.train()
    print("After train:", bert.test())
    wandb.finish()
    return bert


if __name__ == "__main__":
    train()
