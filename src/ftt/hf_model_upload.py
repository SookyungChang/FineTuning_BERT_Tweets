from huggingface_hub import upload_folder, create_repo
from dotenv import load_dotenv
from ftt.config import pathConfig, bertConfig
paths = pathConfig()
berts = bertConfig()

load_dotenv()

def upload(folder_name):
    folder_path = paths.SAVED_MODELS_PATH / folder_name
    repo_id=f"sweetguma/bert-sentiment-model-v{berts.VERSION}"
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    folder_name = f"bert-{berts.VERSION}/checkpoint-75882"
    upload(folder_name)