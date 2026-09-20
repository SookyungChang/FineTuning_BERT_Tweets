from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config():
     SEED: int = 42
     VERSION: str = "1.0.0"
     F1_AVG: str = "macro"

@dataclass
class pathConfig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    EXP_DIR: Path = PROJECT_ROOT / "saved_parameters"
    SAVE_MODEL_PATH: Path = PROJECT_ROOT / "saved_models"

@dataclass
class bertConfig():
        MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
        
        NUM_EPOCHS: int = 1
        DROUP_OUT: float = 0.1
        FINED_TUNED_MODEL_NAME: str = field(init=False)
        def __post_init__(self):
            self.FINED_TUNED_MODEL_NAME = f"sweetguma/bert-sentiment-model-v{self.version}"
    
@dataclass
class baseConfig():
    # hyper parameter searching OPNUNA?
    N_TRIALS: int = 150
    SAMPLE_SIZE: int = 100000

    # Searching space?
    MIN_DF: tuple = (1, 100)
    MAX_FEATURES: tuple = (5000, 30000)
    C: tuple = (1e-3, 50.0)

    # Model Setting
    MAX_ITER: int = 2000
