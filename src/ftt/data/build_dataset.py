# ~/src/ftt/data/build_dataset.py
import duckdb
from datasets import load_dataset, DatasetDict, load_from_disk
from ftt.config import pathConfig, Config
config = Config()
paths = pathConfig()

def convert_to_parquet(conn):
    parquet_path = paths.DATA_DIR / 'tweets_preprocessed.parquet'
    conn.execute(f"""
    COPY tweets_preprocessed
    TO '{str(parquet_path)}'
    (FORMAT PARQUET)
    """)



def split_dataset(parquet_file, test_size):
    dataset = load_dataset("parquet", data_files=parquet_file)

    split = dataset["train"].train_test_split(test_size=test_size, seed=config.SEED)

    train = split["train"]
    test = split["test"]

    val_test = test.train_test_split(test_size=0.5, seed=config.SEED)

    dataset = DatasetDict({
        "train": train,
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    return dataset

def labeling(dataset):
    dataset = dataset.rename_column("target", "label")

    dataset = dataset.map(lambda x: {"label": 1 if x["label"] == 4 else 0})

    print(dataset["train"].features)
    print(dataset["train"][0])
    return dataset
    


if __name__ == "__main__":
    db_file = paths.DATA_DIR / "tweets.duckdb"
    conn = duckdb.connect(str(db_file))

    convert_to_parquet(conn)

    parquet_file = [str(paths.DATA_DIR / 'tweets_preprocessed.parquet')]

    test_size = 0.2
    dataset = split_dataset(parquet_file, test_size)

    dataset = labeling(dataset)

    dataset.save_to_disk(paths.DATA_DIR / "tweets_dataset")