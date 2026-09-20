# ~/src/ftt/data/convert_to_db.py
import duckdb
import pandas as pd
from ftt.config import pathConfig

paths = pathConfig()

csv_file = paths.DATA_DIR / "tweets.csv"
db_file = paths.DATA_DIR / "tweets.duckdb"

columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

conn = duckdb.connect(str(db_file))

conn.execute("""
    CREATE TABLE IF NOT EXISTS kaggle_raw (
        target INTEGER,
        ids BIGINT,
        date VARCHAR,
        flag VARCHAR,
        user VARCHAR,
        text VARCHAR
    )
""")

for i, chunk in enumerate(
    pd.read_csv(
        csv_file,
        names=columns,
        header=None,
        encoding="latin-1",
        chunksize=100_000
    )
):
    conn.register("chunk", chunk)

    conn.execute("""
        INSERT INTO kaggle_raw
        SELECT *
        FROM chunk
    """)

    conn.unregister("chunk")

    print(f"[{i + 1}] chunk inserted")

conn.close()