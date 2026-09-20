# ~/src/ftt/data/preprocess.py
import duckdb
from lingua import LanguageDetectorBuilder
from ftt.config import pathConfig
paths = pathConfig()

def deduplicate_ids(conn):
    conn.execute("""
    CREATE TABLE tweets_unique AS
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY ids
                   ORDER BY ids
               ) AS rn
        FROM kaggle_raw
    )
        WHERE rn = 1
    """)

def text_clean(conn):
    result = conn.execute("""
        SELECT
            text AS original,
            trim(
                regexp_replace(
                    regexp_replace(
                        text,
                        'https?://\\S+|www\\.\\S+',
                        '[URL]',
                        'g'
                    ),
                    '@[A-Za-z0-9_]+',
                    '[USER]',
                    'g'
                )
            ) AS cleaned
        FROM tweets_unique
        LIMIT 20
    """).fetchall()

    for original, cleaned in result:
        print("ORIGINAL:", original)
        print("CLEANED :", cleaned)
        print()

def clean_text(conn):
    conn.execute("""
        CREATE TABLE tweets_clean AS
        SELECT
            target,
            ids,
            date,
            flag,
            user,
            trim(
                regexp_replace(
                    regexp_replace(
                    text, 
                    'https?://\\S+|www\\.\\S+', '[URL]', 'g'),
                    '@[A-Za-z0-9_]+', '[USER]', 'g'
                )
            ) AS text
        FROM tweets_unique
        WHERE text IS NOT NULL AND length(text) > 10 
    """)

detector = LanguageDetectorBuilder.from_all_languages().build() # returns a list of language guesses with probabilities

def language_detect(text: str):
    """Try to detect the language of a comment. Returns result or None if uncertain."""

    confidence_values = detector.compute_language_confidence_values(text)
    top_result = confidence_values[0] # We take the top guess [0]
    lang_code = top_result.language.iso_code_639_1.name  # 'EN'
    confidence = top_result.value  # 0.9999... (0.0 - 1.0)
    
    return lang_code, confidence

def filter_englisch(conn, chunk_size=100_000):

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tweets_preprocessed (
            target INTEGER,
            ids BIGINT,
            date VARCHAR,
            flag VARCHAR,
            user VARCHAR,
            text VARCHAR,
            language VARCHAR,
            confidence DOUBLE
        )
    """)

    offset = 0

    while True:

        # 1. DuckDB → Python
        df = conn.execute(f"""
            SELECT *
            FROM tweets_clean
            LIMIT {chunk_size}
            OFFSET {offset}
        """).df()

        if df.empty:
            break

        print(f"Processing rows {offset:,} - {offset + len(df):,}")

        # 2. Python → Lingua
        results = df["text"].apply(language_detect)

        df["language"] = results.str[0]
        df["confidence"] = results.str[1]

        # 3. filter EN
        df = df[df["language"] == "EN"]

        # 4. Python → DuckDB
        conn.register("language_chunk", df)

        conn.execute("""
            INSERT INTO tweets_preprocessed
            SELECT *
            FROM language_chunk
        """)

        conn.unregister("language_chunk")

        print(f"  → {len(df):,} English rows saved")

        offset += chunk_size


if __name__ == "__main__":

    db_file = paths.DATA_DIR / "tweets.duckdb"
    conn = duckdb.connect(str(db_file))

    # deduplicate_ids(conn)

    # clean_text(conn)

    # filter_englisch(conn)

    conn.close()

