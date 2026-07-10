# utils/schema.py
#
# Shared silver-layer schema. Every source (hackernews, reddit, producthunt)
# must conform to this schema when writing to silver.* tables so gold-layer
# scripts can UNION them without surprises.

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
)

SILVER_SCHEMA = StructType(
    [
        StructField("post_id", StringType(), nullable=False),     # unique id, source-specific (e.g. HN objectID, Reddit id)
        StructField("source", StringType(), nullable=False),      # "hackernews" | "reddit" | "producthunt"
        StructField("tool", StringType(), nullable=False),        # canonical tool name from config/tools.py (one row per matched tool)
        StructField("object_type", StringType(), nullable=True),  # "story" | "comment" | "post" | "review" etc.
        StructField("title", StringType(), nullable=True),
        StructField("text", StringType(), nullable=True),         # cleaned body text used for sentiment scoring
        StructField("author", StringType(), nullable=True),
        StructField("score", IntegerType(), nullable=True),       # points / upvotes, source-dependent
        StructField("num_comments", IntegerType(), nullable=True),
        StructField("url", StringType(), nullable=True),
        StructField("created_at", TimestampType(), nullable=True),   # original post/comment creation time
        StructField("ingested_at", TimestampType(), nullable=True),  # when bronze ingestion pulled it
        StructField("cleaned_at", TimestampType(), nullable=True),   # when silver cleaning ran
    ]
)

# Column order reference, useful when building rows manually
SILVER_COLUMNS = [f.name for f in SILVER_SCHEMA.fields]


def validate_silver_df(df):
    """
    Lightweight check: confirms a DataFrame has all required silver columns
    before writing. Raises ValueError if any are missing. Does not enforce
    strict typing (Spark will coerce/error on write mismatches anyway).
    """
    df_columns = set(df.columns)
    missing = [c for c in SILVER_COLUMNS if c not in df_columns]
    if missing:
        raise ValueError(f"DataFrame missing required silver columns: {missing}")
    return True