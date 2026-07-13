# Databricks notebook source
# silver/clean_reddit.py
#
# Reads bronze.reddit_raw, matches real tool mentions (via config/tools.py),
# explodes to one row per matched tool, cleans text, and writes to
# silver.reddit_clean conforming to utils/schema.py SILVER_SCHEMA.

# COMMAND ----------

import sys
import re
import html
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

sys.path.append("../config")
sys.path.append("../utils")

from tools import match_tools          # noqa: E402
from schema import SILVER_SCHEMA, validate_silver_df  # noqa: E402

# COMMAND ----------

CATALOG = "ai_tool_sentiment"
BRONZE_TABLE = f"{CATALOG}.bronze.reddit_raw"
SILVER_TABLE = f"{CATALOG}.silver.reddit_clean"

# COMMAND ----------

def clean_text(text: str) -> str:
    """Strip HTML entities/markdown artifacts and normalize whitespace."""
    if not text:
        return None
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip stray HTML
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links -> just the text
    text = re.sub(r"[*_~`]", "", text)             # strip markdown emphasis chars
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None

# COMMAND ----------

match_tools_udf = F.udf(lambda t: match_tools(t or ""), ArrayType(StringType()))
clean_text_udf = F.udf(clean_text, StringType())

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)

# COMMAND ----------

# Clean title/text, fall back to title when selftext is null/empty (link posts),
# then match tools against combined text and explode one row per matched tool.
cleaned_df = (
    bronze_df
    .withColumn("clean_title", clean_text_udf(F.col("title")))
    .withColumn("clean_text", clean_text_udf(F.col("text")))
    .withColumn(
        "text_for_matching",
        F.coalesce(F.col("clean_text"), F.col("clean_title"))
    )
    .withColumn(
        "matched_tools",
        match_tools_udf(F.concat_ws(" ", F.col("clean_title"), F.col("clean_text")))
    )
    .filter(F.size(F.col("matched_tools")) > 0)
    .withColumn("tool", F.explode(F.col("matched_tools")))
)

# COMMAND ----------

silver_df = (
    cleaned_df.select(
        F.col("object_id").alias("post_id"),
        F.lit("reddit").alias("source"),
        F.col("tool"),
        F.col("object_type"),
        F.col("clean_title").alias("title"),
        F.col("text_for_matching").alias("text"),   # falls back to title if selftext empty
        F.col("author"),
        F.col("score").cast("int"),
        F.col("num_comments").cast("int"),
        F.col("permalink").alias("url"),             # use permalink over external url for consistency
        F.col("reddit_created_utc").alias("created_at"),
        F.col("ingested_at"),
        F.lit(datetime.now(timezone.utc)).alias("cleaned_at"),
    )
    .dropDuplicates(["post_id", "tool"])   # same post + same tool won't double up across reruns
)

# COMMAND ----------

validate_silver_df(silver_df)

# COMMAND ----------

(
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

print(f"Wrote {silver_df.count()} rows to {SILVER_TABLE}")