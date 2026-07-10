# Databricks notebook source
# silver/clean_hackernews.py
#
# Reads bronze.hn_raw, matches real tool mentions (via config/tools.py),
# explodes to one row per matched tool, cleans text, and writes to
# silver.hn_clean conforming to utils/schema.py SILVER_SCHEMA.

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
BRONZE_TABLE = f"{CATALOG}.bronze.hn_raw"
SILVER_TABLE = f"{CATALOG}.silver.hn_clean"

# COMMAND ----------

def clean_text(text: str) -> str:
    """Strip HTML entities/tags and normalize whitespace."""
    if not text:
        return None
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)      # strip HTML tags (HN comments use <p>, <a>, etc.)
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text if text else None

# COMMAND ----------

# Register match_tools as a Spark UDF (returns list of canonical tool names)
match_tools_udf = F.udf(lambda t: match_tools(t or ""), ArrayType(StringType()))
clean_text_udf = F.udf(clean_text, StringType())

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)

# COMMAND ----------

# 1. Clean text fields
# 2. Match tools against combined title + text (query_tool alone isn't trustworthy —
#    the raw query term may not actually appear in relevant context)
# 3. Explode so each matched tool gets its own row
cleaned_df = (
    bronze_df
    .withColumn("clean_title", clean_text_udf(F.col("title")))
    .withColumn("clean_text", clean_text_udf(F.col("text")))
    .withColumn(
        "matched_tools",
        match_tools_udf(F.concat_ws(" ", F.col("clean_title"), F.col("clean_text")))
    )
    .filter(F.size(F.col("matched_tools")) > 0)   # drop rows with no real tool match
    .withColumn("tool", F.explode(F.col("matched_tools")))
)

# COMMAND ----------

silver_df = (
    cleaned_df.select(
        F.col("object_id").alias("post_id"),
        F.lit("hackernews").alias("source"),
        F.col("tool"),
        F.col("object_type"),
        F.col("clean_title").alias("title"),
        F.col("clean_text").alias("text"),
        F.col("author"),
        F.col("points").cast("int").alias("score"),
        F.col("num_comments").cast("int"),
        F.col("url"),
        F.to_timestamp(F.col("hn_created_at")).alias("created_at"),
        F.col("ingested_at"),
        F.lit(datetime.now(timezone.utc)).alias("cleaned_at"),
    )
    .dropDuplicates(["post_id", "tool"])   # avoid dupes if job reruns on overlapping bronze data
)

# COMMAND ----------

validate_silver_df(silver_df)

# COMMAND ----------

(
    silver_df.write
    .format("delta")
    .mode("overwrite")            # full refresh of silver from bronze; switch to merge/append later if bronze grows large
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

print(f"Wrote {silver_df.count()} rows to {SILVER_TABLE}")