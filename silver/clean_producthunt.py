# Databricks notebook source
# silver/clean_producthunt.py
#
# Reads bronze.producthunt_raw (unfiltered), matches real tool mentions (via
# config/tools.py), explodes to one row per matched tool, cleans text, and
# writes to silver.producthunt_clean conforming to utils/schema.py SILVER_SCHEMA.
#
# NOTE: bronze.producthunt_raw contains ALL recent PH posts/comments, not just
# AI-tool-related ones (PH's API has no keyword search). This script is where
# the actual filtering happens — most rows here will be dropped.

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
BRONZE_TABLE = f"{CATALOG}.bronze.producthunt_raw"
SILVER_TABLE = f"{CATALOG}.silver.producthunt_clean"

# COMMAND ----------

def clean_text(text: str) -> str:
    """Strip HTML entities/tags and normalize whitespace."""
    if not text:
        return None
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None

# COMMAND ----------

match_tools_udf = F.udf(lambda t: match_tools(t or ""), ArrayType(StringType()))
clean_text_udf = F.udf(clean_text, StringType())

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)

# COMMAND ----------

# Clean text, match tools against combined title + text, drop rows with no
# match (the bulk of bronze — expected, since bronze is unfiltered), explode
# so each matched tool gets its own row.
cleaned_df = (
    bronze_df
    .withColumn("clean_title", clean_text_udf(F.col("title")))
    .withColumn("clean_text", clean_text_udf(F.col("text")))
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
        F.lit("producthunt").alias("source"),
        F.col("tool"),
        F.col("object_type"),
        F.col("clean_title").alias("title"),
        F.col("clean_text").alias("text"),
        F.col("author"),
        F.col("votes_count").cast("int").alias("score"),
        F.col("num_comments").cast("int"),
        F.col("url"),
        F.to_timestamp(F.col("ph_created_at")).alias("created_at"),
        F.col("ingested_at"),
        F.lit(datetime.now(timezone.utc)).alias("cleaned_at"),
    )
    .dropDuplicates(["post_id", "tool"])
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

# COMMAND ----------

# Quick check: how much of bronze was actually relevant?
bronze_count = bronze_df.count()
silver_count = silver_df.count()
print(f"Bronze rows scanned: {bronze_count}")
print(f"Silver rows kept (post-match, post-explode): {silver_count}")