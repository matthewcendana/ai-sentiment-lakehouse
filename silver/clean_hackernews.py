# Databricks notebook source
# silver/clean_hackernews.py
#
# Reads NEW rows from bronze.hn_raw (since the last silver run), matches real
# tool mentions (via config/tools.py), explodes to one row per matched tool,
# cleans text, and appends to silver.hn_clean conforming to SILVER_SCHEMA.
#
# INCREMENTAL: uses bronze.ingested_at as a watermark — only bronze rows newer
# than the latest ingested_at already in silver get reprocessed. This means
# each run only cleans what bronze's most recent ingestion run added, not the
# entire table every time.

# COMMAND ----------

import sys
import re
import html
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.utils import AnalysisException

sys.path.append("../config")
sys.path.append("../utils")

# Ship these modules to executor workers too — UDFs run in separate worker
# processes that don't inherit the driver's sys.path.append() above, which
# causes "ModuleNotFoundError: No module named 'tools'" inside UDF execution
# if this isn't done.
spark.addArtifact("../config/tools.py", pyfile=True)
spark.addArtifact("../utils/schema.py", pyfile=True)
spark.addArtifact("../utils/audit_log.py", pyfile=True)

from tools import match_tools          # noqa: E402
from schema import SILVER_SCHEMA, validate_silver_df  # noqa: E402
from audit_log import log_run  # noqa: E402

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
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None

# COMMAND ----------

match_tools_udf = F.udf(lambda t: match_tools(t or ""), ArrayType(StringType()))
clean_text_udf = F.udf(clean_text, StringType())

# COMMAND ----------

def get_silver_watermark():
    """
    Returns the max ingested_at already reflected in silver, or None if the
    silver table doesn't exist yet — meaning "process everything" on first run.
    """
    try:
        result = spark.sql(f"SELECT MAX(ingested_at) AS wm FROM {SILVER_TABLE}").collect()[0]["wm"]
        return result
    except AnalysisException:
        return None

# COMMAND ----------

watermark = get_silver_watermark()

bronze_df = spark.table(BRONZE_TABLE)

if watermark is not None:
    print(f"Incremental run — only processing bronze rows ingested after {watermark}")
    bronze_df = bronze_df.filter(F.col("ingested_at") > F.lit(watermark))
else:
    print("No prior silver data found — processing all bronze rows (first run).")

bronze_count = bronze_df.count()
print(f"Bronze rows to process this run: {bronze_count}")

# COMMAND ----------

if bronze_count == 0:
    print("Nothing new to clean — silver is already up to date.")
    log_run(spark, source="hackernews", layer="silver", rows_processed=0)
else:
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
        .dropDuplicates(["post_id", "tool"])   # safety net against overlapping bronze pulls
    )

    validate_silver_df(silver_df)

    try:
        (
            silver_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(SILVER_TABLE)
        )
        row_count = silver_df.count()
        print(f"Appended {row_count} new rows to {SILVER_TABLE}")
        log_run(spark, source="hackernews", layer="silver", rows_processed=row_count)
    except Exception as e:
        log_run(spark, source="hackernews", layer="silver", rows_processed=0,
                 status="failed", error_message=str(e))
        raise