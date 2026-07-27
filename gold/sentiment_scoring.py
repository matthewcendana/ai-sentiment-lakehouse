# Databricks notebook source
# gold/sentiment_scoring.py
#
# Scores sentiment toward each AI tool using ai_query() against a Databricks-hosted
# foundation model. Loops through ALL pending rows in small batches until either
# everything is scored or a time safety limit is hit — safe to run unattended as
# a scheduled Job. Also safe to stop and re-run manually anytime (fully resumable,
# never re-scores rows that already succeeded).
#
# Requires a SERVERLESS SQL warehouse or serverless compute — ai_query() does not
# work on Classic/Pro warehouses.

# COMMAND ----------

import sys
import time

sys.path.append("../utils")
from audit_log import log_run  # noqa: E402

CATALOG = "ai_tool_sentiment"
GOLD_TABLE = f"{CATALOG}.gold.sentiment_scored"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

BATCH_SIZE = 500            # rows scored per inner batch
MAX_TEXT_CHARS = 800        # truncate long posts to control cost per call
MAX_RUNTIME_MINUTES = 45    # safety cap so the job doesn't run forever / blow past quota unattended

# COMMAND ----------

# Ensure the gold table exists with the right schema before we start appending.
# Runs once; harmless if it already exists.
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_TABLE} (
    post_id       STRING,
    source        STRING,
    tool          STRING,
    title         STRING,
    text          STRING,
    author        STRING,
    score         INT,
    num_comments  INT,
    url           STRING,
    created_at    TIMESTAMP,
    sentiment_score DOUBLE
)
""")

# COMMAND ----------

# Combine all silver sources into one working view.
# Add producthunt_clean here once you're ready to include that source.
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW silver_union AS
SELECT * FROM {CATALOG}.silver.hn_clean
UNION ALL
SELECT * FROM {CATALOG}.silver.reddit_clean
""")

# COMMAND ----------

def refresh_pending_view():
    """
    Rebuild the pending_rows view: rows in silver_union NOT already in gold
    (keyed on post_id + tool). Re-run after each batch write so the anti-join
    reflects what's just been scored — this is what makes looping safe.
    """
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW pending_rows AS
        SELECT su.*
        FROM silver_union su
        LEFT ANTI JOIN (SELECT post_id, tool FROM {GOLD_TABLE}) g
            ON su.post_id = g.post_id AND su.tool = g.tool
        WHERE su.text IS NOT NULL AND LENGTH(su.text) > 10
    """)
    return spark.sql("SELECT COUNT(*) AS n FROM pending_rows").collect()[0]["n"]

# COMMAND ----------

start_time = time.time()
total_scored = 0
batch_num = 0

pending_count = refresh_pending_view()
print(f"Rows remaining to score at start: {pending_count}")

try:
    while pending_count > 0:
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes >= MAX_RUNTIME_MINUTES:
            print(f"Hit time safety cap ({MAX_RUNTIME_MINUTES} min). Stopping — re-run the job to continue.")
            break

        batch_num += 1
        batch_df = spark.sql(f"SELECT * FROM pending_rows LIMIT {BATCH_SIZE}")
        batch_df.createOrReplaceTempView("current_batch")

        scored_df = spark.sql(f"""
            SELECT
                post_id,
                source,
                tool,
                title,
                text,
                author,
                score,
                num_comments,
                url,
                created_at,
                TRY_CAST(
                    ai_query(
                        '{MODEL_ENDPOINT}',
                        CONCAT(
                            'You are scoring sentiment toward ONE SPECIFIC tool: "', tool, '". ',
                            'The text below may mention several different tools — ignore sentiment toward ',
                            'any other tool and score ONLY what is said about "', tool, '" specifically. ',
                            'If the text lists/compares multiple tools, find the part that discusses "', tool, '" ',
                            'and base your score on that part alone, even if the overall tone of the post ',
                            'differs. Watch for sarcasm, backhanded compliments, or jokes that sound positive ',
                            'but are actually critical of the tool. ',
                            'Use the full scale from -1 (very negative) to 1 (very positive), 0 being neutral — ',
                            'use precise decimal values (e.g. 0.3, -0.6, 0.85) rather than only -1, 0, or 1; ',
                            'reserve exactly -1 or 1 for genuinely extreme, unambiguous statements. ',
                            'The text is provided between <text> tags. Respond with ONLY the numeric score, nothing else, no explanation.

<text>', LEFT(text, {MAX_TEXT_CHARS}), '</text>'
                        )
                    ) AS DOUBLE
                ) AS sentiment_score
            FROM current_batch
        """)

        (
            scored_df.write
            .format("delta")
            .mode("append")
            .saveAsTable(GOLD_TABLE)
        )

        batch_scored = scored_df.count()
        total_scored += batch_scored
        print(f"Batch {batch_num}: scored {batch_scored} rows (running total this run: {total_scored})")

        # Refresh pending count so the loop knows whether to continue
        pending_count = refresh_pending_view()

    print(f"Done for this run. Total scored: {total_scored}. Rows still pending: {pending_count}")
    if pending_count > 0:
        print("Re-run this job/notebook to continue — fully resumable.")

    log_run(spark, source="all", layer="gold", rows_processed=total_scored)

except Exception as e:
    log_run(spark, source="all", layer="gold", rows_processed=total_scored,
             status="failed", error_message=str(e))
    raise

# COMMAND ----------

# Quick sanity check on results so far
display(
    spark.sql(f"""
        SELECT tool, COUNT(*) AS rows_scored, ROUND(AVG(sentiment_score), 2) AS avg_sentiment
        FROM {GOLD_TABLE}
        GROUP BY tool
        ORDER BY rows_scored DESC
    """)
)

# COMMAND ----------

# Data quality checks — run after every scoring pass. Logs a warning to the
# audit table (does not fail the job) if any check finds bad data, so issues
# surface in monitoring rather than silently accumulating.

dq_issues = []

# Check 1: sentiment scores must be within the valid -1 to 1 range
out_of_range = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {GOLD_TABLE}
    WHERE sentiment_score IS NOT NULL
      AND (sentiment_score < -1 OR sentiment_score > 1)
""").collect()[0]["n"]
if out_of_range > 0:
    dq_issues.append(f"{out_of_range} rows have sentiment_score outside [-1, 1]")

# Check 2: null sentiment scores (malformed model responses that TRY_CAST caught)
null_scores = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {GOLD_TABLE} WHERE sentiment_score IS NULL
""").collect()[0]["n"]
total_rows = spark.sql(f"SELECT COUNT(*) AS n FROM {GOLD_TABLE}").collect()[0]["n"]
null_pct = round(100 * null_scores / total_rows, 2) if total_rows > 0 else 0
if null_pct > 5:   # more than 5% nulls suggests a systemic prompt/model issue, not just noise
    dq_issues.append(f"{null_scores} rows ({null_pct}%) have NULL sentiment_score — above 5% threshold")

# Check 3: no rows dated before any tool's known launch date (regression
# check for the false-positive keyword matching issue found earlier)
sys.path.append("../config")
from tools import LAUNCH_DATES  # noqa: E402

earliest_launch = min(LAUNCH_DATES.values())
pre_launch_rows = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {GOLD_TABLE} WHERE created_at < '{earliest_launch}'
""").collect()[0]["n"]
if pre_launch_rows > 0:
    dq_issues.append(f"{pre_launch_rows} rows dated before {earliest_launch} (earliest tool launch) — possible false-positive matches")

# COMMAND ----------

if dq_issues:
    issue_summary = "; ".join(dq_issues)
    print(f" Data quality issues found: {issue_summary}")
    log_run(spark, source="all", layer="gold_dq_check", rows_processed=total_rows,
             status="failed", error_message=issue_summary)
else:
    print(" All data quality checks passed.")
    log_run(spark, source="all", layer="gold_dq_check", rows_processed=total_rows, status="success")