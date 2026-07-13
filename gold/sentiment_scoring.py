# Databricks notebook source
# gold/sentiment_scoring.py
#
# Scores sentiment toward each AI tool using ai_query() against a Databricks-hosted
# foundation model. Runs in small batches and is RESUMABLE: safe to stop and
# re-run later (e.g. if you exhaust free-plan quota/credits) without re-scoring
# rows that already succeeded.
#
# Requires a SERVERLESS SQL warehouse or serverless compute — ai_query() does not
# work on Classic/Pro warehouses.

# COMMAND ----------

CATALOG = "ai_tool_sentiment"
GOLD_TABLE = f"{CATALOG}.gold.sentiment_scored"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

BATCH_SIZE = 500          # rows scored per batch — adjust down if you hit quota issues
MAX_TEXT_CHARS = 800       # truncate long posts to control cost per call

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
# Add producthunt_clean here later — no other changes needed elsewhere.
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW silver_union AS
SELECT * FROM {CATALOG}.silver.hn_clean
UNION ALL
SELECT * FROM {CATALOG}.silver.reddit_clean
""")

# COMMAND ----------

# Rows still needing scoring: in silver_union but NOT already in gold
# (keyed on post_id + tool, matching the silver explode grain).
# This is what makes reruns safe/resumable.
spark.sql("""
CREATE OR REPLACE TEMP VIEW pending_rows AS
SELECT su.*
FROM silver_union su
LEFT ANTI JOIN (SELECT post_id, tool FROM {gold}) g
    ON su.post_id = g.post_id AND su.tool = g.tool
WHERE su.text IS NOT NULL AND LENGTH(su.text) > 10
""".format(gold=GOLD_TABLE))

pending_count = spark.sql("SELECT COUNT(*) AS n FROM pending_rows").collect()[0]["n"]
print(f"Rows remaining to score: {pending_count}")

# COMMAND ----------

# Process in small batches. Re-run this cell (or the whole notebook) as many
# times as needed — each run only scores what's left, up to BATCH_SIZE rows,
# then stops. This keeps each execution small and controllable.

if pending_count == 0:
    print("Nothing to score — all rows already processed.")
else:
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
            CAST(
                ai_query(
                    '{MODEL_ENDPOINT}',
                    CONCAT(
                        'Rate the sentiment toward "', tool, '" expressed in this text, ',
                        'on a scale from -1 (very negative) to 1 (very positive), 0 being neutral. ',
                        'Respond with ONLY the numeric score, nothing else.

Text: ', LEFT(text, {MAX_TEXT_CHARS})
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

    scored_count = scored_df.count()
    print(f"Scored and wrote {scored_count} rows to {GOLD_TABLE}")
    print(f"Remaining after this batch: {pending_count - scored_count}")
    print("Re-run this notebook to continue scoring the rest.")

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