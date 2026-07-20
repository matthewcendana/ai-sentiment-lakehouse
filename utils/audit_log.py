# utils/audit_log.py
#
# Shared helper for logging pipeline run metadata to a queryable Delta table.
# Call log_run() once at the end of each bronze/silver/gold notebook (success
# or failure) so you have a single place to check "did today's run add new
# data, and how much" without digging through Job run history.

from datetime import datetime, timezone

AUDIT_TABLE = "ai_tool_sentiment.bronze.pipeline_audit_log"


def ensure_audit_table_exists(spark):
    """Create the audit log table if it doesn't exist yet. Safe to call every run."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
            run_timestamp   TIMESTAMP,
            source          STRING,   -- 'hackernews' | 'reddit' | 'producthunt'
            layer           STRING,   -- 'bronze' | 'silver' | 'gold'
            rows_processed  INT,
            status          STRING,   -- 'success' | 'failed'
            error_message   STRING
        )
    """)


def log_run(spark, source: str, layer: str, rows_processed: int,
            status: str = "success", error_message: str = None):
    """
    Write one row to the audit log table for this run.

    Usage (at the end of a notebook, after the main write step):
        from audit_log import ensure_audit_table_exists, log_run
        ensure_audit_table_exists(spark)
        log_run(spark, source="hackernews", layer="bronze", rows_processed=len(all_rows))

    On failure, wrap the main logic in try/except and call:
        log_run(spark, source="hackernews", layer="bronze", rows_processed=0,
                 status="failed", error_message=str(e))
    """
    ensure_audit_table_exists(spark)

    row = [(
        datetime.now(timezone.utc),
        source,
        layer,
        rows_processed,
        status,
        error_message,
    )]

    df = spark.createDataFrame(
        row,
        schema="run_timestamp timestamp, source string, layer string, "
               "rows_processed int, status string, error_message string"
    )

    df.write.format("delta").mode("append").saveAsTable(AUDIT_TABLE)
    print(f"[audit log] {layer}/{source}: {status}, {rows_processed} rows")