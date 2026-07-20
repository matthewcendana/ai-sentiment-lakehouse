# Databricks notebook source
# bronze/ingest_hackernews.py
#
# Pulls stories + comments mentioning tracked AI tools from the Algolia
# Hacker News Search API (no auth required) and lands raw results into
# a Delta bronze table: bronze.hn_raw
#
# INCREMENTAL: only pulls results created after the latest hn_created_at
# already in the bronze table, using Algolia's numericFilters + sort-by-date
# search, so scheduled re-runs don't keep re-fetching the same old posts.
#
# Run manually or as a scheduled Databricks Job.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
import requests
import time
from datetime import datetime, timezone
from pyspark.sql import Row
from pyspark.sql.utils import AnalysisException

sys.path.append("../config")
sys.path.append("../utils")

from tools import AI_TOOLS  # noqa: E402
from audit_log import log_run  # noqa: E402

# COMMAND ----------

CATALOG = "ai_tool_sentiment"   # adjust to your Unity Catalog name
SCHEMA = "bronze"
TABLE = "hn_raw"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Use search_by_date endpoint (not the relevance-ranked default) so results
# come back in a predictable, page-through-safe chronological order.
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
HITS_PER_PAGE = 100
MAX_PAGES_PER_QUERY = 10   # raised slightly since incremental pulls should be smaller per run
REQUEST_DELAY_SEC = 0.5

# COMMAND ----------

def get_last_ingested_timestamp() -> int:
    """
    Returns the max hn_created_at (as unix seconds) already in bronze, so we
    only fetch newer results. Returns 0 (epoch) if the table doesn't exist yet
    or is empty — meaning "pull everything" on the very first run.
    """
    try:
        result = spark.sql(f"""
            SELECT MAX(hn_created_at) AS max_ts FROM {FULL_TABLE_NAME}
        """).collect()[0]["max_ts"]
        if result is None:
            return 0
        # hn_created_at is stored as ISO string from the API; convert to unix ts
        dt = datetime.fromisoformat(result.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except AnalysisException:
        # Table doesn't exist yet — first run, pull everything
        return 0

# COMMAND ----------

def fetch_hn_results(query: str, since_ts: int, page: int = 0):
    """Query Algolia HN Search API (by date) for a single term, only results after since_ts."""
    params = {
        "query": query,
        "tags": "(story,comment)",
        "hitsPerPage": HITS_PER_PAGE,
        "page": page,
        "numericFilters": f"created_at_i>{since_ts}",
    }
    resp = requests.get(HN_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pull_all_pages(query: str, since_ts: int):
    """Page through NEW results only (search_by_date + numericFilters handles the cutoff)."""
    all_hits = []
    page = 0
    while page < MAX_PAGES_PER_QUERY:
        data = fetch_hn_results(query, since_ts, page=page)
        hits = data.get("hits", [])
        if not hits:
            break
        all_hits.extend(hits)
        if page >= data.get("nbPages", 1) - 1:
            break
        page += 1
        time.sleep(REQUEST_DELAY_SEC)
    return all_hits

# COMMAND ----------

def build_rows(query_tool: str, hits: list) -> list:
    """Convert raw API hits into flat rows for the bronze table."""
    ingested_at = datetime.now(timezone.utc)
    rows = []
    for hit in hits:
        rows.append(
            Row(
                object_id=hit.get("objectID"),
                query_tool=query_tool,
                title=hit.get("title"),
                text=hit.get("comment_text") or hit.get("story_text") or hit.get("title"),
                author=hit.get("author"),
                points=hit.get("points"),
                num_comments=hit.get("num_comments"),
                url=hit.get("url"),
                hn_created_at=hit.get("created_at"),
                object_type="comment" if hit.get("comment_text") else "story",
                ingested_at=ingested_at,
            )
        )
    return rows

# COMMAND ----------

since_ts = get_last_ingested_timestamp()
if since_ts == 0:
    print("No prior data found — this is a first run, pulling all available results.")
else:
    print(f"Incremental run — only pulling HN results after {datetime.fromtimestamp(since_ts, tz=timezone.utc)}")

all_rows = []

for tool_name in AI_TOOLS.keys():
    print(f"Fetching HN results for: {tool_name}")
    try:
        hits = pull_all_pages(tool_name, since_ts)
        rows = build_rows(tool_name, hits)
        all_rows.extend(rows)
        print(f"  -> {len(rows)} new hits")
    except requests.RequestException as e:
        print(f"  !! Failed for {tool_name}: {e}")
    time.sleep(REQUEST_DELAY_SEC)

print(f"Total new rows collected: {len(all_rows)}")

# COMMAND ----------

if all_rows:
    df = spark.createDataFrame(all_rows)

    try:
        (
            df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(FULL_TABLE_NAME)
        )
        row_count = df.count()
        print(f"Wrote {row_count} new rows to {FULL_TABLE_NAME}")
        log_run(spark, source="hackernews", layer="bronze", rows_processed=row_count)
    except Exception as e:
        log_run(spark, source="hackernews", layer="bronze", rows_processed=0,
                 status="failed", error_message=str(e))
        raise
else:
    print("No new rows to write.")
    log_run(spark, source="hackernews", layer="bronze", rows_processed=0)