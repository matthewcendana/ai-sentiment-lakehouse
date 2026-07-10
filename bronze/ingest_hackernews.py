# Databricks notebook source
# bronze/ingest_hackernews.py
#
# Pulls stories + comments mentioning tracked AI tools from the Algolia
# Hacker News Search API (no auth required) and lands raw results into
# a Delta bronze table: bronze.hn_raw
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
 
sys.path.append("../config")
sys.path.append("../utils")
 
from tools import AI_TOOLS  # noqa: E402
 
# COMMAND ----------
 
CATALOG = "ai_tool_sentiment"   # adjust to your Unity Catalog name
SCHEMA = "bronze"
TABLE = "hn_raw"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"
 
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HITS_PER_PAGE = 100
MAX_PAGES_PER_QUERY = 5   # cap to avoid runaway pulls; tune as needed
REQUEST_DELAY_SEC = 0.5   # be polite to the API
 
# COMMAND ----------
 
def fetch_hn_results(query: str, page: int = 0):
    """Query Algolia HN Search API for a single search term."""
    params = {
        "query": query,
        "tags": "(story,comment)",
        "hitsPerPage": HITS_PER_PAGE,
        "page": page,
    }
    resp = requests.get(HN_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()
 
 
def pull_all_pages(query: str):
    """Page through results for a single query up to MAX_PAGES_PER_QUERY."""
    all_hits = []
    page = 0
    while page < MAX_PAGES_PER_QUERY:
        data = fetch_hn_results(query, page=page)
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
 
all_rows = []
 
# Use each tool's primary name as the search query (aliases handled at silver layer)
for tool_name in AI_TOOLS.keys():
    print(f"Fetching HN results for: {tool_name}")
    try:
        hits = pull_all_pages(tool_name)
        rows = build_rows(tool_name, hits)
        all_rows.extend(rows)
        print(f"  -> {len(rows)} hits")
    except requests.RequestException as e:
        print(f"  !! Failed for {tool_name}: {e}")
    time.sleep(REQUEST_DELAY_SEC)
 
print(f"Total rows collected: {len(all_rows)}")
 
# COMMAND ----------
 
if all_rows:
    df = spark.createDataFrame(all_rows)
 
    (
        df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(FULL_TABLE_NAME)
    )
 
    print(f"Wrote {df.count()} rows to {FULL_TABLE_NAME}")
else:
    print("No rows to write.")