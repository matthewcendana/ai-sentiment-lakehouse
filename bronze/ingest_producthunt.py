# Databricks notebook source
# bronze/ingest_producthunt.py
#
# Pulls posts (+ top comments) from Product Hunt's official GraphQL API v2
# (newest-first) and lands raw results into a Delta bronze table:
# bronze.producthunt_raw
#
# INCREMENTAL: stops paginating as soon as it reaches a post older than the
# latest one already in bronze, so scheduled re-runs don't keep re-fetching
# the same old posts.
#
# Requires a Product Hunt Developer Token and Databricks Secrets scope
# "producthunt" with key: api_token.
#
# Run manually or as a scheduled Databricks Job.

# COMMAND ----------

import sys
import time
from datetime import datetime, timezone

import requests
from pyspark.sql import Row
from pyspark.sql.utils import AnalysisException

sys.path.append("../config")
sys.path.append("../utils")

from audit_log import log_run  # noqa: E402

# COMMAND ----------

CATALOG = "ai_tool_sentiment"
SCHEMA = "bronze"
TABLE = "producthunt_raw"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

PH_API_URL = "https://api.producthunt.com/v2/api/graphql"
POSTS_PER_QUERY = 20   # PH API page size, max ~20 per page typically
MAX_PAGES_PER_QUERY = 3
REQUEST_DELAY_SEC = 1

# COMMAND ----------

api_token = dbutils.secrets.get(scope="producthunt", key="api_token")

HEADERS = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json",
}

# COMMAND ----------

# GraphQL query: Product Hunt's API does NOT support keyword search on posts.
# We pull recent posts broadly (paginated) and filter for tool mentions
# client-side using match_tools() in build_rows() below.
SEARCH_QUERY = """
query RecentPosts($after: String) {
  posts(first: %d, after: $after, order: NEWEST) {
    edges {
      node {
        id
        name
        tagline
        description
        votesCount
        commentsCount
        url
        createdAt
        comments(first: 5) {
          edges {
            node {
              id
              body
              createdAt
              user { username }
            }
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
""" % POSTS_PER_QUERY

# COMMAND ----------

def get_last_ingested_timestamp():
    """
    Returns the max ph_created_at already in bronze as a datetime, or None if
    the table doesn't exist yet / is empty — meaning "pull everything" on the
    first run.
    """
    try:
        result = spark.sql(f"""
            SELECT MAX(ph_created_at) AS max_ts FROM {FULL_TABLE_NAME}
        """).collect()[0]["max_ts"]
        if result is None:
            return None
        return datetime.fromisoformat(result.replace("Z", "+00:00"))
    except AnalysisException:
        return None

# COMMAND ----------

def fetch_ph_page(after_cursor: str = None):
    """Pull one page of recent posts. No keyword search — PH API doesn't support it."""
    variables = {"after": after_cursor}
    payload = {"query": SEARCH_QUERY, "variables": variables}

    resp = requests.post(PH_API_URL, json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "errors" in data:
        raise RuntimeError(f"PH API error: {data['errors']}")

    return data["data"]["posts"]

# COMMAND ----------

def build_post_rows(edges: list) -> list:
    """
    Flatten post + comments into raw bronze rows. NO filtering here — bronze
    stores everything as pulled. Tool matching/filtering happens in
    silver/clean_producthunt.py via match_tools(), keeping bronze a true raw layer.
    One row per post AND per comment.
    """
    rows = []
    ingested_at = datetime.now(timezone.utc)

    for edge in edges:
        post = edge["node"]
        post_text = f"{post.get('tagline', '')} {post.get('description', '')}".strip()

        rows.append(
            Row(
                object_id=post["id"],
                title=post.get("name"),
                text=post_text,
                author=None,
                votes_count=post.get("votesCount"),
                num_comments=post.get("commentsCount"),
                url=post.get("url"),
                ph_created_at=post.get("createdAt"),
                object_type="post",
                ingested_at=ingested_at,
            )
        )

        for c_edge in post.get("comments", {}).get("edges", []):
            comment = c_edge["node"]
            rows.append(
                Row(
                    object_id=comment["id"],
                    title=post.get("name"),
                    text=comment.get("body"),
                    author=(comment.get("user") or {}).get("username"),
                    votes_count=None,
                    num_comments=None,
                    url=post.get("url"),
                    ph_created_at=comment.get("createdAt"),
                    object_type="comment",
                    ingested_at=ingested_at,
                )
            )

    return rows

# COMMAND ----------

# PH API has no keyword search, so we paginate through recent posts broadly
# (newest-first) and store everything raw. Filtering for AI-tool relevance
# happens in silver/clean_producthunt.py, not here.
# TOTAL_PAGES is a safety cap in case since_ts is None (first run) — otherwise
# we stop early once we hit posts older than since_ts.
TOTAL_PAGES = 20   # ~20 pages x 20 posts/page = up to 400 recent posts per run

since_ts = get_last_ingested_timestamp()
if since_ts is None:
    print("No prior data found — this is a first run, pulling all available results.")
else:
    print(f"Incremental run — only pulling PH posts after {since_ts}")

all_rows = []
after_cursor = None
page = 0
reached_old_posts = False

print("Fetching Product Hunt posts (raw pull, no filtering — that happens in silver)...")

try:
    while page < TOTAL_PAGES and not reached_old_posts:
        posts_data = fetch_ph_page(after_cursor)
        edges = posts_data["edges"]

        if not edges:
            break

        # Since results come back newest-first, filter out (and stop after)
        # any post at or older than our cutoff.
        new_edges = []
        for edge in edges:
            post_created = edge["node"].get("createdAt")
            post_dt = datetime.fromisoformat(post_created.replace("Z", "+00:00")) if post_created else None

            if since_ts is not None and post_dt is not None and post_dt <= since_ts:
                reached_old_posts = True
                break   # everything from here on is already ingested
            new_edges.append(edge)

        rows = build_post_rows(new_edges)
        all_rows.extend(rows)
        print(f"  Page {page + 1}: pulled {len(new_edges)} new posts, {len(rows)} rows this page")

        if reached_old_posts:
            print("  Reached already-ingested posts — stopping.")
            break

        page_info = posts_data["pageInfo"]
        if not page_info["hasNextPage"]:
            print("  Reached end of available posts.")
            break

        after_cursor = page_info["endCursor"]
        page += 1
        time.sleep(REQUEST_DELAY_SEC)

except Exception as e:
    print(f"  !! Failed during pagination: {e}")

print(f"Total new rows collected: {len(all_rows)}")

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
        log_run(spark, source="producthunt", layer="bronze", rows_processed=row_count)
    except Exception as e:
        log_run(spark, source="producthunt", layer="bronze", rows_processed=0,
                 status="failed", error_message=str(e))
        raise
else:
    print("No new rows to write.")
    log_run(spark, source="producthunt", layer="bronze", rows_processed=0)