# Databricks notebook source
# bronze/ingest_producthunt.py
#
# Pulls posts (+ top comments) mentioning tracked AI tools from Product Hunt's
# official GraphQL API v2 and lands raw results into a Delta bronze table:
# bronze.producthunt_raw
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

sys.path.append("../config")
sys.path.append("../utils")

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
# and store everything raw. Filtering for AI-tool relevance happens in
# silver/clean_producthunt.py, not here.
# Increase TOTAL_PAGES to pull further back in time / more volume per run.
TOTAL_PAGES = 20   # ~20 pages x 20 posts/page = up to 400 recent posts per run

all_rows = []
after_cursor = None
page = 0

print("Fetching recent Product Hunt posts (raw pull, no filtering — that happens in silver)...")

try:
    while page < TOTAL_PAGES:
        posts_data = fetch_ph_page(after_cursor)
        edges = posts_data["edges"]

        if not edges:
            break

        rows = build_post_rows(edges)
        all_rows.extend(rows)
        print(f"  Page {page + 1}: pulled {len(edges)} posts, {len(rows)} rows total this page")

        page_info = posts_data["pageInfo"]
        if not page_info["hasNextPage"]:
            print("  Reached end of available posts.")
            break

        after_cursor = page_info["endCursor"]
        page += 1
        time.sleep(REQUEST_DELAY_SEC)

except Exception as e:
    print(f"  !! Failed during pagination: {e}")

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
    print("No rows to write. Check API query support / token scope.")