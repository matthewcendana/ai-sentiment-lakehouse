# Databricks notebook source
# bronze/ingest_reddit.py
#
# Pulls submissions + comments mentioning tracked AI tools from Reddit
# via PRAW and lands raw results into a Delta bronze table: bronze.reddit_raw
#

# COMMAND ----------

%pip install -r ../requirements.txt

# COMMAND ----------

import sys
import time
from datetime import datetime, timezone

import praw
from pyspark.sql import Row

sys.path.append("../config")
sys.path.append("../utils")

from tools import AI_TOOLS  # noqa: E402

# COMMAND ----------

CATALOG = "ai_tool_sentiment"
SCHEMA = "bronze"
TABLE = "reddit_raw"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Subreddits likely to discuss AI tools — tune this list as you go
SUBREDDITS = [
    "artificial",
    "ChatGPT",
    "ClaudeAI",
    "LocalLLaMA",
    "singularity",
    "OpenAI",
    "programming",
    "webdev",
]

POSTS_PER_QUERY = 100   # PRAW max per search call
REQUEST_DELAY_SEC = 1   # be polite to the API / avoid rate limits

# COMMAND ----------

client_id = dbutils.secrets.get(scope="reddit", key="client_id")
client_secret = dbutils.secrets.get(scope="reddit", key="client_secret")
user_agent = dbutils.secrets.get(scope="reddit", key="user_agent")

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)
reddit.read_only = True

# COMMAND ----------

def search_subreddit_for_tool(subreddit_name: str, tool_name: str):
    """Search a subreddit for a tool name, return list of submissions."""
    subreddit = reddit.subreddit(subreddit_name)
    results = []
    try:
        for submission in subreddit.search(tool_name, limit=POSTS_PER_QUERY, sort="new"):
            results.append(submission)
    except Exception as e:
        print(f"  !! Search failed for r/{subreddit_name} / {tool_name}: {e}")
    return results


def build_submission_row(submission, query_tool: str) -> Row:
    return Row(
        object_id=submission.id,
        query_tool=query_tool,
        subreddit=str(submission.subreddit),
        title=submission.title,
        text=submission.selftext,
        author=str(submission.author) if submission.author else None,
        score=submission.score,
        num_comments=submission.num_comments,
        url=submission.url,
        permalink=f"https://reddit.com{submission.permalink}",
        reddit_created_utc=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
        object_type="submission",
        ingested_at=datetime.now(timezone.utc),
    )

# COMMAND ----------

all_rows = []

for subreddit_name in SUBREDDITS:
    for tool_name in AI_TOOLS.keys():
        print(f"Searching r/{subreddit_name} for: {tool_name}")
        submissions = search_subreddit_for_tool(subreddit_name, tool_name)
        rows = [build_submission_row(s, tool_name) for s in submissions]
        all_rows.extend(rows)
        print(f"  -> {len(rows)} submissions")
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