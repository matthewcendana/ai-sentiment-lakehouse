# Databricks notebook source
# bronze/ingest_reddit.py
#
# Pulls submissions mentioning tracked AI tools from Reddit via PRAW and lands
# raw results into a Delta bronze table: bronze.reddit_raw
#
# INCREMENTAL: sorts search results newest-first and stops as soon as it
# reaches a submission older than the latest one already in bronze, so
# scheduled re-runs don't keep re-fetching the same old posts.
#
# Requires a Reddit API app (reddit.com/prefs/apps) and Databricks Secrets
# scope "reddit" with keys: client_id, client_secret, user_agent.
#
# Run manually or as a scheduled Databricks Job.

import sys
import time
from datetime import datetime, timezone

import praw
from pyspark.sql import Row
from pyspark.sql.utils import AnalysisException

sys.path.append("../config")
sys.path.append("../utils")

from tools import AI_TOOLS  # noqa: E402
from audit_log import log_run  # noqa: E402

# COMMAND ----------

CATALOG = "ai_tool_sentiment"
SCHEMA = "bronze"
TABLE = "reddit_raw"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

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

POSTS_PER_QUERY = 100
REQUEST_DELAY_SEC = 1

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

def get_last_ingested_timestamp() -> float:
    """
    Returns the max reddit_created_utc already in bronze, as a unix timestamp.
    Returns 0 (epoch) if the table doesn't exist yet or is empty — meaning
    "pull everything" on the very first run.
    """
    try:
        result = spark.sql(f"""
            SELECT MAX(reddit_created_utc) AS max_ts FROM {FULL_TABLE_NAME}
        """).collect()[0]["max_ts"]
        if result is None:
            return 0.0
        return result.timestamp()
    except AnalysisException:
        return 0.0

# COMMAND ----------

def search_subreddit_for_tool(subreddit_name: str, tool_name: str, since_ts: float):
    """
    Search a subreddit for a tool name sorted newest-first, stopping as soon as
    results drop below since_ts (everything after that point is already ingested).
    """
    subreddit = reddit.subreddit(subreddit_name)
    results = []
    try:
        for submission in subreddit.search(tool_name, limit=POSTS_PER_QUERY, sort="new"):
            if submission.created_utc <= since_ts:
                break   # hit already-ingested territory — stop, since results are newest-first
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

since_ts = get_last_ingested_timestamp()
if since_ts == 0:
    print("No prior data found — this is a first run, pulling all available results.")
else:
    print(f"Incremental run — only pulling Reddit posts after {datetime.fromtimestamp(since_ts, tz=timezone.utc)}")

all_rows = []

for subreddit_name in SUBREDDITS:
    for tool_name in AI_TOOLS.keys():
        print(f"Searching r/{subreddit_name} for: {tool_name}")
        submissions = search_subreddit_for_tool(subreddit_name, tool_name, since_ts)
        rows = [build_submission_row(s, tool_name) for s in submissions]
        all_rows.extend(rows)
        print(f"  -> {len(rows)} new submissions")
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
        log_run(spark, source="reddit", layer="bronze", rows_processed=row_count)
    except Exception as e:
        log_run(spark, source="reddit", layer="bronze", rows_processed=0,
                 status="failed", error_message=str(e))
        raise
else:
    print("No new rows to write.")
    log_run(spark, source="reddit", layer="bronze", rows_processed=0)