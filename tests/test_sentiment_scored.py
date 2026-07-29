# tests/test_sentiment_scored.py
#
# Data-quality checks that run natively on Databricks against the live Delta
# table ai_tool_sentiment.gold.sentiment_scored (catalog.schema.table =
# project.layer.table) -- the Spark/cluster counterpart to
# test_gold_sentiment_scored_csv.py, which checks a local CSV export of the
# same table for use outside Databricks (e.g. in VS Code).
#
# INTENDED ENVIRONMENT: this repo synced as a Databricks Repo (Git folder),
# opened from the workspace file editor's Tests pane, on a cluster with
# Unity Catalog read access to ai_tool_sentiment.gold. Needs the folder
# structure preserved (config/ as a sibling of tests/) for the
# `from tools import LAUNCH_DATES` import below.
#

import os
import sys

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.utils import AnalysisException  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
from tools import LAUNCH_DATES  # noqa: E402

CATALOG = "ai_tool_sentiment"
GOLD_TABLE = f"{CATALOG}.gold.sentiment_scored"
VIEW_NAME = "gold_sentiment_scored_dq_view"

EXPECTED_COLUMNS = {
    "post_id", "source", "tool", "title", "text", "author", "score",
    "num_comments", "url", "created_at", "sentiment_score",
}

# SparkSession.builder.getOrCreate() returns the cluster's already-active
# session rather than starting a new one -- notebooks get `spark` injected
# automatically, but a plain .py file like this one has to ask for it.
spark = SparkSession.builder.getOrCreate()

_setup_error = None
_columns = set()
_row_count = 0

try:
    _gold_df = spark.table(GOLD_TABLE)
    _gold_df.createOrReplaceTempView(VIEW_NAME)
    _columns = set(_gold_df.columns)
    _row_count = spark.sql(f"SELECT COUNT(*) AS n FROM {VIEW_NAME}").collect()[0]["n"]

    launch_dates_df = spark.createDataFrame(
        [(tool, launch) for tool, launch in LAUNCH_DATES.items()],
        schema="tool string, launch_date date",
    )
    launch_dates_df.createOrReplaceTempView("tool_launch_dates")
except AnalysisException as e:
    _setup_error = str(e)

pytestmark = pytest.mark.skipif(
    _setup_error is not None,
    reason=f"Could not read {GOLD_TABLE} (check cluster's Unity Catalog access): {_setup_error}",
)


class TestSchema:
    def test_not_empty(self):
        assert _row_count > 0, f"{GOLD_TABLE} has no rows"

    def test_has_expected_columns(self):
        missing = EXPECTED_COLUMNS - _columns
        assert not missing, f"{GOLD_TABLE} missing expected columns: {missing}"


class TestSentimentScoreQuality:
    def test_all_scores_within_valid_range(self):
        out_of_range = spark.sql(f"""
            SELECT COUNT(*) AS n FROM {VIEW_NAME}
            WHERE sentiment_score IS NOT NULL
              AND (sentiment_score < -1 OR sentiment_score > 1)
        """).collect()[0]["n"]
        assert out_of_range == 0, (
            f"{out_of_range} rows in {GOLD_TABLE} have sentiment_score outside [-1, 1]"
        )

    def test_null_sentiment_scores_below_five_percent(self):
        # Same 5% threshold as the DQ check in gold/sentiment_scoring.py --
        # above that suggests a systemic prompt/model issue, not just noise.
        null_count = spark.sql(f"""
            SELECT COUNT(*) AS n FROM {VIEW_NAME} WHERE sentiment_score IS NULL
        """).collect()[0]["n"]
        null_pct = 100 * null_count / _row_count if _row_count else 0
        assert null_pct <= 5, (
            f"{null_count} rows ({null_pct:.1f}%) in {GOLD_TABLE} have a null sentiment_score"
        )


class TestNoDuplicates:
    def test_no_duplicate_post_id_tool_pairs(self):
        dupes = spark.sql(f"""
            SELECT post_id, tool, COUNT(*) AS n
            FROM {VIEW_NAME}
            GROUP BY post_id, tool
            HAVING COUNT(*) > 1
        """).collect()
        assert not dupes, (
            f"{len(dupes)} duplicate (post_id, tool) pairs in {GOLD_TABLE} "
            f"(showing up to 5): {dupes[:5]}"
        )


class TestLaunchDateIntegrity:
    # Regression guard for the original data-quality incident: a post dated
    # before a tool's LAUNCH_DATES entry cannot possibly be about that tool.
    #
    # This is a per-tool join, same fix as the (now-corrected) Check 3 in
    # gold/sentiment_scoring.py -- comparing against a single global
    # min(LAUNCH_DATES.values()) instead would miss violations for every
    # tool whose own launch date is later than that minimum.

    def test_no_rows_predate_their_tool_launch_date(self):
        violations = spark.sql(f"""
            SELECT g.post_id, g.tool, g.created_at
            FROM {VIEW_NAME} g
            JOIN tool_launch_dates l ON g.tool = l.tool
            WHERE g.created_at < CAST(l.launch_date AS TIMESTAMP)
        """).collect()

        assert not violations, (
            f"{len(violations)} rows in {GOLD_TABLE} dated before their tool's "
            f"launch date (showing up to 5): "
            f"{[(r['post_id'], r['tool'], str(r['created_at'])) for r in violations[:5]]}"
        )
