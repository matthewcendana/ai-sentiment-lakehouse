# tests/test_tools.py
#
# Unit tests for config/tools.py's match_tools() -- regression coverage for
# false-positive patterns found during data quality debugging (Claude
# Shannon, bare "lovable"/"windsurf", etc).
#
# Run: pytest tests/test_tools.py -v

import sys
import os
from datetime import date

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))

from tools import match_tools, AI_TOOLS, LAUNCH_DATES  # noqa: E402


class TestStrongAliasMatching:
    def test_chatgpt_strong_alias(self):
        assert "ChatGPT" in match_tools("I've been using ChatGPT daily for coding help.")

    def test_claude_strong_alias_no_context_needed(self):
        assert "Claude" in match_tools("Just tried claude.ai for the first time, impressed.")

    def test_cursor_strong_alias(self):
        assert "Cursor" in match_tools("Cursor IDE has great autocomplete.")


class TestWeakAliasRequiresContext:
    def test_bare_lovable_without_context_does_not_match(self):
        result = match_tools("What a lovable little dog you have!")
        assert "Lovable" not in result

    def test_bare_lovable_with_context_matches(self):
        result = match_tools("Lovable is such an easy AI app builder, I built my MVP in an hour.")
        assert "Lovable" in result

    def test_windsurf_sport_without_context_does_not_match(self):
        result = match_tools("Went windsurfing at the lake this weekend, the wind was perfect.")
        assert "Windsurf" not in result

    def test_windsurf_tool_with_context_matches(self):
        result = match_tools("Windsurf has become my favorite AI coding assistant this year.")
        assert "Windsurf" in result

    def test_bare_gemini_without_context_does_not_match(self):
        # Regression test: "model" was removed from CONTEXT_KEYWORDS
        # specifically because of this false positive.
        result = match_tools("Gemini 0.3.0 released: Model Driven REST framework.")
        assert "Gemini" not in result

    def test_bare_perplexity_without_context_does_not_match(self):
        result = match_tools("I was in a state of perplexity after reading that math proof.")
        assert "Perplexity" not in result


class TestExclusionPhrases:
    def test_claude_shannon_excluded(self):
        result = match_tools("Claude Shannon is considered the father of information theory.")
        assert "Claude" not in result

    def test_claude_shannon_excluded_even_with_ai_context_present(self):
        # Exclusion phrases must beat context matching -- Shannon articles
        # are inherently tech-flavored, so context alone wouldn't filter them.
        result = match_tools(
            "Claude Shannon's work laid the foundation for machine learning "
            "and modern AI systems decades before they existed."
        )
        assert "Claude" not in result

    def test_claude_monet_excluded(self):
        result = match_tools("Claude Monet's paintings defined French Impressionism.")
        assert "Claude" not in result


class TestMultiToolText:
    def test_multiple_tools_in_one_text_all_matched(self):
        result = match_tools("Comparing ChatGPT vs Claude vs Gemini for coding tasks, all AI models.")
        assert "ChatGPT" in result
        assert "Claude" in result
        assert "Gemini" in result

    def test_no_duplicate_entries(self):
        result = match_tools("ChatGPT ChatGPT ChatGPT is mentioned three times in this AI post.")
        assert result.count("ChatGPT") == 1


class TestEdgeCases:
    def test_empty_string_returns_empty_list(self):
        assert match_tools("") == []

    def test_none_returns_empty_list(self):
        assert match_tools(None) == []

    def test_no_tools_mentioned_returns_empty_list(self):
        assert match_tools("I went to the store and bought some groceries today.") == []

    def test_case_insensitive_matching(self):
        assert "ChatGPT" in match_tools("CHATGPT is capitalized differently here.")


class TestConfigConsistency:
    # A tool missing from LAUNCH_DATES silently disables its date-cutoff
    # guard in the silver-layer filter (NULL launch date = no restriction).

    def test_every_ai_tool_has_a_launch_date(self):
        missing = set(AI_TOOLS) - set(LAUNCH_DATES)
        assert not missing, f"AI_TOOLS entries missing from LAUNCH_DATES: {missing}"

    def test_no_orphaned_launch_dates(self):
        orphaned = set(LAUNCH_DATES) - set(AI_TOOLS)
        assert not orphaned, f"LAUNCH_DATES entries with no matching AI_TOOLS entry: {orphaned}"

    def test_launch_dates_are_date_objects(self):
        bad = {k: type(v) for k, v in LAUNCH_DATES.items() if not isinstance(v, date)}
        assert not bad, f"LAUNCH_DATES values must be datetime.date instances: {bad}"


class TestKnownLimitations:
    # No exclusion-phrase backstop for generic "lovable" usage (unlike
    # "claude"). In production this is caught by the LAUNCH_DATES cutoff,
    # not by match_tools() itself -- documented gap, not a passing guarantee.
    @pytest.mark.xfail(
        reason="No exclusion-phrase backstop for generic 'lovable' usage; "
               "only caught by the LAUNCH_DATES cutoff, not by match_tools() itself.",
        strict=True,
    )
    def test_lovable_generic_usage_with_unrelated_ai_context_does_not_match(self):
        result = match_tools(
            "This keyboard brand is so lovable, and its companion app even "
            "has an AI typing assistant."
        )
        assert "Lovable" not in result
