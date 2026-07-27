# tests/test_tools.py
#
# Unit tests for config/tools.py's match_tools() function.
# Covers the specific false-positive patterns discovered during data
# quality debugging (Claude Shannon, generic "lovable"/"windsurf", etc.)
# so regressions get caught automatically instead of resurfacing silently
# in production data.
#
# Run locally with: pytest tests/test_tools.py -v
# (requires the config/ folder on the path — run from repo root, or add
#  a conftest.py / adjust sys.path as needed for your test runner)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))

from tools import match_tools  # noqa: E402


class TestStrongAliasMatching:
    def test_chatgpt_strong_alias(self):
        assert "ChatGPT" in match_tools("I've been using ChatGPT daily for coding help.")

    def test_claude_strong_alias_no_context_needed(self):
        assert "Claude" in match_tools("Just tried claude.ai for the first time, impressed.")

    def test_cursor_strong_alias(self):
        assert "Cursor" in match_tools("Cursor IDE has great autocomplete.")


class TestWeakAliasRequiresContext:
    def test_bare_lovable_without_context_does_not_match(self):
        # "lovable" as a plain adjective, no AI context — should NOT match
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
        result = match_tools("Gemini 0.3.0 released: Model Driven REST framework.")
        # "model" was removed from context keywords specifically because of
        # this exact false positive — regression test for that fix.
        assert "Gemini" not in result

    def test_bare_perplexity_without_context_does_not_match(self):
        result = match_tools("I was in a state of perplexity after reading that math proof.")
        assert "Perplexity" not in result


class TestExclusionPhrases:
    def test_claude_shannon_excluded(self):
        result = match_tools("Claude Shannon is considered the father of information theory.")
        assert "Claude" not in result

    def test_claude_shannon_excluded_even_with_ai_context_present(self):
        # This is the key regression test: an article about Claude Shannon
        # discussing information theory/computing will often ALSO contain
        # AI-adjacent context words, which would defeat a naive context
        # check. Exclusion phrases must take priority over context matching.
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