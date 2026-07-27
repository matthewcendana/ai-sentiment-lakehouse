"""
Central config for AI tools tracked in the sentiment lakehouse project.
Used by silver-layer cleaning scripts to tag/match mentions in raw text.

Aliases are split into two tiers:
  - STRONG aliases are unambiguous and safe to match on their own
    (e.g. "claude.ai", "windsurf.dev", "lovable.dev").
  - WEAK aliases are generic English words / common terms that ALSO happen
    to be a tool's bare name (e.g. "lovable", "windsurf", "claude", "gemini",
    "perplexity"). These only count as a match if the surrounding text also
    contains at least one CONTEXT_KEYWORD.

  - EXCLUSION_PHRASES are known named entities / specific phrases that must
    be actively excluded even when a WEAK alias + context keyword both
    appear — e.g. "Claude Shannon" articles are about computing/information
    theory and will often contain tech context words, which would otherwise
    defeat the context-based filter above. Exclusions are checked first and
    take priority over any match.

  - LAUNCH_DATES is the strongest filter of all: a post dated before a tool
    publicly existed cannot possibly be about that tool, regardless of what
    words matched. Applied in silver cleaning scripts as a hard date cutoff
    per tool, catching any keyword false-positive the alias/context/exclusion
    logic above missed.
"""

from datetime import date

# Approximate public launch/release dates. Deliberately set a few weeks
# before the actual public date to be safe (early access, beta announcements,
# etc. can predate the "official" launch slightly) — the goal is to filter
# out CLEARLY impossible old posts, not to be perfectly precise to the day.
LAUNCH_DATES = {
    "ChatGPT": date(2022, 11, 1),
    "Claude": date(2023, 3, 1),
    "Gemini": date(2023, 12, 1),       # launched as Gemini; Bard existed earlier but is a separate alias concern
    "Perplexity": date(2022, 12, 1),
    "Cursor": date(2023, 3, 1),
    "GitHub Copilot": date(2021, 6, 1),
    "Windsurf": date(2024, 11, 1),     # formerly Codeium, rebranded to Windsurf
    "NotebookLM": date(2023, 7, 1),
    "Lovable": date(2024, 11, 1),
    "Bolt.new": date(2024, 9, 1),
    "v0": date(2023, 10, 1),
    "Replit Agent": date(2024, 9, 1),
}

# Known non-tool named entities / phrases that would otherwise slip past the
# context-keyword check (e.g. Claude Shannon articles are inherently
# tech/computing-related, so "context present" alone isn't enough for "claude").
EXCLUSION_PHRASES = [
    "claude shannon",
    "claude monet",
    "claude debussy",
    "claude rains",   # actor
    "claude van damme",
]

# Words that signal the text is actually AI/tech-tool related. Used to
# validate WEAK alias matches. Kept as specific multi-word phrases where
# possible — bare generic words like "model" or "agent" are intentionally
# excluded since they also mean things like "data model" or "travel agent"
# and would defeat the whole point of this filter (e.g. a 2019 post titled
# "Model Driven REST framework" would otherwise false-match "Gemini").
CONTEXT_KEYWORDS = [
    "ai", "a.i.", "artificial intelligence", "llm", "chatbot", "chat bot",
    "coding assistant", "code assistant", "vibe coding", "ai prompt",
    "generative ai", "machine learning", "openai", "anthropic", "google ai",
    "app builder", "no-code app", "low-code app", "ai agent", "ai copilot",
    "language model", "ai model", "ai tool", "ai startup", "ai coding",
    "prompt engineering", "text-to-app", "text to app",
]

AI_TOOLS = {
    "ChatGPT": {
        "strong": ["chatgpt", "chat gpt", "gpt-4", "gpt4", "openai chat"],
        "weak": [],
    },
    "Claude": {
        "strong": ["claude.ai", "anthropic claude", "claude ai"],
        "weak": ["claude"],   # common first name — needs context
    },
    "Gemini": {
        "strong": ["google gemini", "gemini ai"],
        "weak": ["gemini", "bard"],   # NASA program / zodiac sign — needs context
    },
    "Perplexity": {
        "strong": ["perplexity ai", "perplexity.ai"],
        "weak": ["perplexity"],   # common English word — needs context
    },
    "Cursor": {
        "strong": ["cursor ai", "cursor editor", "cursor ide", "cursor.sh", "cursor.com"],
        "weak": [],
    },
    "GitHub Copilot": {
        "strong": ["github copilot", "gh copilot"],
        "weak": ["copilot"],   # generic term (flight copilot, etc.) — needs context
    },
    "Windsurf": {
        "strong": ["windsurf editor", "windsurf ide", "windsurf.dev", "codeium windsurf"],
        "weak": ["windsurf"],   # the actual sport — needs context
    },
    "NotebookLM": {
        "strong": ["notebooklm", "notebook lm"],
        "weak": [],
    },
    "Lovable": {
        "strong": ["lovable.dev", "lovable ai"],
        "weak": ["lovable"],   # common adjective — needs context
    },
    "Bolt.new": {
        "strong": ["bolt.new", "stackblitz bolt"],
        "weak": ["bolt new", "boltnew"],
    },
    "v0": {
        "strong": ["v0.dev", "v0 by vercel", "vercel v0"],
        "weak": [],   # bare "v0" intentionally excluded entirely — too ambiguous even with context
    },
    "Replit Agent": {
        "strong": ["replit agent", "replit ai agent", "replit ghostwriter"],
        "weak": [],
    },
}


def _has_context(text_lower: str) -> bool:
    return any(kw in text_lower for kw in CONTEXT_KEYWORDS)


def _is_excluded(text_lower: str) -> bool:
    return any(phrase in text_lower for phrase in EXCLUSION_PHRASES)


def match_tools(text: str) -> list:
    """
    Return list of canonical tool names mentioned in the given text.
    Case-insensitive substring match. STRONG aliases always count (unless
    an EXCLUSION_PHRASES entry is present, which blocks matching entirely
    for this text — e.g. "Claude Shannon" articles are excluded outright).
    WEAK aliases only count if the text also contains an AI/tech
    CONTEXT_KEYWORD, to filter out false positives from generic English words.
    """
    if not text:
        return []
    text_lower = text.lower()

    if _is_excluded(text_lower):
        return []

    context_present = _has_context(text_lower)

    matched = set()
    for tool, aliases in AI_TOOLS.items():
        if any(alias in text_lower for alias in aliases["strong"]):
            matched.add(tool)
            continue
        if context_present and any(alias in text_lower for alias in aliases["weak"]):
            matched.add(tool)

    return sorted(matched)