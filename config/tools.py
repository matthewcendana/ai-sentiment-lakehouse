
"""
Central config for AI tools tracked in the sentiment lakehouse project.
Used by silver-layer cleaning scripts to tag/match mentions in raw text.
 
Each tool maps to a list of keywords/aliases (lowercase) to search for.
Keep aliases specific enough to avoid false positives (e.g. "v0" alone
is too generic — pair with context terms).
"""
 
AI_TOOLS = {
    "ChatGPT": ["chatgpt", "chat gpt", "gpt-4", "gpt4", "openai chat"],
    "Claude": ["claude", "claude.ai", "anthropic claude"],
    "Gemini": ["gemini", "google gemini", "bard"],  # bard = legacy name
    "Perplexity": ["perplexity", "perplexity ai", "perplexity.ai"],
    "Cursor": ["cursor ai", "cursor editor", "cursor ide", "cursor.sh", "cursor.com"],
    "GitHub Copilot": ["github copilot", "copilot", "gh copilot"],
    "Windsurf": ["windsurf", "windsurf editor", "windsurf ide", "codeium windsurf"],
    "NotebookLM": ["notebooklm", "notebook lm"],
    "Lovable": ["lovable", "lovable.dev", "lovable ai"],
    "Bolt.new": ["bolt.new", "bolt new", "boltnew", "stackblitz bolt"],
    "v0": ["v0.dev", "v0 by vercel", "vercel v0"],  # avoid bare "v0" -> too generic
    "Replit Agent": ["replit agent", "replit ai agent", "replit ghostwriter"],
}
 
# Flat lookup: alias -> canonical tool name (build once, reuse in matching functions)
ALIAS_TO_TOOL = {
    alias: tool
    for tool, aliases in AI_TOOLS.items()
    for alias in aliases
}
 
 
def match_tools(text: str) -> list:
    """
    Return list of canonical tool names mentioned in the given text.
    Case-insensitive substring match against known aliases.
    """
    if not text:
        return []
    text_lower = text.lower()
    matched = {
        tool for alias, tool in ALIAS_TO_TOOL.items() if alias in text_lower
    }
    return sorted(matched)