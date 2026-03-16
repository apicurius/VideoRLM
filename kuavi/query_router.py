"""Rule-based query router for tiered execution."""

from __future__ import annotations

import re


class QueryRouter:
    """
    Classifies an incoming natural language query and routes it
    to the appropriate execution tier. Zero external calls — pure
    rule-based classification using regex + keyword matching.
    """

    TIER_1_PATTERNS = [
        # Temporal probes
        r"what (is |was )?(happening|going on|occurring) at \d",
        r"what (happens|happened) (at|around|near) (minute|second|timestamp)",
        r"what comes (before|after)",
        r"what (action|activity|motion) is",
        r"is (the motion|this motion|this|it) (natural|smooth|continuous|coherent)",
        r"predict (what|the next)",
        r"what sport",
        r"what (is|are) (they|he|she|the person) doing",
        r"classify (this|the) (scene|segment|action|activity)",
    ]

    TIER_2_PATTERNS = [
        # Search + retrieval
        r"find (the )?(scene|moment|part|segment|clip) where",
        r"find (the )?(scene|moment|part|segment|clip) when",
        r"when does",
        r"at what (time|point|moment|timestamp)",
        r"show me (the part|where|when)",
        r"search for",
        r"locate (the|a)",
        r"which (scene|segment|part|moment)",
        r"(transcript|caption|subtitle)",
        r"what (is said|was said|do they say|does .* say)",
        # Multiple choice VQA patterns
        r"\(A\)|\(B\)|\(C\)|\(D\)",
        r"choose (one|between|from)",
        r"which of the following",
        r"select the (best|correct)",
    ]

    TIER_3_PATTERNS = [
        # Open-ended — LLM required
        r"(summarize|summary|overview)",
        r"explain (what|why|how)",
        r"why (did|does|is|was)",
        r"what (caused|is the reason|led to)",
        r"describe (the|what|everything)",
        r"tell me (about|everything|what)",
    ]

    def classify(self, query: str) -> dict:
        """
        Returns:
        {
            "tier": 1 | 2 | 3,
            "reason": str,           # human-readable explanation
            "suggested_tools": list[str],  # which kuavi tools to call
            "output_format": str,    # "timestamp" | "label" | "multiple_choice" | "text"
        }
        """
        query_lower = query.lower().strip()

        # Check Tier 1 first
        for pattern in self.TIER_1_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "tier": 1,
                    "reason": f"Matched temporal/classification pattern: {pattern}",
                    "suggested_tools": self._suggest_tier1_tools(query_lower),
                    "output_format": self._infer_output_format(query_lower),
                }

        # Check Tier 2
        for pattern in self.TIER_2_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "tier": 2,
                    "reason": f"Matched search/retrieval pattern: {pattern}",
                    "suggested_tools": self._suggest_tier2_tools(query_lower),
                    "output_format": self._infer_output_format(query_lower),
                }

        # Check Tier 3
        for pattern in self.TIER_3_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "tier": 3,
                    "reason": f"Matched open-ended pattern: {pattern}",
                    "suggested_tools": ["full_agent"],
                    "output_format": "text",
                }

        # Default: Tier 2 (search is safer default than LLM)
        return {
            "tier": 2,
            "reason": "No pattern matched — defaulting to semantic search",
            "suggested_tools": ["search_all"],
            "output_format": "text",
        }

    def _suggest_tier1_tools(self, query: str) -> list[str]:
        """Map query to specific V-JEPA tools."""
        tools = []
        if any(w in query for w in ["action", "sport", "doing", "activity", "motion"]):
            tools.append("classify_segment")
        if any(w in query for w in ["next", "after", "predict", "future"]):
            tools.append("predict_next_action")
        if any(w in query for w in ["coherent", "smooth", "natural", "continuous"]):
            tools.append("verify_temporal_coherence")
        if any(w in query for w in ["happening", "going on", "occurring", "at minute", "at second"]):
            tools.extend(["orient", "extract_frames"])
        return tools or ["orient"]

    def _suggest_tier2_tools(self, query: str) -> list[str]:
        """Map query to search/retrieval tools."""
        tools = []
        if any(w in query for w in ["said", "say", "spoken", "transcript", "caption"]):
            tools.append("search_transcript")
        if any(w in query for w in ["(a)", "(b)", "(c)", "(d)", "which of", "choose"]):
            tools.extend(["search_video", "discriminative_vqa"])
        else:
            tools.append("search_all")
        return tools

    def _infer_output_format(self, query: str) -> str:
        """Infer what kind of answer the user wants."""
        if any(w in query for w in ["when", "timestamp", "time", "at what point"]):
            return "timestamp"
        if any(w in query for w in ["(a)", "(b)", "(c)", "(d)", "which of", "choose", "select"]):
            return "multiple_choice"
        if any(w in query for w in ["what action", "what sport", "classify", "what is"]):
            return "label"
        return "text"
