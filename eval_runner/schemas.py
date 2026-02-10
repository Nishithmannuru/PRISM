"""Standard Run Trace schema returned by PRISM wrapper for each query."""

from typing import List, Dict, Any, Optional

# RunTrace: one per (eval_id, variant, profile) run
RunTrace = Dict[str, Any]

# Expected keys (all optional for robustness):
# - final_answer_text: str
# - route_taken: "clarify" | "refuse" | "web" | "course"
# - tools_used: List[str]
# - retrieval_context: List[Dict]  # chunk_id, doc_id, location, text
# - web_context: List[Dict]         # url, domain, snippet, text
# - citations: List[Dict]
# - internal_eval_scores: Optional[Dict[str, float]]
# - latency_ms: Optional[int]
# - token_usage: Optional[Dict]
# - error: Optional[str]
