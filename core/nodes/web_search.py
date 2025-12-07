"""Web Search Agent - Performs internet search when course content not found."""

import logging
from typing import Dict, Any
from search.internet_search import InternetSearchAgent

logger = logging.getLogger(__name__)


def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for web search.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    agent = InternetSearchAgent()
    
    query = state.get("refined_query", state["query"])
    
    # Detect if query needs current information
    query_lower = query.lower()
    needs_current_info = any(keyword in query_lower for keyword in [
        "latest", "current", "recent", "new", "updated", "now", "today", "2024", "2025"
    ])
    
    # Get more results for current info queries
    num_results = 10 if needs_current_info else 5
    
    # Perform web search
    result = agent.search(
        query=query,
        course_name=state["course_name"],
        num_results=num_results
    )
    
    # Check if search was successful
    search_results = result.get("results", "")
    search_citations = result.get("citations", [])
    
    # Check if we got an error message instead of actual results
    if "not available" in search_results.lower() or "error" in search_results.lower():
        logger.error(f"Web search failed: {search_results}")
        # Still store it, but log the issue
        state["web_search_results"] = search_results
        state["web_search_citations"] = []
    else:
        state["web_search_results"] = search_results
        state["web_search_citations"] = search_citations
        logger.info(f"Web search completed successfully. Found {len(search_citations)} sources.")
        logger.info(f"Search results preview: {search_results[:200]}...")
    
    state["current_node"] = "web_search"
    state["should_continue"] = True
    state["next_node"] = "personalization"
    
    return state

