"""State management for LangGraph agentic flow."""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class AgentState(TypedDict):
    """State structure for the agentic flow."""
    # Conversation history
    messages: List[BaseMessage]
    
    # Query information
    query: str
    refined_query: Optional[str]
    
    # Agent decisions
    is_vague: bool
    follow_up_questions: List[str]
    is_relevant: bool
    relevance_reason: Optional[str]
    
    # Course content
    course_content_found: bool
    course_context: Optional[str]
    course_citations: List[Dict[str, Any]]
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    
    # Web search
    web_search_results: Optional[str]
    web_search_citations: List[Dict[str, Any]]
    
    # User context
    user_context: Dict[str, Any]
    course_name: str
    
    # Flow control
    current_node: str
    next_node: Optional[str]
    should_continue: bool
    
    # Final response
    final_response: Optional[str]
    response_citations: List[Dict[str, Any]]


def create_initial_state(
    query: str,
    course_name: str,
    user_context: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> AgentState:
    """Create initial state for the agentic flow."""
    # Convert conversation history to LangChain messages
    messages = []
    if conversation_history:
        for msg in conversation_history:
            content = msg.get("content", "")
            # Skip messages with None or empty content
            if not content:
                continue
            if msg["role"] == "user":
                messages.append(HumanMessage(content=str(content)))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=str(content)))
    
    # Add current query (ensure it's a string)
    if query:
        messages.append(HumanMessage(content=str(query)))
    
    return AgentState(
        messages=messages,
        query=query,
        refined_query=None,
        is_vague=False,
        follow_up_questions=[],
        is_relevant=False,
        relevance_reason=None,
        course_content_found=False,
        course_context=None,
        course_citations=[],
        retrieved_chunks=None,
        web_search_results=None,
        web_search_citations=[],
        user_context=user_context,
        course_name=course_name,
        current_node="start",
        next_node=None,
        should_continue=True,
        final_response=None,
        response_citations=[]
    )

