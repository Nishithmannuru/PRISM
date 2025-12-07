"""LangGraph flow definition for agentic RAG system."""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from core.state import AgentState
from core.nodes.query_refinement import query_refinement_node
from core.nodes.relevance import relevance_node
from core.nodes.course_rag import course_rag_node
from core.nodes.web_search import web_search_node
from core.nodes.personalization import personalization_node

logger = logging.getLogger(__name__)


def route_after_query_refinement(state: AgentState) -> Literal["relevance", "end"]:
    """Route after query refinement - check if vague."""
    if state.get("is_vague", False):
        # Query is vague, need to ask follow-up questions
        return "end"
    else:
        # Query is clear, proceed to relevance
        return "relevance"


def route_after_relevance(state: AgentState) -> Literal["course_rag", "end"]:
    """Route after relevance check."""
    if state.get("is_relevant", False):
        return "course_rag"
    else:
        return "end"


def route_after_course_rag(state: AgentState) -> Literal["personalization", "web_search"]:
    """Route after course RAG - check if content found."""
    if state.get("course_content_found", False):
        # Content found, skip web search, go to personalization
        return "personalization"
    else:
        # Content not found, need web search
        return "web_search"


def create_agent_graph():
    """Create and compile the LangGraph agent flow."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_refinement", query_refinement_node)
    workflow.add_node("relevance", relevance_node)
    workflow.add_node("course_rag", course_rag_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("personalization", personalization_node)
    
    # Set entry point
    workflow.set_entry_point("query_refinement")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "query_refinement",
        route_after_query_refinement,
        {
            "relevance": "relevance",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "relevance",
        route_after_relevance,
        {
            "course_rag": "course_rag",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "course_rag",
        route_after_course_rag,
        {
            "personalization": "personalization",
            "web_search": "web_search"
        }
    )
    
    # Web search always goes to personalization
    workflow.add_edge("web_search", "personalization")
    
    # Personalization is the end
    workflow.add_edge("personalization", END)
    
    # Compile with memory
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("LangGraph agent flow created successfully")
    
    return app

