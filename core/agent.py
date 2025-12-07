"""Main agent orchestrator for LangGraph-based agentic RAG system."""

import logging
from typing import Dict, Any, Optional, List
from core.state import create_initial_state, AgentState
from core.graph import create_agent_graph

logger = logging.getLogger(__name__)


class PRISMAgent:
    """Main orchestrator for the PRISM agentic RAG system."""
    
    def __init__(self):
        """Initialize the PRISM agent."""
        self.graph = None
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the LangGraph."""
        try:
            self.graph = create_agent_graph()
            logger.info("PRISM agent graph initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent graph: {e}")
            raise
    
    def process_query(
        self,
        query: str,
        course_name: str,
        user_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a user query through the agentic flow.
        
        Args:
            query: User's question
            course_name: Name of the course
            user_context: Student information (degree, major, etc.)
            conversation_history: Previous conversation messages (optional, LangGraph handles this via checkpointing)
            thread_id: Thread ID for conversation memory
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            from langchain_core.messages import HumanMessage
            
            # Create config for thread (memory)
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            
            # Get previous state from checkpoint (LangGraph handles this automatically)
            # If this is the first message in the thread, state will be None
            previous_state = None
            try:
                previous_state = self.graph.get_state(config)
                if previous_state and previous_state.values:
                    existing_messages = previous_state.values.get("messages", [])
                    logger.info(f"Retrieved previous state for thread {thread_id}. Found {len(existing_messages)} existing messages in checkpoint")
                    if existing_messages:
                        # Log a preview of recent messages
                        recent_preview = existing_messages[-3:] if len(existing_messages) > 3 else existing_messages
                        for msg in recent_preview:
                            if hasattr(msg, 'content'):
                                logger.debug(f"  Recent message: {msg.type}: {str(msg.content)[:100]}...")
                else:
                    logger.info(f"No previous state values found for thread {thread_id}")
            except Exception as e:
                logger.info(f"No previous state found for thread {thread_id} (first message or error): {e}")
                previous_state = None
            
            # Create initial state - merge with previous state if it exists
            if previous_state and previous_state.values:
                # Get existing messages from checkpoint
                existing_messages = previous_state.values.get("messages", [])
                logger.info(f"Found {len(existing_messages)} existing messages in checkpoint")
                
                # Create new state with existing messages + new query
                initial_state = {
                    "messages": existing_messages + [HumanMessage(content=query)],
                    "query": query,
                    "refined_query": None,
                    "is_vague": False,
                    "follow_up_questions": [],
                    "is_relevant": False,
                    "relevance_reason": None,
                    "course_content_found": False,
                    "course_context": None,
                    "course_citations": [],
                    "web_search_results": None,
                    "web_search_citations": [],
                    "user_context": user_context,
                    "course_name": course_name,
                    "current_node": "start",
                    "next_node": None,
                    "should_continue": True,
                    "final_response": None,
                    "response_citations": []
                }
            else:
                # First message in thread - create fresh state
                initial_state = create_initial_state(
                    query=query,
                    course_name=course_name,
                    user_context=user_context,
                    conversation_history=conversation_history  # Use provided history for first message
                )
                logger.info("Created new state (first message in thread)")
            
            # Run the graph - use invoke for proper checkpointing
            # LangGraph will automatically save state to checkpoint after invoke
            final_state = self.graph.invoke(initial_state, config=config)
            
            logger.info(f"Graph execution completed. Final state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
            
            # Extract final state - invoke returns the final state directly
            last_node_state = final_state
            
            # Check if we need follow-up questions
            if last_node_state.get("is_vague") and last_node_state.get("follow_up_questions"):
                return {
                    "response": None,
                    "needs_follow_up": True,
                    "follow_up_questions": last_node_state["follow_up_questions"],
                    "is_relevant": None,
                    "citations": []
                }
            
            # Check if question is not relevant
            if not last_node_state.get("is_relevant", True):
                return {
                    "response": last_node_state.get("final_response", "Question not relevant to course."),
                    "needs_follow_up": False,
                    "follow_up_questions": [],
                    "is_relevant": False,
                    "citations": []
                }
            
            # Return final response
            final_response = last_node_state.get("final_response")
            
            # Ensure we have a response
            if not final_response:
                # Try to construct a response from available information
                if last_node_state.get("course_context"):
                    final_response = f"Based on the course materials, I found some relevant information. However, I couldn't generate a complete response. Please try rephrasing your question."
                elif last_node_state.get("web_search_results"):
                    final_response = f"I searched the internet but couldn't generate a complete response. Please try rephrasing your question."
                else:
                    final_response = "I couldn't generate a response. Please try rephrasing your question or ensure your question is relevant to the course."
            
            return {
                "response": final_response,
                "needs_follow_up": False,
                "follow_up_questions": [],
                "is_relevant": True,
                "citations": last_node_state.get("response_citations", []),
                "used_web_search": not last_node_state.get("course_content_found", False)
            }
            
            return {
                "response": "Error: No final state generated.",
                "needs_follow_up": False,
                "follow_up_questions": [],
                "is_relevant": None,
                "citations": []
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error while processing your question. Please try again. Error: {str(e)}",
                "needs_follow_up": False,
                "follow_up_questions": [],
                "is_relevant": None,
                "citations": []
            }
    
    def refine_query_with_follow_up(
        self,
        original_query: str,
        follow_up_answer: str,
        course_name: str,
        user_context: Dict[str, Any],
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Refine a query using follow-up answer and check if it's clear.
        If still vague, ask another follow-up question.
        
        Args:
            original_query: Original vague query
            follow_up_answer: User's answer to follow-up question
            course_name: Name of the course
            user_context: Student information
            thread_id: Thread ID for conversation memory
            
        Returns:
            Dictionary with response and metadata (may include needs_follow_up if still vague)
        """
        from core.nodes.query_refinement import QueryRefinementAgent
        
        agent = QueryRefinementAgent()
        
        # Get conversation history for context
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            previous_state = self.graph.get_state(config)
            messages = previous_state.values.get("messages", []) if previous_state.values else []
            
            # Format conversation history
            conversation_history_parts = []
            for msg in messages[-10:]:  # Last 10 messages
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = "User" if msg.type == "human" else "Assistant"
                    content = str(msg.content)[:300]  # Limit length
                    conversation_history_parts.append(f"{role}: {content}")
            
            conversation_history = "\n".join(conversation_history_parts) if conversation_history_parts else ""
        except Exception as e:
            logger.info(f"Could not retrieve conversation history: {e}")
            conversation_history = ""
        
        # Refine query and check if it's clear
        refinement_result = agent.refine_query(
            query=original_query,
            follow_up_answer=follow_up_answer,
            conversation_history=conversation_history
        )
        
        # If still vague, return with follow-up question
        if not refinement_result.get("is_clear", True):
            follow_up_question = refinement_result.get("follow_up_question")
            return {
                "response": None,
                "needs_follow_up": True,
                "follow_up_questions": [follow_up_question] if follow_up_question else [],
                "is_relevant": None,
                "citations": []
            }
        
        # Query is now clear, process it
        refined_query = refinement_result.get("refined_query", f"{original_query} {follow_up_answer}")
        return self.process_query(
            query=refined_query,
            course_name=course_name,
            user_context=user_context,
            conversation_history=None,  # Will use thread memory
            thread_id=thread_id
        )
