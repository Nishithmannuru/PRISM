"""Relevance Agent - Determines if a question is relevant to the course."""

import json
import logging
from typing import Dict, Any
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class RelevanceAgent:
    """Agent that determines if a question is relevant to the course."""
    
    def __init__(self):
        """Initialize the relevance agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Load prompts and course descriptions
        config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def check_relevance(
        self,
        query: str,
        course_name: str,
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        Check if a query is relevant to the course.
        
        Args:
            query: User's question
            course_name: Name of the course
            conversation_history: Previous conversation context
            
        Returns:
            Dictionary with relevance flag and reason
        """
        try:
            # Get course description
            course_descriptions = self.config.get('course_descriptions', {})
            course_description = course_descriptions.get(
                course_name,
                f"This course covers topics related to {course_name}."
            )
            
            # Get relevance prompt
            relevance_config = self.config.get('relevance_prompts', {})
            system_prompt = relevance_config.get(
                'system',
                """You are a relevance classifier for course questions.
Determine if a student's question is relevant to the course based on:
1. The course description
2. The course name and context
3. Whether the question relates to course topics, concepts, materials, or content

IMPORTANT: Be VERY lenient. If a question could potentially relate to the course (even indirectly or tangentially), mark it as relevant.
Only mark as NOT relevant if the question is clearly about completely unrelated topics (e.g., weather, sports, cooking, completely unrelated subjects).

Questions about:
- Course content, materials, figures, tables, concepts, architecture, agents, methods, etc. = RELEVANT
- General topics that might be in the course = RELEVANT
- Current/updated information about topics mentioned in the course (e.g., "latest version", "current state", "recent updates") = RELEVANT
- Technologies, tools, or concepts mentioned in the course = RELEVANT
- Questions that build on course topics even if asking for current/outside information = RELEVANT
- Completely unrelated topics (weather, cooking, etc.) = NOT RELEVANT

Respond with valid JSON only: {"relevant": true/false, "reason": "brief explanation"}"""
            )
            
            user_prompt = f"""Course: {course_name}
Course Description: {course_description}

Conversation History:
{conversation_history if conversation_history else "No previous conversation"}

Student Question: {query}

Is this question relevant to the course? Be VERY lenient:
- If it relates to course topics, concepts, materials, figures, tables, architecture, or content = RELEVANT
- If it asks about current/updated information related to topics in the course (e.g., "latest", "current", "recent") = RELEVANT
- If it's about technologies, tools, or concepts mentioned in the course = RELEVANT
- Only mark as NOT relevant if it's clearly about completely unrelated topics (weather, sports, cooking, etc.)

Remember: Questions asking for current/updated information about course-related topics are still RELEVANT."""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "relevant": result.get("relevant", False),
                "reason": result.get("reason", "")
            }
            
        except Exception as e:
            logger.error(f"Error in relevance check: {e}")
            # Default to relevant if error (to avoid blocking legitimate questions)
            return {
                "relevant": True,
                "reason": "Error in relevance check, defaulting to relevant"
            }


def relevance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for relevance checking.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    agent = RelevanceAgent()
    
    # Get conversation history from messages in state
    messages = state.get("messages", [])
    # Include last 10 messages for context
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    
    conversation_history_parts = []
    for msg in recent_messages[:-1]:  # Exclude current query
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role = "User" if msg.type == "human" else "Assistant"
            conversation_history_parts.append(f"{role}: {msg.content}")
    
    conversation_history = "\n".join(conversation_history_parts) if conversation_history_parts else "No previous conversation"
    logger.info(f"Relevance check - Using {len(recent_messages)-1} previous messages for context")
    
    # Check relevance
    result = agent.check_relevance(
        query=state.get("refined_query", state["query"]),
        course_name=state["course_name"],
        conversation_history=conversation_history
    )
    
    state["is_relevant"] = result["relevant"]
    state["relevance_reason"] = result["reason"]
    state["current_node"] = "relevance"
    
    if not state["is_relevant"]:
        # Question is not relevant, stop here
        state["should_continue"] = False
        state["next_node"] = None
        state["final_response"] = (
            f"I'm sorry, but your question doesn't seem to be related to {state['course_name']}. "
            f"{result['reason']}\n\n"
            "Please ask a question that is relevant to the course material."
        )
        logger.info(f"Question not relevant: {result['reason']}")
    else:
        # Question is relevant, proceed to course RAG
        state["should_continue"] = True
        state["next_node"] = "course_rag"
        logger.info("Question is relevant. Proceeding to course RAG.")
    
    return state

