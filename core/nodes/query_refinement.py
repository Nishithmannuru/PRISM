"""Query Refinement Agent - Detects vague queries and asks follow-up questions."""

import json
import logging
from typing import Dict, Any
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryRefinementAgent:
    """Agent that detects vague queries and asks clarifying questions."""
    
    def __init__(self):
        """Initialize the query refinement agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Load prompts
        config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def check_vagueness(
        self,
        query: str,
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        Check if a query is vague and needs clarification.
        
        Args:
            query: User's question
            conversation_history: Previous conversation context
            
        Returns:
            Dictionary with is_vague flag and follow-up questions
        """
        try:
            prompt_config = self.config.get('query_refinement', {})
            vague_detection_prompt = prompt_config.get('vague_detection', '')
            
            system_prompt = """You are a query refinement agent. Analyze if a question is vague or needs clarification.

A vague question is:
- Too broad or general (e.g., "tell me about the course" without specifics) AND has no context in conversation history
- Ambiguous (could mean multiple things) AND the conversation history doesn't clarify it
- Missing important context that makes it impossible to answer EVEN AFTER checking conversation history
- Unclear intent AND conversation history doesn't help

NOT vague (these should pass through):
- Simple greetings (hello, hi, hey, greetings) - these are clear social interactions
- Direct questions (what is X, how does Y work, explain Z, etc.)
- Specific queries about topics, concepts, or course material
- Questions with clear intent even if brief
- Questions that use pronouns/references (like "the paper", "it", "they") IF the conversation history provides the referent
- Questions that can be answered using conversation context (e.g., if history mentions "NeuroQuest paper", then "who are the authors of the paper?" is NOT vague)

CRITICAL: 
- ALWAYS check the conversation history FIRST before marking a question as vague
- If the conversation history provides context that clarifies pronouns/references, the question is NOT vague
- Be lenient - only mark as vague if the question is truly unanswerable even with conversation context

Respond with valid JSON only: {"is_vague": true/false, "follow_up_questions": ["question1", "question2"]}"""
            
            user_prompt = f"""Conversation History:
{conversation_history if conversation_history else "No previous conversation"}

Current Question: "{query}"

STEP 1: Check the conversation history above. Look for:
- Any mentions of papers, documents, topics, or entities that could be referenced
- Previous questions and answers that provide context
- Any specific names or terms that pronouns/references might refer to

STEP 2: Analyze if this question is vague:
- If it's a simple greeting (hello, hi, hey), it is NOT vague - set is_vague to false.
- If it's a direct question (what is X, how does Y, explain Z), it is NOT vague.
- If the question uses pronouns or references (like "the paper", "it", "they", "this", "the authors"):
  * FIRST check if the conversation history provides a clear referent
  * If history mentions "NeuroQuest paper" and question is "who are the authors of the paper?", it is NOT vague
  * If history mentions a topic and question references "it", it is NOT vague
  * Only mark as vague if NO referent exists in the conversation history
- If the conversation history provides ANY context that makes the question answerable, it is NOT vague
- Only mark as vague if the question is truly ambiguous AND the conversation history doesn't help

EXAMPLES:
- History: "What is NeuroQuest?" Answer: "NeuroQuest is a paper about..." Question: "Who are the authors of the paper?" → NOT vague (paper = NeuroQuest)
- History: "Tell me about the agents" Answer: "There are 3 agents..." Question: "What are they?" → NOT vague (they = agents)
- History: None, Question: "Who are the authors of the paper?" → VAGUE (no referent)

If it is vague, provide ONLY ONE follow-up question at a time that will help clarify the query. Ask the most important question first. If not vague, set follow_up_questions to an empty array."""
            
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
                "is_vague": result.get("is_vague", False),
                "follow_up_questions": result.get("follow_up_questions", [])
            }
            
        except Exception as e:
            logger.error(f"Error in query refinement: {e}")
            # Default to not vague if error
            return {
                "is_vague": False,
                "follow_up_questions": []
            }
    
    def refine_query(
        self,
        query: str,
        follow_up_answer: str = "",
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        Refine a query based on follow-up answer and check if it's now clear.
        
        Args:
            query: Original query
            follow_up_answer: User's answer to follow-up question
            conversation_history: Previous conversation context
            
        Returns:
            Dictionary with refined_query and is_clear flag
        """
        if not follow_up_answer:
            return {"refined_query": query, "is_clear": True}
        
        try:
            # Combine original query with follow-up answer
            combined_query = f"{query} {follow_up_answer}"
            
            # Check if the combined query is now clear
            vagueness_check = self.check_vagueness(
                query=combined_query,
                conversation_history=conversation_history
            )
            
            # If still vague, return the combined query but mark as not clear
            if vagueness_check["is_vague"]:
                return {
                    "refined_query": combined_query,
                    "is_clear": False,
                    "follow_up_question": vagueness_check["follow_up_questions"][0] if vagueness_check["follow_up_questions"] else None
                }
            
            # Query is now clear, refine it properly
            system_prompt = """You are a query refinement agent. Combine the original question with the follow-up answer to create a more specific, clear question."""
            
            user_prompt = f"""Original Question: {query}
Follow-up Answer: {follow_up_answer}

Create a refined, more specific question that incorporates both pieces of information. Make it clear and specific."""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            refined = response.choices[0].message.content.strip()
            
            return {
                "refined_query": refined,
                "is_clear": True,
                "follow_up_question": None
            }
            
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return {"refined_query": query, "is_clear": True}


def query_refinement_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for query refinement.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    agent = QueryRefinementAgent()
    
    # Get conversation history from messages in state
    # Include all previous messages for context
    messages = state.get("messages", [])
    current_query = state.get("query", "")
    
    # Format conversation history for the LLM
    # Include last 15 messages for better context (to catch references)
    recent_messages = messages[-15:] if len(messages) > 15 else messages
    
    conversation_history_parts = []
    for msg in recent_messages[:-1]:  # Exclude the current query message
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role = "User" if msg.type == "human" else "Assistant"
            content = str(msg.content)
            # Include full content for better context (limit to 500 chars per message to avoid token limits)
            content_preview = content[:500] + "..." if len(content) > 500 else content
            conversation_history_parts.append(f"{role}: {content_preview}")
    
    conversation_history = "\n".join(conversation_history_parts) if conversation_history_parts else "No previous conversation"
    
    logger.info(f"Query refinement - Conversation history: {len(recent_messages)-1} previous messages, {len(conversation_history)} chars")
    if conversation_history and len(conversation_history) > 0:
        logger.debug(f"Conversation history preview: {conversation_history[:300]}...")
    
    # Check if query is vague
    result = agent.check_vagueness(
        query=current_query,
        conversation_history=conversation_history
    )
    
    # Fallback: If query uses common reference words and we have conversation history, be lenient
    query_lower = current_query.lower()
    reference_words = ["the paper", "the document", "it", "they", "this", "that", "these", "those", "the authors", "the agents", "the figures", "the tables"]
    uses_reference = any(word in query_lower for word in reference_words)
    
    if result["is_vague"] and uses_reference and conversation_history and conversation_history != "No previous conversation":
        # Query uses references and we have history - check if history might provide context
        # Look for common entities in history that could be referenced
        history_lower = conversation_history.lower()
        has_paper_mention = any(term in history_lower for term in ["paper", "document", "neuroquest", "article"])
        has_author_mention = "author" in history_lower or "written by" in history_lower
        
        if has_paper_mention or has_author_mention:
            logger.info(f"Query uses reference words but history contains relevant context. Overriding vague detection.")
            result["is_vague"] = False
            result["follow_up_questions"] = []
    
    state["is_vague"] = result["is_vague"]
    # Store only the first follow-up question (one at a time)
    follow_up_questions = result.get("follow_up_questions", [])
    state["follow_up_questions"] = [follow_up_questions[0]] if follow_up_questions else []
    state["current_node"] = "query_refinement"
    
    if state["is_vague"]:
        # Need to ask ONE follow-up question
        state["should_continue"] = False
        state["next_node"] = None
        logger.info(f"Query is vague. Asking follow-up question: {state['follow_up_questions'][0] if state['follow_up_questions'] else 'None'}")
    else:
        # Query is clear, proceed to relevance check
        state["refined_query"] = state["query"]
        state["should_continue"] = True
        state["next_node"] = "relevance"
        logger.info("Query is clear. Proceeding to relevance check.")
    
    return state

