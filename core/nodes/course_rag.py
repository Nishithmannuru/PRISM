"""Course RAG Agent - Retrieves and checks course content."""

import logging
from typing import Dict, Any
from retrieval.retriever import CourseRetriever

logger = logging.getLogger(__name__)


class CourseRAGAgent:
    """Agent that retrieves course content and checks if it answers the question."""
    
    def __init__(self):
        """Initialize the course RAG agent."""
        self.retriever = CourseRetriever()
    
    def retrieve_and_check(
        self,
        query: str,
        course_name: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve course content and check if it answers the question.
        
        Args:
            query: User's question
            course_name: Name of the course
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with content, citations, and found flag
        """
        try:
            logger.info(f"Retrieving content for query: '{query}' in course: '{course_name}'")
            
            # Retrieve relevant chunks - try with original query first
            retrieved_chunks = self.retriever.retrieve(
                query=query,
                course_name=course_name,
                top_k=top_k
            )
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from vector store for query: '{query}'")
            
            # For queries about lists, counts, or "all" items, try additional queries to get comprehensive results
            query_lower = query.lower()
            needs_comprehensive = any(keyword in query_lower for keyword in [
                "all", "different", "various", "list", "what are", "how many", "name", "types", "kinds"
            ])
            
            if needs_comprehensive and retrieved_chunks:
                logger.info("Query requires comprehensive results. Retrieving additional chunks...")
                # Extract the main topic/keyword from the query (generic approach)
                # Remove common question words and get the core topic
                question_words = ["what", "are", "the", "different", "various", "all", "how", "many", "list", "name"]
                words = [w for w in query_lower.split() if w not in question_words and len(w) > 2]
                
                if words:
                    # Use the main topic word(s) for broader retrieval
                    main_topic = words[0]  # Primary keyword
                    additional_queries = [
                        main_topic,  # Singular/plural variations
                        main_topic + "s" if not main_topic.endswith("s") else main_topic[:-1],  # Pluralize or singularize
                    ]
                    
                    # If there's a second significant word, combine it
                    if len(words) > 1:
                        combined = f"{words[0]} {words[1]}"
                        additional_queries.append(combined)
                    
                    # Retrieve additional chunks with broader queries
                    all_chunks = list(retrieved_chunks)  # Start with existing chunks
                    seen_ids = {chunk.get('content', '')[:50] for chunk in all_chunks}  # Track seen chunks
                    
                    for alt_query in additional_queries:
                        if alt_query == query_lower:  # Skip if same as original
                            continue
                        alt_chunks = self.retriever.retrieve(
                            query=alt_query,
                            course_name=course_name,
                            top_k=top_k
                        )
                        # Add unique chunks
                        for chunk in alt_chunks:
                            chunk_id = chunk.get('content', '')[:50]
                            if chunk_id not in seen_ids:
                                all_chunks.append(chunk)
                                seen_ids.add(chunk_id)
                    
                    if len(all_chunks) > len(retrieved_chunks):
                        logger.info(f"Expanded from {len(retrieved_chunks)} to {len(all_chunks)} chunks for comprehensive query")
                        retrieved_chunks = all_chunks[:top_k * 2]  # Allow up to 2x top_k for comprehensive queries
            
            # If no results, try with a simplified/expanded query
            if not retrieved_chunks:
                logger.info(f"No chunks found with original query. Trying alternative query formulations...")
                # Try alternative query formulations
                alternative_queries = [
                    query.lower(),  # Lowercase
                    query + " " + course_name,  # Add course name
                ]
                
                for alt_query in alternative_queries:
                    logger.info(f"Trying alternative query: '{alt_query}'")
                    alt_chunks = self.retriever.retrieve(
                        query=alt_query,
                        course_name=course_name,
                        top_k=top_k
                    )
                    if alt_chunks:
                        retrieved_chunks = alt_chunks
                        logger.info(f"Found {len(retrieved_chunks)} chunks with alternative query")
                        break
            
            if not retrieved_chunks:
                logger.warning(f"No chunks retrieved for query: '{query}' in course: '{course_name}' after trying alternatives")
                # Even if no chunks found, check if query asks for current/updated information
                # Such queries should go to web search even if course mentions the topic
                query_lower = query.lower()
                needs_current_info = any(keyword in query_lower for keyword in [
                    "latest", "current", "recent", "new", "updated", "now", "today", "2024", "2025"
                ])
                if needs_current_info:
                    logger.info(f"Query asks for current/updated information. Marking as not found to trigger web search.")
                return {
                    "found": False,
                    "context": None,
                    "citations": []
                }
            
            # Log scores for debugging
            scores = [chunk.get("score", 0) for chunk in retrieved_chunks]
            logger.info(f"Retrieval scores: {scores}")
            
            # Format context
            context = self.retriever.format_context(retrieved_chunks)
            citations = self.retriever.get_citations(retrieved_chunks)
            
            logger.info(f"Formatted context length: {len(context)} characters")
            logger.info(f"Found {len(citations)} citations")
            
            # Check if content is relevant - if we got results, they're relevant
            # However, if query asks for current/updated info, we should still check web search
            query_lower = query.lower()
            needs_current_info = any(keyword in query_lower for keyword in [
                "latest", "current", "recent", "new", "updated", "now", "today", "2024", "2025"
            ])
            
            # If query asks for current info and we have chunks, check if chunks actually answer the question
            # For "latest" type questions, course materials might be outdated, so prefer web search
            if needs_current_info and retrieved_chunks:
                logger.info(f"Query asks for current/updated information. Even though chunks found, will check web search for latest info.")
                # Mark as not found to trigger web search for current information
                found = False
            else:
                found = len(retrieved_chunks) > 0
            
            if found:
                logger.info(f"Content found! Using {len(retrieved_chunks)} chunks with context length {len(context)} chars")
            else:
                if needs_current_info:
                    logger.info(f"Query asks for current info - routing to web search for latest information")
                else:
                    logger.warning(f"No content found for query: '{query}'")
            
            return {
                "found": found,
                "context": context,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks
            }
            
        except Exception as e:
            logger.error(f"Error in course RAG: {e}")
            return {
                "found": False,
                "context": None,
                "citations": []
            }


def course_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for course RAG.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    agent = CourseRAGAgent()
    
    query = state.get("refined_query", state["query"])
    
    # Retrieve and check course content - use more chunks for better context
    result = agent.retrieve_and_check(
        query=query,
        course_name=state["course_name"],
        top_k=10  # Increased from 5 to get more context
    )
    
    state["course_content_found"] = result["found"]
    state["course_context"] = result["context"]
    state["course_citations"] = result["citations"]
    state["current_node"] = "course_rag"
    
    if state["course_content_found"]:
        # Content found in course, proceed to personalization
        state["should_continue"] = True
        state["next_node"] = "personalization"
        logger.info(f"Course content found ({len(result.get('retrieved_chunks', []))} chunks). Proceeding to personalization.")
        logger.info(f"Context preview: {state.get('course_context', '')[:200]}...")
    else:
        # Content not found, need web search
        state["should_continue"] = True
        state["next_node"] = "web_search"
        logger.warning(f"Course content NOT found for query: '{query}' in course: '{state['course_name']}'. Proceeding to web search.")
    
    return state

