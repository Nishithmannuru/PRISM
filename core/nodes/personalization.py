"""Personalization Agent - Tailors response to student's background."""

import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


class PersonalizationAgent:
    """Agent that personalizes responses based on student background."""
    
    def __init__(self):
        """Initialize the personalization agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Load prompts
        config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def personalize_response(
        self,
        query: str,
        context: str,
        user_context: Dict[str, Any],
        course_name: str,
        citations: list,
        is_from_web: bool = False,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized response based on student background.
        
        Args:
            query: User's question
            context: Retrieved context (course or web search)
            user_context: Student information (degree, major)
            course_name: Name of the course
            citations: List of citations
            is_from_web: Whether context is from web search
            
        Returns:
            Dictionary with personalized response and citations
        """
        try:
            degree = user_context.get("degree", "N/A")
            major = user_context.get("major", "N/A")
            
            # Determine complexity level based on degree
            if "PhD" in degree or "Doctor" in degree:
                complexity = "advanced"
                explanation_style = "detailed and technical"
            elif "Master" in degree:
                complexity = "intermediate"
                explanation_style = "balanced with some technical detail"
            else:
                complexity = "introductory"
                explanation_style = "simple and accessible"
            
            # Adapt to major
            major_adaptation = ""
            if major.lower() not in ["computer science", "cs", "engineering"]:
                major_adaptation = (
                    f"Since you're a {major} student, I'll explain this in terms you'll find familiar. "
                    "I'll use simpler language and provide examples that relate to your field of study."
                )
            
            system_prompt = f"""You are an expert teaching assistant for {course_name}.
You help students understand course material by providing clear, personalized answers.

Student Background:
- Degree Level: {degree} ({complexity} level)
- Major: {major}
{major_adaptation}

Adapt your explanation to be {explanation_style}. Use examples and analogies that a {major} student would understand.

CRITICAL CITATION FORMAT:
- Use INLINE citations within your response text, NOT at the end
- Format: (Document_Name, Page X) or (Source_Name, URL)
- Place citations immediately after the information you're citing
- Example: "The authors are John Doe and Jane Smith (NeuroQuest_Paper, Page 2)."
- Do NOT create a separate citations section at the end
- Integrate citations naturally into your response"""
            
            context_source = "Internet search results" if is_from_web else "Course materials"
            
            # Add special instruction for current info queries from web search
            current_info_instruction = ""
            if is_from_web:
                query_lower = query.lower()
                needs_current_info = any(keyword in query_lower for keyword in [
                    "latest", "current", "recent", "new", "updated", "now", "today", "2024", "2025"
                ])
                if needs_current_info:
                    from datetime import datetime
                    current_date = datetime.now().strftime("%B %d, %Y")
                    current_year = datetime.now().year
                    current_info_instruction = f"""
CRITICAL: This question asks for CURRENT/LATEST information as of {current_date}. 
- TODAY'S DATE IS: {current_date} ({current_year})
- You MUST prioritize the MOST RECENT information from the search results
- Look for dates, years, version numbers, and timestamps in the search results
- If multiple results mention different dates/years, use the one with the LATEST date/year
- AI-generated answers from Tavily are typically the most current - prioritize those
- If search results mention "latest", "newest", "recent", or specific dates, use that information
- Do NOT use information that is clearly outdated (e.g., if it says "as of 2023" and today is {current_year}, look for {current_year} information)
- If the search results contain conflicting dates, use the most recent one
- Extract and mention the date/year of the information you're using in your response"""
            
            # Handle case where context might indicate no results or errors
            context_lower = context.lower() if context else ""
            if "couldn't find" in context_lower or "no specific" in context_lower or "not available" in context_lower or "error" in context_lower:
                user_prompt = f"""Student Question: {query}

Student Background: {degree} student in {major}

{context}

Please provide a helpful, personalized response that:
1. Acknowledges that specific information wasn't found
2. Provides a general answer appropriate for a {degree} student studying {major}
3. Suggests how they might find more information
4. Uses language and examples relevant to their background"""
            else:
                # Detect if query requires comprehensive extraction
                query_lower = query.lower()
                needs_all_items = any(keyword in query_lower for keyword in [
                    "all", "different", "various", "list", "what are", "how many", "name all", "types", "kinds"
                ])
                
                comprehensive_instruction = ""
                if needs_all_items:
                    # Extract the main topic from the query (generic approach)
                    question_words = ["what", "are", "the", "different", "various", "all", "how", "many", "list", "name", "in", "it"]
                    topic_words = [w for w in query_lower.split() if w not in question_words and len(w) > 2]
                    main_topic = topic_words[0] if topic_words else "items"
                    
                    comprehensive_instruction = f"""
CRITICAL: This question asks for ALL {main_topic.upper()}. You MUST:
- Extract and list EVERY SINGLE {main_topic} mentioned in the context by its exact name or identifier
- Do NOT use generic terms - you MUST list each {main_topic} by its SPECIFIC NAME/IDENTIFIER
- Scan the ENTIRE context word-by-word for any mention of {main_topic} names/identifiers
- Look for patterns like: "[Name] {main_topic}", "the [Name] {main_topic}", "{main_topic}: [Name]", or any capitalized {main_topic} names
- If the context mentions multiple {main_topic}s, you MUST list ALL of them
- For each {main_topic} found, provide: 1) The exact name/identifier, 2) What it is/does (if described)
- Do not say "not detailed" or "not mentioned" if {main_topic} names are in the context
- Review ALL sources carefully - {main_topic}s may be mentioned in different pages or sections
- Your response MUST start with a clear numbered or bulleted list format like:
  1. [{main_topic.capitalize()} Name/Identifier]: [Description]
  2. [{main_topic.capitalize()} Name/Identifier]: [Description]
  3. [{main_topic.capitalize()} Name/Identifier]: [Description]
- If you only find one {main_topic}, you MUST still search the entire context for others - do not stop after finding one"""
                
                user_prompt = f"""{context_source}:
{context}

Student Question: {query}

Student Background: {degree} student in {major}
{comprehensive_instruction}
{current_info_instruction}

Based on the {context_source.lower()} provided above, please provide a comprehensive, personalized answer to the student's question that:
1. Directly answers the question using information from the {context_source.lower()} - USE THE INFORMATION PROVIDED
2. Is appropriate for a {degree} student studying {major}
3. Uses language and examples relevant to their background
4. Explains concepts in a way they'll understand
5. If the question asks for specific information (like counts, lists, names, agents, figures, tables), extract and provide ALL of that information from the context - be thorough and complete
6. Review ALL sources provided to ensure you don't miss any information
7. Uses INLINE citations in the format (Document_Name, Page X) immediately after cited information - do NOT create a separate citations section

CRITICAL INSTRUCTIONS:
- The {context_source.lower()} above contains REAL information - USE IT to answer the question
- Do NOT say "I don't have access" or "I'm unable to access" - you HAVE the information in the context above
- If the context contains search results or answers, USE THEM - they are real and current
- Extract information directly from the context provided
- If the question asks for "latest" or "current" information, look for the MOST RECENT dates, years, or version numbers in the context
- Prioritize information with the latest dates/years mentioned in the search results
- If you see multiple dates or versions, use the one with the most recent date/year
- If asked for a list or "all" items, you MUST extract and list EVERY item mentioned across all sources
- Do not omit information that is present in the context
- If asked for a list of items (e.g., "what are the different X", "list all Y", "how many Z"), you MUST:
  * Identify the main topic (X, Y, or Z) from the question
  * Scan the ENTIRE context word-by-word for every mention of that topic
  * List each item by its exact name/identifier - do NOT use vague generic terms
  * Look for patterns like "[Name] [Topic]", "[Topic]: [Name]", "the [Name] [topic]", or any capitalized names
  * Continue searching the entire context even after finding one item - do not stop early
  * Provide a clear numbered or bulleted list format starting your response with ALL items found
  * Extract what you can find from the context - if information is present, use it completely"""
            
            response_settings = self.config.get('response_settings', {})
            temperature = response_settings.get('temperature', 0.7)
            max_tokens = response_settings.get('max_tokens', 2000)
            
            logger.info(f"Calling OpenAI API with model: {OPENAI_MODEL}, temperature: {temperature}, max_tokens: {max_tokens}")
            logger.debug(f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")
            
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                answer = response.choices[0].message.content
                logger.info(f"OpenAI API call successful. Response length: {len(answer) if answer else 0}")
            except Exception as api_error:
                logger.error(f"OpenAI API error: {api_error}", exc_info=True)
                raise  # Re-raise to be caught by outer exception handler
            
            # Ensure answer is not None
            if not answer:
                answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            # Filter citations to only include sources actually referenced in the response
            # Extract inline citations from the answer in format (Document_Name, Page X)
            import re
            # Match inline citation format: (Document_Name, Page X) or (Source_Name, URL)
            # Pattern matches: (text, Page number) or (text, URL)
            inline_citation_pattern = r'\(([^,)]+),\s*(?:Page\s+)?(\d+|[^)]+)\)'
            referenced_citations = []
            
            for match in re.finditer(inline_citation_pattern, answer, re.IGNORECASE):
                doc_name = match.group(1).strip()
                page_or_url = match.group(2).strip()
                
                # Check if it's a page number (digits) or URL
                if page_or_url.isdigit():
                    page_num = int(page_or_url)
                    referenced_citations.append({
                        "document": doc_name,
                        "page": page_num
                    })
                else:
                    # It's a URL or other format
                    referenced_citations.append({
                        "document": doc_name,
                        "url": page_or_url
                    })
            
            logger.info(f"Found {len(referenced_citations)} inline citations in response")
            if referenced_citations:
                logger.info(f"Inline citations found: {referenced_citations[:3]}...")  # Log first 3
            
            # If inline citations are found, filter to match them
            # Otherwise, use all citations (for web search or when no explicit citations)
            filtered_citations = citations
            if referenced_citations and not is_from_web and retrieved_chunks:
                logger.info(f"Filtering citations: {len(referenced_citations)} inline citations found, {len(retrieved_chunks)} chunks available, {len(citations)} total citations")
                
                # Create a set of (document, page) tuples from referenced citations for matching
                referenced_keys = set()
                for ref_citation in referenced_citations:
                    doc = ref_citation.get("document", "").strip()
                    page = ref_citation.get("page")
                    if doc and page is not None:
                        # Normalize document name (remove spaces, handle variations)
                        doc_normalized = doc.replace(" ", "_").replace("-", "_").lower()
                        referenced_keys.add((doc_normalized, page))
                
                # Match referenced citations against actual citations
                filtered_citations = []
                seen_citations = set()
                
                for citation in citations:
                    doc = citation.get("document", "").strip()
                    page = citation.get("page")
                    
                    if doc and page is not None:
                        # Normalize document name for comparison
                        doc_normalized = doc.replace(" ", "_").replace("-", "_").lower()
                        citation_key = (doc_normalized, page)
                        
                        # Check if this citation matches any referenced citation
                        # Use fuzzy matching to handle variations in document names
                        matches = False
                        for ref_key in referenced_keys:
                            ref_doc, ref_page = ref_key
                            # Exact match on page and document name contains the reference or vice versa
                            if ref_page == page and (ref_doc in doc_normalized or doc_normalized in ref_doc):
                                matches = True
                                break
                        
                        if matches:
                            citation_unique_key = (doc, page)
                            if citation_unique_key not in seen_citations:
                                filtered_citations.append(citation)
                                seen_citations.add(citation_unique_key)
                
                logger.info(f"Filtered citations: {len(referenced_citations)} inline citations found, {len(filtered_citations)} matching citations after filtering (from {len(citations)} total)")
                logger.info(f"Filtered citation details: {filtered_citations}")
                
                # If no citations matched, fall back to all citations
                if not filtered_citations:
                    logger.warning("No matching citations found for inline citations, using all citations")
                    filtered_citations = citations
                else:
                    logger.info(f"Successfully filtered to {len(filtered_citations)} citations from {len(citations)} total")
            else:
                # For web search or when no explicit sources, use all citations
                # But deduplicate by URL for web search
                if is_from_web and citations:
                    seen_urls = set()
                    unique_citations = []
                    for citation in citations:
                        url = citation.get('url', '')
                        if url and url not in seen_urls:
                            unique_citations.append(citation)
                            seen_urls.add(url)
                        elif not url:  # Include citations without URLs
                            unique_citations.append(citation)
                    filtered_citations = unique_citations
                    logger.info(f"Deduplicated web citations: {len(filtered_citations)} unique citations (from {len(citations)} total)")
                elif not is_from_web and not referenced_citations and citations and retrieved_chunks:
                    # No sources referenced in response - use only top citations by score
                    # Limit to top 3-5 citations to avoid showing too many
                    logger.info(f"No source references found in response. Limiting to top citations.")
                    # Get unique citations and limit to top 5
                    seen_citations = set()
                    unique_citations = []
                    for chunk in retrieved_chunks[:5]:  # Only use top 5 chunks
                        citation_key = (chunk.get('document_name'), chunk.get('page_number'))
                        if citation_key not in seen_citations:
                            unique_citations.append({
                                "document": chunk.get('document_name', 'Unknown'),
                                "page": chunk.get('page_number', 'Unknown')
                            })
                            seen_citations.add(citation_key)
                    filtered_citations = unique_citations
                    logger.info(f"Limited to top {len(filtered_citations)} citations (from {len(citations)} total) when no sources referenced")
            
            # No separate citations section - citations should be inline in the answer
            # The LLM should have already included inline citations in the format (Document_Name, Page X)
            final_response = answer
            
            return {
                "response": final_response,
                "citations": filtered_citations
            }
            
        except Exception as e:
            logger.error(f"Error in personalization: {e}", exc_info=True)
            # Provide more helpful error message
            error_msg = str(e)
            if "api" in error_msg.lower() or "key" in error_msg.lower():
                return {
                    "response": f"I encountered an API configuration error. Please check your API keys. Error: {error_msg[:100]}",
                    "citations": []
                }
            else:
                return {
                    "response": f"I encountered an error while generating a response: {error_msg[:200]}. Please try again or rephrase your question.",
                    "citations": []
                }


def personalization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for personalization.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final response
    """
    agent = PersonalizationAgent()
    
    query = state.get("refined_query", state["query"])
    
    # Determine context source
    course_content_found = state.get("course_content_found", False)
    logger.info(f"Personalization node - course_content_found: {course_content_found}")
    
    if course_content_found:
        context = state.get("course_context")
        citations = state.get("course_citations", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        is_from_web = False
        logger.info(f"Using course context (length: {len(context) if context else 0} chars, citations: {len(citations)}, chunks: {len(retrieved_chunks)})")
    else:
        context = state.get("web_search_results")
        citations = state.get("web_search_citations", [])
        retrieved_chunks = None
        is_from_web = True
        logger.info(f"Using web search context (length: {len(context) if context else 0} chars, citations: {len(citations)})")
    
    # Ensure we have context
    if not context or context.strip() == "":
        # Fallback: try to get any available context
        context = state.get("course_context") or state.get("web_search_results")
        if not context or context.strip() == "":
            # If still no context, provide a helpful message
            context = (
                f"I couldn't find specific information about '{query}' in the course materials. "
                "However, I'll do my best to provide a helpful answer based on general knowledge about the topic."
            )
    
    # Check if web search returned an error message
    if is_from_web and context:
        context_lower = context.lower()
        if "not available" in context_lower or "error performing" in context_lower or "configure" in context_lower or "no search results found" in context_lower:
            logger.warning(f"Web search returned error message. Context: {context[:200]}")
            # Try to provide a helpful response even if web search failed
            context = (
                f"I attempted to search the internet for current information about '{query}', but encountered an issue. "
                "This might be due to API configuration. However, based on general knowledge: "
                f"{query} is a topic that requires up-to-date information. "
                "I recommend checking official sources or recent publications for the most current details."
            )
        elif "internet search results" in context_lower or "[ai answer]" in context_lower or "[1]" in context:
            # Valid search results - log success
            logger.info(f"Web search returned valid results. Context length: {len(context)} chars")
    
    # Get conversation history for context (for resolving references like "the paper")
    messages = state.get("messages", [])
    conversation_context = ""
    if len(messages) > 1:
        # Include last 5 messages for context
        recent_messages = messages[-5:]
        conversation_context_parts = []
        for msg in recent_messages[:-1]:  # Exclude current query
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "User" if msg.type == "human" else "Assistant"
                conversation_context_parts.append(f"{role}: {msg.content[:200]}")  # Limit length
        if conversation_context_parts:
            conversation_context = "\nPrevious conversation:\n" + "\n".join(conversation_context_parts)
            logger.info(f"Adding conversation context ({len(conversation_context_parts)} messages) for personalization")
    
    # Generate personalized response
    result = agent.personalize_response(
        query=query,
        context=(context or "No context available") + conversation_context,
        user_context=state["user_context"],
        course_name=state["course_name"],
        citations=citations,
        is_from_web=is_from_web,
        retrieved_chunks=retrieved_chunks if not is_from_web else None
    )
    
    # Ensure we have a valid response
    final_response = result.get("response")
    if not final_response or final_response.strip() == "":
        final_response = (
            f"I apologize, but I couldn't generate a complete response to your question about '{query}'. "
            "Please try rephrasing your question or ask about a different topic."
        )
    
    state["final_response"] = final_response
    state["response_citations"] = result.get("citations", [])
    state["current_node"] = "personalization"
    state["should_continue"] = False
    state["next_node"] = None
    
    # Add the assistant's response to messages for checkpointing
    from langchain_core.messages import AIMessage
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append(AIMessage(content=final_response))
    logger.info(f"Added response to messages. Total messages in state: {len(state['messages'])}")
    
    logger.info("Personalized response generated.")
    
    return state

