"""Internet search integration using Tavily API."""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed. Install with: pip install tavily-python")


class InternetSearchAgent:
    """Agent for performing internet searches using Tavily API."""
    
    def __init__(self):
        """Initialize the internet search agent."""
        # Try to load from environment, also check dotenv
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.environ.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not found in environment variables. Web search will be disabled.")
            logger.warning("Please ensure TAVILY_API_KEY is set in your .env file")
            self.client = None
        elif not TAVILY_AVAILABLE:
            logger.warning("tavily-python package not installed. Web search will be disabled.")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily API client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Tavily client: {e}")
                self.client = None
    
    def search(
        self,
        query: str,
        course_name: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """
        Perform internet search for a query.
        
        Args:
            query: Search query
            course_name: Course name for context
            num_results: Number of results to return
            
        Returns:
            Dictionary with search results and citations
        """
        if not self.api_key or not self.client:
            return {
                "results": "Web search is not available. Please configure TAVILY_API_KEY and install tavily-python.",
                "citations": []
            }
        
        try:
            # Check if query asks for current/latest information
            query_lower = query.lower()
            needs_current_info = any(keyword in query_lower for keyword in [
                "latest", "current", "recent", "new", "updated", "now", "today", "2024", "2025"
            ])
            
            # For current info queries, don't add course name (it dilutes results)
            # For general queries, add course context if helpful
            if needs_current_info:
                enhanced_query = query  # Use original query for current info
                logger.info(f"Current info query detected - using original query without course context")
            else:
                enhanced_query = f"{query} {course_name}"
                logger.info(f"General query - enhancing with course context")
            
            # Perform search using Tavily
            # For current info queries, use advanced search and get more results
            search_params = {
                "query": enhanced_query,
                "max_results": num_results * 2 if needs_current_info else num_results,
                "search_depth": "advanced" if needs_current_info else "basic",
                "include_answer": True,  # Get AI-generated answer if available
                "include_raw_content": False,
                "include_domains": [],  # Don't restrict domains
            }
            
            # Add date context for current info queries
            if needs_current_info:
                # Add current year/month to query to get most recent results
                from datetime import datetime
                current_year = datetime.now().year
                current_month = datetime.now().strftime("%B")
                # Enhance query with date context
                enhanced_query = f"{enhanced_query} {current_year} {current_month}"
                search_params["query"] = enhanced_query
                logger.info(f"Added date context ({current_month} {current_year}) to query for current information")
            
            logger.info(f"Searching Tavily with query: '{enhanced_query}'")
            logger.info(f"Search params: max_results={search_params['max_results']}, search_depth={search_params['search_depth']}")
            
            search_response = self.client.search(**search_params)
            
            logger.info(f"Tavily search response keys: {list(search_response.keys()) if isinstance(search_response, dict) else 'Not a dict'}")
            
            # Format results
            formatted_results = []
            citations = []
            
            # Add AI-generated answer if available (Tavily provides this)
            if search_response.get("answer"):
                formatted_results.append({
                    "title": "AI-Generated Answer",
                    "snippet": search_response["answer"],
                    "link": "",
                    "position": 0,
                    "type": "answer"
                })
                logger.info("Found AI-generated answer from Tavily")
            
            # Extract search results
            results_list = search_response.get("results", [])
            
            # For current info queries, try to extract dates and prioritize recent ones
            if needs_current_info:
                from datetime import datetime
                current_year = datetime.now().year
                
                # Try to extract year from content and prioritize recent results
                def extract_year_from_text(text):
                    """Extract year from text (look for 4-digit years like 2024, 2025)"""
                    import re
                    years = re.findall(r'\b(20\d{2})\b', str(text))
                    if years:
                        return max([int(y) for y in years if 2020 <= int(y) <= current_year + 1])
                    return None
                
                # Add year information to results for sorting
                for result in results_list:
                    content = result.get("content", "")
                    title = result.get("title", "")
                    combined_text = f"{title} {content}"
                    year = extract_year_from_text(combined_text)
                    result['_extracted_year'] = year if year else 0
                
                # Sort by extracted year (most recent first), then by score
                results_list.sort(key=lambda x: (x.get('_extracted_year', 0), x.get('score', 0)), reverse=True)
                logger.info(f"Sorted results by date relevance for current info query")
            
            for i, result in enumerate(results_list[:num_results], 1):
                title = result.get("title", "No title")
                content = result.get("content", "")
                url = result.get("url", "")
                score = result.get("score", 0)
                extracted_year = result.get("_extracted_year")
                
                # Add year info to snippet if available
                snippet = content[:500] if content else ""
                if extracted_year and needs_current_info:
                    snippet = f"[Year: {extracted_year}] {snippet}"
                
                formatted_results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": url,
                    "position": i,
                    "score": score,
                    "year": extracted_year
                })
                
                citations.append({
                    "source": title,
                    "url": url
                })
            
            # Sort results: answer first, then by year (if current info), then by score
            if needs_current_info:
                formatted_results.sort(key=lambda x: (
                    x.get('position', 0) == 0,  # Answer first
                    -x.get('year', 0),  # Most recent year first
                    -x.get('score', 0)  # Then by score
                ))
            else:
                formatted_results.sort(key=lambda x: (x.get('position', 0) == 0, -x.get('score', 0)))
            
            # Format as text
            results_text = "Internet Search Results:\n\n"
            for result in formatted_results:
                result_type = result.get('type', 'organic')
                if result_type == 'answer':
                    results_text += f"[AI Answer] {result['title']}\n"
                    results_text += f"{result['snippet']}\n\n"
                else:
                    results_text += f"[{int(result['position'])}] {result['title']}\n"
                    results_text += f"{result['snippet']}\n"
                    if result.get('link'):
                        results_text += f"Source: {result['link']}\n"
                    results_text += "\n"
            
            logger.info(f"Found {len(formatted_results)} search results for query: {query}")
            
            # Validate we have actual results
            if not formatted_results or len(formatted_results) == 0:
                logger.warning(f"No results returned from Tavily for query: {query}")
                return {
                    "results": f"No search results found for '{query}'. Please try rephrasing your question.",
                    "citations": []
                }
            
            return {
                "results": results_text,
                "citations": citations,
                "raw_results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error performing internet search with Tavily: {e}", exc_info=True)
            # Provide more helpful error message
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return {
                    "results": "Web search is not available. Please check your TAVILY_API_KEY configuration.",
                    "citations": []
                }
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                return {
                    "results": "Web search rate limit exceeded. Please try again later.",
                    "citations": []
                }
            else:
                return {
                    "results": f"Error performing search: {error_msg}. Please check your Tavily API configuration.",
                    "citations": []
                }
