"""Flashcard Generator - Creates Q&A flashcards from course content."""

import logging
import json
import re
from typing import List, Dict, Any, Set
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from retrieval.retriever import CourseRetriever

logger = logging.getLogger(__name__)


class FlashcardGenerator:
    """Generates flashcards from course content."""
    
    def __init__(self):
        """Initialize the flashcard generator."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = CourseRetriever()
    
    def generate_flashcards(
        self,
        topic: str,
        course_name: str,
        existing_flashcards: List[Dict[str, Any]] = None,
        num_flashcards: int = 5
    ) -> Dict[str, Any]:
        """
        Generate flashcards for a given topic.
        
        Args:
            topic: The topic/question to generate flashcards for
            course_name: Name of the course
            existing_flashcards: Previously generated flashcards to avoid duplicates
            num_flashcards: Number of flashcards to generate
            
        Returns:
            Dictionary with flashcards, has_more flag, and message
        """
        try:
            # Enhance query to avoid references when asking about authors
            query_lower = topic.lower()
            enhanced_query = topic
            
            # If asking about authors, make it more specific to avoid cited references
            if any(keyword in query_lower for keyword in ['author', 'who wrote', 'who created', 'who developed']):
                # Enhance to focus on the document's own authors, not cited authors
                enhanced_query = f"{topic} document paper authors contributors"
                logger.info(f"Enhanced query for authors: '{enhanced_query}'")
            
            # Retrieve relevant content
            logger.info(f"Retrieving content for flashcard topic: '{topic}'")
            retrieved_chunks = self.retriever.retrieve(
                query=enhanced_query,
                course_name=course_name,
                top_k=15  # Get more chunks for flashcard generation
            )
            
            # If enhanced query didn't return good results, try original
            if not retrieved_chunks or len(retrieved_chunks) < 3:
                logger.info("Enhanced query returned few results, trying original query...")
                retrieved_chunks = self.retriever.retrieve(
                    query=topic,
                    course_name=course_name,
                    top_k=15
                )
            
            if not retrieved_chunks:
                return {
                    "flashcards": [],
                    "has_more": False,
                    "message": f"No content found for '{topic}'. Please try a different topic."
                }
            
            # Track used content to avoid duplicates
            used_content_ids = set()
            if existing_flashcards:
                # Extract content snippets from existing flashcards to avoid duplicates
                for card in existing_flashcards:
                    if 'content_id' in card:
                        used_content_ids.add(card['content_id'])
            
            # Filter out already used chunks and reference sections
            available_chunks = []
            import re
            
            for chunk in retrieved_chunks:
                chunk_id = chunk.get('content', '')[:100]  # Use first 100 chars as ID
                if chunk_id not in used_content_ids:
                    content_lower = chunk.get('content', '').lower()
                    is_reference = False
                    
                    # Strong indicators of reference sections
                    # Check if chunk starts with or is primarily citation patterns
                    content_start = content_lower[:200].strip()
                    
                    # Pattern 1: Starts with author name pattern followed by year (typical reference format)
                    # e.g., "Smith, J., & Doe, A. (2024). Title..."
                    author_year_pattern = r'^[A-Z][a-z]+,\s*[A-Z]\.\s*(?:&|and)?\s*[A-Z][a-z]+.*\(\d{4}\)'
                    if re.match(author_year_pattern, content_start):
                        is_reference = True
                    
                    # Pattern 2: Multiple citation patterns in a row (reference list)
                    citation_count = len(re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)', content_lower))
                    if citation_count > 3:  # More than 3 citations suggests reference list
                        is_reference = True
                    
                    # Pattern 3: Contains "References" or "Bibliography" as section header
                    if re.search(r'^(references|bibliography)\s*$', content_start, re.MULTILINE):
                        is_reference = True
                    
                    # Pattern 4: Document name indicates references
                    doc_name = chunk.get('document_name', '').lower()
                    if 'reference' in doc_name or 'bibliography' in doc_name:
                        is_reference = True
                    
                    # Pattern 5: Mostly URLs and DOIs (typical of reference entries)
                    url_count = len(re.findall(r'https?://|doi:', content_lower))
                    if url_count > 2 and len(content_lower) < 500:  # Short chunk with many URLs
                        is_reference = True
                    
                    # Skip reference chunks
                    if not is_reference:
                        available_chunks.append(chunk)
                    else:
                        logger.debug(f"Filtered out reference chunk: {content_start[:100]}...")
            
            # If we filtered out all chunks, use original chunks (better than nothing)
            if not available_chunks and retrieved_chunks:
                logger.warning("All chunks appeared to be references. Using original chunks anyway.")
                available_chunks = [chunk for chunk in retrieved_chunks 
                                  if chunk.get('content', '')[:100] not in used_content_ids]
            
            if not available_chunks:
                return {
                    "flashcards": [],
                    "has_more": False,
                    "message": "We've covered everything available for this topic! Try asking about a different aspect or topic."
                }
            
            # Re-format context with only available chunks
            context = self.retriever.format_context(available_chunks[:10])  # Use top 10 available
            
            # Generate flashcards using LLM
            system_prompt = """You are a flashcard generator for educational content. 
Create clear, concise question-and-answer flashcards based on the provided course content.

Guidelines:
- Each flashcard should have a clear question and a concise answer
- Questions should test understanding, not just recall
- Answers should be accurate and based ONLY on the provided content
- Avoid creating flashcards that are too similar to each other
- Focus on key concepts, definitions, processes, and important facts
- Keep answers brief but informative (2-4 sentences max)
- IMPORTANT: Ignore any reference citations, bibliography entries, or cited works. Focus only on the main content of the document itself.
- If the content appears to be from a references section, skip it and use other available content

Format your response as a JSON object with a "flashcards" key containing an array of objects, each with "question" and "answer" fields."""
            
            user_prompt = f"""Based on the following course content about '{topic}', generate exactly {num_flashcards} flashcards.

Course Content:
{context}

Topic: {topic}

IMPORTANT: 
- Focus on the MAIN CONTENT of the document, NOT on cited references or bibliography entries
- If asking about "authors", use the authors of THIS document/paper, not authors of cited works
- Ignore any citation patterns like [1], (Author, Year), or reference lists
- Use only information from the actual document content

Generate {num_flashcards} diverse flashcards covering different aspects of this topic. Return ONLY a valid JSON object with a "flashcards" array, no other text.

Example format:
{{
  "flashcards": [
    {{"question": "What is X?", "answer": "X is..."}},
    {{"question": "How does Y work?", "answer": "Y works by..."}}
  ]
}}"""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON
            try:
                parsed = json.loads(content)
                # Extract flashcards array
                if isinstance(parsed, dict):
                    flashcards_data = parsed.get("flashcards", [])
                elif isinstance(parsed, list):
                    flashcards_data = parsed
                else:
                    flashcards_data = []
                
            except json.JSONDecodeError:
                # Try to extract JSON array from text
                json_match = re.search(r'\{.*"flashcards".*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        flashcards_data = parsed.get("flashcards", [])
                    except:
                        flashcards_data = []
                else:
                    logger.error(f"Could not parse flashcard JSON: {content}")
                    flashcards_data = []
            
            # Create flashcard objects with metadata
            flashcards = []
            for i, card_data in enumerate(flashcards_data[:num_flashcards]):
                if isinstance(card_data, dict) and "question" in card_data and "answer" in card_data:
                    # Use the chunk that was most relevant for this flashcard
                    chunk_idx = min(i, len(available_chunks) - 1)
                    source_chunk = available_chunks[chunk_idx] if available_chunks else {}
                    
                    flashcard = {
                        "question": card_data["question"],
                        "answer": card_data["answer"],
                        "topic": topic,
                        "content_id": source_chunk.get('content', '')[:100] if source_chunk else f"card_{i}",
                        "source": {
                            "document": source_chunk.get('document_name', 'Unknown'),
                            "module": source_chunk.get('module_name'),
                            "page": source_chunk.get('page_number'),
                            "timestamp": source_chunk.get('timestamp')
                        }
                    }
                    flashcards.append(flashcard)
            
            # Check if there's more content available
            remaining_chunks = len(available_chunks) - len(flashcards)
            has_more = remaining_chunks > 0 and len(flashcards) > 0
            
            return {
                "flashcards": flashcards,
                "has_more": has_more,
                "message": None
            }
            
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}", exc_info=True)
            return {
                "flashcards": [],
                "has_more": False,
                "message": f"Error generating flashcards: {str(e)}"
            }

