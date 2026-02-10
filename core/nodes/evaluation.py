"""Evaluation Agent - Mathematically evaluates response quality."""

import logging
import numpy as np
import re
import textstat
from typing import Dict, Any, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import ssl

# Fix SSL context for NLTK downloads (macOS certificate issue)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from retrieval.vector_store import PineconeVectorStore
from config.settings import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Download required NLTK data
# NLTK 3.8+ uses punkt_tab, older versions use punkt
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download punkt_tab: {e}. Will use fallback sentence splitting.")
        # Fallback for older NLTK versions
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e2:
                logger.warning(f"Could not download punkt: {e2}. Will use fallback sentence splitting.")


class EvaluationAgent:
    """Agent that evaluates response quality using mathematical formulas."""
    
    def __init__(self):
        """Initialize the evaluation agent."""
        self.vector_store = PineconeVectorStore()
        self.openai_client = None
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
    
    def _embed_one(self, text: str) -> np.ndarray:
        """Create embedding for a single text."""
        if not text or not text.strip():
            # Return zero vector if empty
            try:
                dummy_emb = self.vector_store.create_embeddings(["dummy"])
                return np.zeros(len(dummy_emb[0]))
            except:
                return np.zeros(3072)  # Default dimension
        
        try:
            embeddings = self.vector_store.create_embeddings([text])
            return np.array(embeddings[0])
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return zero vector on error
            return np.zeros(3072)  # Default dimension
    
    def _normalize(self, x: float, a: float, b: float) -> float:
        """Normalize value x to [0, 1] range between a and b."""
        if b == a:
            return 0.0
        return np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    
    def _inv_normalize(self, x: float, a: float, b: float) -> float:
        """Inverse normalize (higher is better)."""
        return 1.0 - self._normalize(x, a, b)
    
    def _weighted_sum(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted sum."""
        if len(values) != len(weights):
            logger.warning(f"Values and weights length mismatch: {len(values)} vs {len(weights)}")
            # Normalize weights if mismatch
            weights = weights[:len(values)]
            if weights:
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
        
        return sum(a * b for a, b in zip(values, weights))
    
    def readability_complexity(
        self, 
        text: str, 
        degree_level: str
    ) -> float:
        """
        Calculate readability match for degree level.
        
        Args:
            text: Answer text
            degree_level: User's degree level (e.g., "Bachelors", "Masters", "PhD")
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0
        
        try:
            # Calculate Flesch-Kincaid grade level
            fk_grade = textstat.flesch_kincaid_grade(text)
            
            # Define target grade bands based on degree level
            if "PhD" in degree_level or "Doctor" in degree_level:
                target_band = (14, 18)  # Graduate level
            elif "Master" in degree_level:
                target_band = (12, 16)  # Graduate level
            else:
                target_band = (10, 14)  # Undergraduate level
            
            # Calculate match using Gaussian-like function
            mu = (target_band[0] + target_band[1]) / 2.0
            sigma_sq = ((target_band[1] - target_band[0]) / 2.0) ** 2 + 1e-9
            
            # Gaussian probability density (normalized)
            match_score = np.exp(-((fk_grade - mu) ** 2) / (2 * sigma_sq))
            
            return float(np.clip(match_score, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.5  # Default middle score
    
    def coherence_fluency(
        self, 
        sentence_embeddings: List[np.ndarray],
        sentences: List[str] = None
    ) -> float:
        """
        Calculate coherence and fluency score.
        
        Args:
            sentence_embeddings: List of sentence embeddings
            sentences: List of sentence strings (optional, for fluency calculation)
            
        Returns:
            Score between 0.0 and 1.0
        """
        if len(sentence_embeddings) < 2:
            return 0.0
        
        try:
            # Local coherence: average cosine similarity between adjacent sentences
            local_coherences = []
            for i in range(len(sentence_embeddings) - 1):
                emb1 = sentence_embeddings[i]
                emb2 = sentence_embeddings[i + 1]
                
                # Check for zero vectors or invalid embeddings
                if np.all(emb1 == 0) or np.all(emb2 == 0):
                    # Skip zero vectors
                    continue
                
                # Check for NaN or Inf values
                if np.any(np.isnan(emb1)) or np.any(np.isnan(emb2)) or \
                   np.any(np.isinf(emb1)) or np.any(np.isinf(emb2)):
                    # Skip invalid embeddings
                    continue
                
                sim = cosine_similarity(
                    emb1.reshape(1, -1),
                    emb2.reshape(1, -1)
                )[0][0]
                
                # Check if similarity is valid
                if not np.isnan(sim) and not np.isinf(sim):
                    local_coherences.append(sim)
            
            local_coherence = np.mean(local_coherences) if local_coherences else 0.0
            
            # Fluency: based on sentence length consistency (proxy for perplexity)
            # More consistent sentence lengths indicate better fluency
            if sentences and len(sentences) > 1:
                sentence_lengths = [len(s.split()) for s in sentences]
                length_mean = np.mean(sentence_lengths)
                length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
                
                # Normalize: lower std relative to mean = better fluency
                # Score is higher when std is small relative to mean
                if length_mean > 0:
                    cv = length_std / length_mean  # Coefficient of variation
                    # Normalize CV: 0.0 (perfect) to 1.0 (poor)
                    # Good fluency: CV < 0.5, excellent: CV < 0.3
                    fluency_score = self._inv_normalize(cv, 0.0, 1.0)
                else:
                    fluency_score = 0.5
            else:
                # Default fluency score if sentences not provided
                fluency_score = 0.7
            
            # Weighted combination: coherence (70%) + fluency (30%)
            return self._weighted_sum([local_coherence, fluency_score], [0.7, 0.3])
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.5
    
    def relevance_score(
        self,
        query_embedding: np.ndarray,
        answer_embedding: np.ndarray,
        context_embeddings: List[np.ndarray]
    ) -> float:
        """
        Calculate relevance score.
        
        Args:
            query_embedding: Query vector
            answer_embedding: Answer vector
            context_embeddings: List of context chunk vectors
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Check for zero vectors or invalid embeddings
            if np.all(query_embedding == 0) or np.all(answer_embedding == 0):
                return 0.0
            
            # Check for NaN or Inf values
            if np.any(np.isnan(query_embedding)) or np.any(np.isnan(answer_embedding)) or \
               np.any(np.isinf(query_embedding)) or np.any(np.isinf(answer_embedding)):
                return 0.0
            
            # Query-Answer similarity
            qa_similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                answer_embedding.reshape(1, -1)
            )[0][0]
            
            # Check if similarity is valid
            if np.isnan(qa_similarity) or np.isinf(qa_similarity):
                qa_similarity = 0.0
            
            # Answer-Context similarity (average)
            if context_embeddings:
                ac_similarities = []
                for ctx_emb in context_embeddings:
                    # Check for zero or invalid context embeddings
                    if np.all(ctx_emb == 0) or np.any(np.isnan(ctx_emb)) or np.any(np.isinf(ctx_emb)):
                        continue
                    
                    sim = cosine_similarity(
                        answer_embedding.reshape(1, -1),
                        ctx_emb.reshape(1, -1)
                    )[0][0]
                    
                    # Check if similarity is valid
                    if not np.isnan(sim) and not np.isinf(sim):
                        ac_similarities.append(sim)
                ac_similarity = np.mean(ac_similarities) if ac_similarities else 0.0
            else:
                ac_similarity = 0.0
            
            # Weighted combination: favor query-answer similarity (70%) over answer-context (30%)
            # Direct relevance to the question is more important than context alignment
            return self._weighted_sum([qa_similarity, ac_similarity], [0.7, 0.3])
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5
    
    def coverage(self, answer: str, query: str) -> float:
        """
        Calculate coverage score (how many query aspects are covered).
        Uses both keyword matching and semantic similarity for better accuracy.
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not answer or not query:
            return 0.0
        
        try:
            # Split query into sub-questions/aspects
            # Split by common separators and conjunctions
            sub_parts = re.split(r"[;,.]|\band\b|\bor\b", query.lower())
            sub_parts = [s.strip() for s in sub_parts if s.strip() and len(s.strip()) > 2]
            
            if not sub_parts:
                # If no clear sub-parts, use semantic similarity for main query
                query_emb = self._embed_one(query)
                answer_emb = self._embed_one(answer)
                
                if np.all(query_emb == 0) or np.all(answer_emb == 0):
                    # Fallback to word overlap
                    query_words = set(query.lower().split())
                    answer_words = set(answer.lower().split())
                    overlap = len(query_words.intersection(answer_words))
                    return min(overlap / max(len(query_words), 1), 1.0)
                
                # Use semantic similarity
                sim = cosine_similarity(
                    query_emb.reshape(1, -1),
                    answer_emb.reshape(1, -1)
                )[0][0]
                
                if np.isnan(sim) or np.isinf(sim):
                    # Fallback to word overlap
                    query_words = set(query.lower().split())
                    answer_words = set(answer.lower().split())
                    overlap = len(query_words.intersection(answer_words))
                    return min(overlap / max(len(query_words), 1), 1.0)
                
                return float(np.clip(sim, 0.0, 1.0))
            
            # Count how many sub-parts are found in answer using semantic similarity
            found_scores = []
            answer_emb = self._embed_one(answer)
            
            for part in sub_parts:
                if not part.strip():
                    continue
                
                # Get embedding for this query part
                part_emb = self._embed_one(part)
                
                # Check for zero or invalid embeddings
                if np.all(part_emb == 0) or np.all(answer_emb == 0) or \
                   np.any(np.isnan(part_emb)) or np.any(np.isnan(answer_emb)):
                    # Fallback to keyword matching
                    part_words = [w for w in part.split() if len(w) > 2]
                    answer_lower = answer.lower()
                    if part_words and any(word in answer_lower for word in part_words):
                        found_scores.append(0.7)  # Partial credit for keyword match
                    continue
                
                # Calculate semantic similarity
                sim = cosine_similarity(
                    part_emb.reshape(1, -1),
                    answer_emb.reshape(1, -1)
                )[0][0]
                
                if not np.isnan(sim) and not np.isinf(sim):
                    # Threshold: similarity > 0.3 means the aspect is covered
                    if sim > 0.3:
                        found_scores.append(min(sim, 1.0))
            
            if not found_scores:
                return 0.0
            
            # Average coverage score across all aspects
            return np.mean(found_scores)
        except Exception as e:
            logger.error(f"Error calculating coverage: {e}")
            return 0.5
    
    def source_credibility(self, sources: List[Dict[str, Any]]) -> float:
        """
        Calculate source credibility score.
        
        Args:
            sources: List of source metadata dicts with keys:
                - venue: Publication venue (0-1 score)
                - author: Author reputation (0-1 score)
                - recency: How recent (0-1 score, 1.0 = very recent)
                - citation: Citation count normalized (0-1 score)
                - integrity: Source integrity score (0-1)
                
        Returns:
            Score between 0.0 and 1.0
        """
        if not sources:
            return 0.0
        
        try:
            credibility_scores = []
            for source in sources:
                # Extract scores (default to 0.5 if not provided)
                venue = source.get("venue", 0.5)
                author = source.get("author", 0.5)
                recency = source.get("recency", 0.5)
                citation = source.get("citation", 0.5)
                integrity = source.get("integrity", 0.5)
                
                # Weighted average
                score = self._weighted_sum(
                    [venue, author, recency, citation, integrity],
                    [0.2, 0.2, 0.2, 0.2, 0.2]
                )
                credibility_scores.append(score)
            
            return np.mean(credibility_scores) if credibility_scores else 0.0
        except Exception as e:
            logger.error(f"Error calculating credibility: {e}")
            return 0.5
    
    def consensus_score(self, entailment_scores: List[List[float]]) -> float:
        """
        Calculate consensus score from entailment matrix.
        
        Note: This method is defined for future claim-level entailment modeling.
        Currently, consensus is approximated as a lightweight proxy based on source count
        in the web response evaluation, under the assumption that multiple authoritative
        sources increase confidence in answer accuracy.
        
        Args:
            entailment_scores: 2D list where each row is a claim and each column is a source
                Values should be in [-1, 1] range (-1 = contradiction, 0 = neutral, 1 = entailment)
                
        Returns:
            Score between 0.0 and 1.0
        """
        if not entailment_scores or not entailment_scores[0]:
            return 0.0
        
        try:
            arr = np.array(entailment_scores, dtype=float)
            
            # Average across sources for each claim
            claim_avg = np.mean(arr, axis=1)
            
            # Normalize from [-1, 1] to [0, 1]
            normalized = (claim_avg + 1.0) / 2.0
            
            # Overall consensus is the mean
            return float(np.mean(normalized))
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return 0.5
    
    def logical_consistency(self, contradiction_rate: float) -> float:
        """
        Calculate logical consistency score.
        
        Uses a conservative prior approach: initializes with neutral consistency assumption
        and penalizes only when explicit contradictions are detected. In the current
        implementation, explicit contradiction detection is deferred to future work.
        
        Args:
            contradiction_rate: Rate of contradictions found (0.0 = none, 1.0 = all contradictory)
            
        Returns:
            Score between 0.0 and 1.0 (1.0 = no contradictions)
        """
        return 1.0 - np.clip(contradiction_rate, 0.0, 1.0)
    
    def evaluate_course_response(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        degree_level: str
    ) -> Dict[str, float]:
        """
        Evaluate response for course-based answers.
        
        Pedagogical quality metrics (relevance, readability, coherence, coverage) are
        weighted to emphasize learning effectiveness. Relevance receives the highest
        weight (35%) as it directly measures how well the answer addresses the student's
        question, which is the primary indicator of pedagogical value.
        
        Args:
            query: User's question
            answer: Generated answer
            retrieved_chunks: Retrieved context chunks
            degree_level: User's degree level
            
        Returns:
            Dictionary with scores: relevance, readability, coherence, coverage, overall
        """
        try:
            # Ensure retrieved_chunks is a list
            if retrieved_chunks is None:
                retrieved_chunks = []
            
            # Create embeddings
            query_emb = self._embed_one(query)
            answer_emb = self._embed_one(answer)
            
            # Get context embeddings (top 3 chunks)
            context_embeddings = []
            for chunk in retrieved_chunks[:3]:
                content = chunk.get("content", "")
                if content:
                    ctx_emb = self._embed_one(content)
                    context_embeddings.append(ctx_emb)
            
            # Calculate metrics
            relevance = self.relevance_score(query_emb, answer_emb, context_embeddings)
            readability = self.readability_complexity(answer, degree_level)
            
            # Coherence: tokenize sentences and embed
            try:
                sentences = sent_tokenize(answer)
                sentence_embeddings = [self._embed_one(s) for s in sentences if s.strip()]
            except LookupError:
                # Fallback: split by periods if NLTK tokenizer not available
                logger.warning("NLTK punkt_tab not available, using simple sentence splitting")
                sentences = [s.strip() + '.' for s in answer.split('.') if s.strip()]
                sentence_embeddings = [self._embed_one(s) for s in sentences if s.strip()]
            
            coherence = self.coherence_fluency(sentence_embeddings, sentences)
            
            coverage = self.coverage(answer, query)
            
            # Calculate overall (weighted): emphasize relevance (35%), balance others
            # Weighting philosophy: Pedagogical quality metrics receive full weight for
            # course-based responses, with relevance prioritized as the primary indicator
            # of how well the answer addresses the student's question
            overall = self._weighted_sum(
                [relevance, readability, coherence, coverage],
                [0.35, 0.25, 0.2, 0.2]
            )
            
            return {
                "relevance": float(relevance),
                "readability": float(readability),
                "coherence": float(coherence),
                "coverage": float(coverage),
                "overall": float(overall)
            }
        except Exception as e:
            logger.error(f"Error evaluating course response: {e}", exc_info=True)
            # Log more details about what might be wrong
            logger.error(f"Query: {query[:100] if query else 'None'}")
            logger.error(f"Answer length: {len(answer) if answer else 0}")
            logger.error(f"Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
            logger.error(f"Degree level: {degree_level}")
            # Return default scores on error
            return {
                "relevance": 0.5,
                "readability": 0.5,
                "coherence": 0.5,
                "coverage": 0.5,
                "overall": 0.5
            }
    
    def evaluate_web_response(
        self,
        query: str,
        answer: str,
        web_sources: List[Dict[str, Any]],
        degree_level: str
    ) -> Dict[str, float]:
        """
        Evaluate response for web-based answers.
        
        Combines pedagogical quality metrics (80% weight) with trust metrics (20% weight).
        This weighting reflects PRISM's primary goal of supporting learning rather than
        fact verification alone. Trust metrics (credibility, consensus, consistency) serve
        as secondary safeguards to ensure information reliability without dominating the
        quality assessment.
        
        Args:
            query: User's question
            answer: Generated answer
            web_sources: Web search sources with metadata
            degree_level: User's degree level
            
        Returns:
            Dictionary with all scores including credibility, consensus, consistency
        """
        try:
            # Ensure web_sources is a list
            if web_sources is None:
                web_sources = []
            
            # Base metrics (same as course)
            query_emb = self._embed_one(query)
            answer_emb = self._embed_one(answer)
            
            # For web, we don't have retrieved chunks, so use answer itself as context
            context_embeddings = [answer_emb]
            
            relevance = self.relevance_score(query_emb, answer_emb, context_embeddings)
            readability = self.readability_complexity(answer, degree_level)
            
            # Coherence: tokenize sentences and embed
            try:
                sentences = sent_tokenize(answer)
                sentence_embeddings = [self._embed_one(s) for s in sentences if s.strip()]
            except LookupError:
                # Fallback: split by periods if NLTK tokenizer not available
                logger.warning("NLTK punkt_tab not available, using simple sentence splitting")
                sentences = [s.strip() + '.' for s in answer.split('.') if s.strip()]
                sentence_embeddings = [self._embed_one(s) for s in sentences if s.strip()]
            
            coherence = self.coherence_fluency(sentence_embeddings, sentences)
            coverage = self.coverage(answer, query)
            
            # Web-specific metrics
            # Extract source metadata with basic domain-based credibility
            source_metadata = []
            for source in web_sources:
                # Basic domain-based credibility scoring
                url = source.get("url", "") if isinstance(source, dict) else ""
                domain = ""
                if url:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        domain = parsed.netloc.lower()
                    except:
                        pass
                
                # Credibility based on domain (simplified heuristic)
                venue_score = 0.5  # Default
                if domain:
                    # Academic/educational domains
                    if any(edu in domain for edu in ['.edu', '.ac.', '.gov']):
                        venue_score = 0.9
                    # Reputable news/orgs
                    elif any(rep in domain for rep in ['.org', 'wikipedia', 'scholar']):
                        venue_score = 0.7
                    # Known unreliable domains
                    elif any(unrel in domain for unrel in ['blogspot', 'wordpress.com']):
                        venue_score = 0.4
                
                source_metadata.append({
                    "venue": venue_score,
                    "author": 0.5,  # Unknown author
                    "recency": 0.8,  # Web sources are usually recent
                    "citation": 0.5,  # Unknown citation count
                    "integrity": 0.6  # Moderate integrity
                })
            
            credibility = self.source_credibility(source_metadata) if source_metadata else 0.5
            
            # Consensus: Lightweight proxy for source agreement
            # Approximates consensus as a function of the number of independent sources
            # Assumes multiple authoritative sources increase confidence in answer accuracy
            # Note: This is a computational heuristic; future work will incorporate
            # explicit claim-level entailment modeling for semantic agreement measurement
            if len(web_sources) >= 5:
                consensus = 0.8  # High consensus proxy with many sources
            elif len(web_sources) >= 3:
                consensus = 0.7  # Moderate consensus proxy
            elif len(web_sources) >= 1:
                consensus = 0.5  # Single source = unknown consensus
            else:
                consensus = 0.3  # No sources = low consensus
            
            # Logical consistency: Conservative prior approach
            # Initializes with neutral consistency prior (0.75), reflecting assumption
            # that responses are generally consistent in absence of detected contradictions
            # This prior would be penalized when explicit contradictions are identified
            # Future work: Implement explicit contradiction detection mechanisms
            consistency = 0.75  # Consistency prior, not a hardcoded score
            
            # Calculate overall (weighted): emphasize relevance and core metrics
            # Weighting philosophy: Pedagogical quality metrics (relevance, readability,
            # coherence, coverage) receive 80% weight, while trust metrics (credibility,
            # consensus, consistency) receive 20%. This reflects PRISM's primary goal
            # of supporting learning and knowledge acquisition rather than fact verification
            # alone. Trust metrics serve as secondary safeguards for information reliability.
            overall = self._weighted_sum(
                [relevance, readability, coherence, coverage, 
                 credibility, consensus, consistency],
                [0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]
            )
            
            return {
                "relevance": float(relevance),
                "readability": float(readability),
                "coherence": float(coherence),
                "coverage": float(coverage),
                "credibility": float(credibility),
                "consensus": float(consensus),
                "consistency": float(consistency),
                "overall": float(overall)
            }
        except Exception as e:
            logger.error(f"Error evaluating web response: {e}", exc_info=True)
            # Log more details about what might be wrong
            logger.error(f"Query: {query[:100] if query else 'None'}")
            logger.error(f"Answer length: {len(answer) if answer else 0}")
            logger.error(f"Web sources count: {len(web_sources) if web_sources else 0}")
            logger.error(f"Degree level: {degree_level}")
            return {
                "relevance": 0.5,
                "readability": 0.5,
                "coherence": 0.5,
                "coverage": 0.5,
                "credibility": 0.5,
                "consensus": 0.5,
                "consistency": 0.5,
                "overall": 0.5
            }


def evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for evaluation.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with evaluation scores
    """
    # Eval runner: skip internal evaluation when running ablation (no_internal_eval)
    if state.get("skip_internal_eval"):
        state["evaluation_passed"] = True
        state["evaluation_scores"] = {"overall": 1.0}
        state["refinement_attempts"] = state.get("refinement_attempts", 0)
        if "response_history" not in state:
            state["response_history"] = []
        state["response_history"].append({
            "response": state.get("final_response", ""),
            "score": 1.0
        })
        return state

    agent = EvaluationAgent()
    
    query = state.get("refined_query", state["query"])
    answer = state.get("final_response", "")
    course_content_found = state.get("course_content_found", False)
    user_context = state.get("user_context", {})
    degree_level = user_context.get("degree", "Bachelors")
    
    if not answer:
        logger.warning("No answer to evaluate")
        state["evaluation_scores"] = {"overall": 0.0}
        state["evaluation_passed"] = False
        return state
    
    # Evaluate based on source type
    if course_content_found:
        retrieved_chunks = state.get("retrieved_chunks", [])
        # Ensure retrieved_chunks is a list, not None
        if retrieved_chunks is None:
            retrieved_chunks = []
        if not retrieved_chunks:
            # Fallback: try to get from course_context if available
            retrieved_chunks = []
        logger.debug(f"Evaluating course response with {len(retrieved_chunks)} chunks")
        scores = agent.evaluate_course_response(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            degree_level=degree_level
        )
    else:
        web_sources = state.get("web_search_citations", [])
        if web_sources is None:
            web_sources = []
        logger.debug(f"Evaluating web response with {len(web_sources)} sources")
        scores = agent.evaluate_web_response(
            query=query,
            answer=answer,
            web_sources=web_sources,
            degree_level=degree_level
        )
    
    # Check if passes threshold
    threshold = 0.70
    passed = scores["overall"] >= threshold
    
    state["evaluation_scores"] = scores
    state["evaluation_passed"] = passed
    state["refinement_attempts"] = state.get("refinement_attempts", 0)
    
    # Track response and score in history for logging
    if "response_history" not in state:
        state["response_history"] = []
    
    state["response_history"].append({
        "response": answer,
        "score": scores["overall"]
    })
    
    logger.info(f"Evaluation complete. Overall score: {scores['overall']:.3f}, Passed: {passed}")
    logger.debug(f"Detailed scores: {scores}")
    
    return state
