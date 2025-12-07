"""Pinecone vector store integration for course materials."""

import logging
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Manages Pinecone vector store for course materials."""
    
    def __init__(self):
        """Initialize Pinecone client and index."""
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.index = None
            self._initialize_index()
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index."""
        try:
            existing_indexes = self.pc.list_indexes().names()
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                # Create index if it doesn't exist
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")
            else:
                logger.info(f"Connecting to existing index: {PINECONE_INDEX_NAME}")
            
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """Upsert documents to Pinecone with metadata."""
        if not documents:
            logger.warning("No documents to upsert")
            return
        
        try:
            texts = [doc["content"] for doc in documents]
            embeddings = self.create_embeddings(texts)
            
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Create unique vector ID
                vector_id = (
                    f"{doc['course_name']}_{doc['document_name']}_"
                    f"page{doc['page_number']}_chunk{doc.get('chunk_index', 0)}"
                )
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "course_name": doc["course_name"],
                        "document_name": doc["document_name"],
                        "page_number": doc["page_number"],
                        "content": doc["content"][:1000],  # Limit metadata size
                        "type": doc.get("type", "text"),
                        "chunk_index": doc.get("chunk_index", 0)
                    }
                })
            
            # Batch upsert
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
                logger.info(f"Upserted {total_upserted}/{len(vectors)} vectors")
            
            logger.info(f"Successfully upserted {len(vectors)} documents")
            
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        course_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone with course filtering.
        
        Args:
            query_text: The query text
            course_name: Course name to filter by
            top_k: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        try:
            logger.info(f"Creating embedding for query: '{query_text[:50]}...'")
            # Create query embedding
            query_embedding = self.create_embeddings([query_text])[0]
            logger.info(f"Embedding created, dimension: {len(query_embedding)}")
            
            # Normalize course_name for matching (strip whitespace)
            normalized_course_name = course_name.strip()
            
            logger.info(f"Querying Pinecone with filter: course_name='{normalized_course_name}', top_k={top_k}")
            
            # Query with metadata filter
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={
                        "course_name": {"$eq": normalized_course_name}
                    }
                )
            except Exception as filter_error:
                logger.warning(f"Error with course filter '{normalized_course_name}': {filter_error}")
                logger.info("Attempting query without filter and filtering results manually...")
                # If filter fails, try without filter and filter manually
                all_results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k * 3,  # Get more results to filter manually
                    include_metadata=True
                )
                # Filter manually by course name
                filtered_matches = []
                for match in all_results.matches:
                    match_course = match.metadata.get("course_name", "").strip()
                    if match_course == normalized_course_name:
                        filtered_matches.append(match)
                    else:
                        logger.debug(f"Skipping match with course_name='{match_course}' (expected '{normalized_course_name}')")
                
                # Create a results-like object
                from types import SimpleNamespace
                results = SimpleNamespace(matches=filtered_matches[:top_k])
                logger.info(f"Manually filtered to {len(results.matches)} matches for course '{normalized_course_name}'")
            
            logger.info(f"Pinecone returned {len(results.matches)} matches")
            
            # Format results
            formatted_results = []
            for match in results.matches:
                match_course = match.metadata.get("course_name", "")
                logger.debug(f"Match course: '{match_course}', score: {match.score}")
                formatted_results.append({
                    "content": match.metadata.get("content", ""),
                    "page_number": match.metadata.get("page_number", 0),
                    "document_name": match.metadata.get("document_name", ""),
                    "type": match.metadata.get("type", "text"),
                    "score": match.score,
                    "course_name": match_course
                })
            
            logger.info(f"Formatted {len(formatted_results)} results for course: {normalized_course_name}")
            if formatted_results:
                logger.info(f"Top result score: {formatted_results[0].get('score', 'N/A')}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}", exc_info=True)
            raise

