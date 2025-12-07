"""Script to check what's in the Pinecone vector store."""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.vector_store import PineconeVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_vector_store():
    """Check what course names and documents are in the vector store."""
    try:
        vs = PineconeVectorStore()
        
        # Get index stats
        stats = vs.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Try to query without filter to see what course names exist
        # We'll use a generic query
        test_query = "test"
        query_embedding = vs.create_embeddings([test_query])[0]
        
        # Query without filter to see all course names
        results = vs.index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        logger.info(f"\nFound {len(results.matches)} sample vectors:")
        course_names = set()
        for i, match in enumerate(results.matches[:10], 1):
            course_name = match.metadata.get("course_name", "N/A")
            doc_name = match.metadata.get("document_name", "N/A")
            page = match.metadata.get("page_number", "N/A")
            course_names.add(course_name)
            logger.info(f"  {i}. Course: '{course_name}', Doc: '{doc_name}', Page: {page}, Score: {match.score:.4f}")
        
        logger.info(f"\nUnique course names found in sample: {sorted(course_names)}")
        
        # Test a specific course name query
        if course_names:
            test_course = list(course_names)[0]
            logger.info(f"\nTesting query with course filter: '{test_course}'")
            filtered_results = vs.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"course_name": {"$eq": test_course}}
            )
            logger.info(f"Filtered query returned {len(filtered_results.matches)} results for course '{test_course}'")
        
    except Exception as e:
        logger.error(f"Error checking vector store: {e}", exc_info=True)


if __name__ == "__main__":
    check_vector_store()

