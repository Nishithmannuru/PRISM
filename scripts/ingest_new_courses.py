"""Script to ingest specific new courses into Pinecone vector store.
Targets only the courses listed in TARGET_COURSES (leaves existing courses untouched)."""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.document_loader import MultimodalPDFLoader
from retrieval.ppt_loader import PPTLoader
from retrieval.vector_store import PineconeVectorStore
from config.settings import COURSES_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Only ingest these courses
TARGET_COURSES = [
    "INFO 6945 - Trends and Issues in Information Science",
    "LTEC 4510 - Communications in Business, Education and Industry",
    "ADTA 5340 - Discovery and Learning with Big Data",
]


def process_file(file_path: Path, course_name: str, module_name: str, vector_store: PineconeVectorStore):
    """Process a single file (PDF or PPT/PPTX) and ingest into vector store."""
    file_ext = file_path.suffix.lower()

    try:
        if file_ext == '.pdf':
            logger.info(f"Processing PDF: {file_path.name} | course={course_name}" +
                        (f", module={module_name}" if module_name else ""))
            loader = MultimodalPDFLoader(
                course_name=course_name,
                document_path=str(file_path),
                module_name=module_name,
            )
        elif file_ext in ('.ppt', '.pptx'):
            logger.info(f"Processing PPT: {file_path.name} | course={course_name}" +
                        (f", module={module_name}" if module_name else ""))
            loader = PPTLoader(
                course_name=course_name,
                document_path=str(file_path),
                module_name=module_name,
            )
        else:
            logger.warning(f"Skipping unsupported file type: {file_ext} ({file_path.name})")
            return 0, 0

        documents = loader.load()

        if documents:
            vector_store.upsert_documents(documents)
            logger.info(f"  ✓ Ingested {len(documents)} chunks from {file_path.name}")
            return 1, len(documents)
        else:
            logger.warning(f"  ⚠ No chunks extracted from {file_path.name}")
            return 0, 0

    except Exception as e:
        logger.error(f"  ✗ Error processing {file_path.name}: {e}", exc_info=True)
        return 0, 0


def ingest_course(course_folder: Path, vector_store: PineconeVectorStore):
    """Ingest all documents for a single course, respecting module subfolders."""
    course_name = course_folder.name
    logger.info(f"\n{'='*60}")
    logger.info(f"INGESTING COURSE: {course_name}")
    logger.info(f"{'='*60}")

    total_files = 0
    total_chunks = 0

    # Collect files at root level AND inside subfolders
    # Walk the entire tree so nested folders (e.g., Readings/Week 5 - Greer/..) are found
    for dirpath, _dirnames, filenames in os.walk(course_folder):
        dirpath = Path(dirpath)
        # Determine module_name: first-level subfolder relative to course root
        rel = dirpath.relative_to(course_folder)
        parts = rel.parts
        module_name = parts[0] if parts else None

        for fname in sorted(filenames):
            fpath = dirpath / fname
            ext = fpath.suffix.lower()
            if ext not in ('.pdf', '.ppt', '.pptx'):
                continue

            files, chunks = process_file(fpath, course_name, module_name, vector_store)
            total_files += files
            total_chunks += chunks

    logger.info(f"\nCourse '{course_name}' done: {total_files} files → {total_chunks} chunks")
    return total_files, total_chunks


def main():
    courses_dir = Path(COURSES_PATH)
    if not courses_dir.exists():
        logger.error(f"Courses directory not found: {COURSES_PATH}")
        return

    vector_store = PineconeVectorStore()

    grand_files = 0
    grand_chunks = 0

    for target in TARGET_COURSES:
        folder = courses_dir / target
        if not folder.exists():
            logger.error(f"Course folder not found: {folder}")
            continue
        f, c = ingest_course(folder, vector_store)
        grand_files += f
        grand_chunks += c

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE — {grand_files} files, {grand_chunks} total chunks ingested.")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
