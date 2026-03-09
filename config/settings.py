"""Configuration settings for PRISM application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Override shell env vars with .env values

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = 3072  # For text-embedding-3-large

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "prism-course-materials")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
COURSES_DIR = "courses"
COURSES_PATH = BASE_DIR / COURSES_DIR

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

