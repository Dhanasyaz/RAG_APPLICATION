"""Configuration and API keys"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys - read from environment variables
EURON_API_KEY = os.getenv("EURON_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Pinecone Configuration
PINECONE_INDEX_NAME = "rag-docs"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Configuration
LLM_MODEL = "gpt-4.1-nano"
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# Retrieval Configuration
TOP_K = 3  # Number of chunks to retrieve

# Validation (warn but don't crash during import)
if not EURON_API_KEY:
    print("WARNING: EURON_API_KEY not found in .env file")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY not found in .env file")