"""Pinecone vector store operations"""
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION, TOP_K
from utils.embeddings import generate_embeddings

def initialize_pinecone():
    """
    Initialize Pinecone connection and return index
    
    Returns:
        Index: Pinecone index object
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
    
    return pc.Index(PINECONE_INDEX_NAME)


def store_chunks(index, chunks, source_name):
    """
    Store text chunks in Pinecone with embeddings
    
    Args:
        index: Pinecone index object
        chunks (list): List of text chunks
        source_name (str): Source document name
        
    Returns:
        int: Number of chunks successfully stored
    """
    stored_count = 0
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        if embedding:
            index.upsert(vectors=[{
                "id": f"{source_name}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": source_name,
                    "chunk_id": i
                }
            }])
            stored_count += 1
    
    return stored_count


def search_similar_chunks(index, query, top_k=None):
    """
    Search for similar chunks in Pinecone
    
    Args:
        index: Pinecone index object
        query (str): Search query
        top_k (int, optional): Number of results. Defaults to config value.
        
    Returns:
        dict: Pinecone query results with matches
    """
    query_embedding = generate_embeddings(query)
    
    if not query_embedding:
        return {"matches": []}
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k or TOP_K,
        include_metadata=True
    )
    
    return results


def get_index_stats(index):
    """
    Get statistics about the Pinecone index
    
    Args:
        index: Pinecone index object
        
    Returns:
        dict: Index statistics
    """
    return index.describe_index_stats()