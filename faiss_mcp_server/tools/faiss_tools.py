from server import mcp
from utils.faiss_handler import faiss_handler # Singleton instance
from typing import List, Dict, Any, Tuple
import json
import logging
import time
import os

logger = logging.getLogger(__name__)

@mcp.tool()
def search_embeddings(query_text: str, k: int = 5) -> str:
    """
    Generates an embedding for the query text using OpenAI and searches the
    FAISS index for the k nearest neighbors.

    Args:
        query_text: The query text to embed and search for.
        k: The number of nearest neighbors to retrieve (default: 5).

    Returns:
        A JSON string representing a list of results (dict with id, distance, metadata),
        or a JSON string with an error message.
    """
    try:
        # Input validation
        if not query_text or not isinstance(query_text, str):
            return json.dumps([{"error": "Query text must be a non-empty string"}])
        
        query_text = query_text.strip()
        if len(query_text) == 0:
            return json.dumps([{"error": "Query text cannot be empty after stripping whitespace"}])
        
        if not isinstance(k, int) or k <= 0:
            return json.dumps([{"error": "k must be a positive integer"}])
        
        if k > 100:  # Reasonable limit
            logger.warning(f"k value ({k}) is very high, limiting to 100")
            k = 100
        
        logger.info(f"Attempting to search for {k} nearest neighbors for query text: '{query_text[:50]}...'" )
        # Pass query_text to the handler
        success, results = faiss_handler.search_embeddings(query_text=query_text, k=k)

        if success:
            logger.info(f"Search successful, found {len(results)} results for query text.")
            # Ensure results are JSON serializable (distances are floats, IDs are ints)
            # The handler already prepares this format
            return json.dumps(results)
        else:
            # Results contain the error dict in case of failure in the handler
            logger.error(f"Search failed: {results}")
            return json.dumps(results)
    except Exception as e:
        logger.error(f"Unexpected error in search_embeddings tool: {e}", exc_info=True)
        return json.dumps([{"error": f"An unexpected error occurred: {e}"}])

@mcp.tool()
def get_index_status() -> str:
    """
    Retrieves the current status of the FAISS index.

    Returns:
        A JSON string containing the index status (e.g., number of vectors, dimension).
    """
    try:
        if faiss_handler.index:
            status = {
                "initialized": True,
                "dimension": faiss_handler.index.d,
                "vector_count": faiss_handler.index.ntotal,
                "metadata_count": len(faiss_handler.metadata_store),
                "index_file": faiss_handler.index_file,
                "metadata_file": faiss_handler.metadata_file
            }
        else:
            status = {"initialized": False}
        logger.info(f"Reporting index status: {status}")
        return json.dumps(status)
    except Exception as e:
        logger.error(f"Error retrieving index status: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to retrieve status: {e}"}) 