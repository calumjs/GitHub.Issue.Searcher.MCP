import faiss
import numpy as np
import os
import json
import uuid
import logging
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError # Import OpenAI
import time
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

# Configuration with validation
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_db")
try:
    FAISS_DIMENSION = int(os.getenv("FAISS_DIMENSION", "1536"))
    if FAISS_DIMENSION <= 0:
        raise ValueError("FAISS_DIMENSION must be positive")
except ValueError as e:
    logger.error(f"Invalid FAISS_DIMENSION configuration: {e}")
    FAISS_DIMENSION = 1536  # Safe default

INDEX_FILE = f"{FAISS_INDEX_PATH}.faiss"
METADATA_FILE = f"{FAISS_INDEX_PATH}.meta"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Validate embedding model
SUPPORTED_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
if OPENAI_EMBEDDING_MODEL not in SUPPORTED_MODELS:
    logger.warning(f"Unsupported embedding model '{OPENAI_EMBEDDING_MODEL}'. Using default.")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Ensure the data directory exists
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

class FaissHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaissHandler, cls).__new__(cls)
            # Initialize only once
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, index_dim: int = FAISS_DIMENSION, index_file: str = INDEX_FILE, metadata_file: str = METADATA_FILE):
        if self._initialized:
            return
        self._initialized = True

        self.index_dim = index_dim
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index: Optional[faiss.IndexIDMap] = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {} # Map FAISS ID (int) to metadata

        # Initialize tokenizer for accurate token counting
        try:
            # Use the appropriate tokenizer for the embedding model
            if OPENAI_EMBEDDING_MODEL in ["text-embedding-3-small", "text-embedding-3-large"]:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5 tokenizer
            else:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default fallback
            logger.info(f"Tokenizer initialized for model: {OPENAI_EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}. Falling back to estimation.")
            self.tokenizer = None

        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable not set.")
            # Decide how to handle this: raise error, run without OpenAI, etc.
            # For now, log error and client will be None
            self.openai_client = None
        else:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info(f"OpenAI client initialized. Using model: {OPENAI_EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
                self.openai_client = None

        self.load_or_initialize_index()

    def _count_tokens(self, text: str) -> int:
        """
        Accurately count the number of tokens in a text string using tiktoken.
        Falls back to estimation if tokenizer is not available.
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Error counting tokens with tiktoken: {e}. Using estimation.")
        
        # Fallback to estimation if tokenizer fails
        return len(text) // 3  # Conservative estimate: ~3 chars per token

    def _truncate_text_smartly(self, text: str, max_tokens: int = 8000) -> str:
        """
        Intelligently truncate text to fit within token limits while preserving important content.
        
        Args:
            text: The input text to truncate
            max_tokens: Maximum number of tokens allowed (default: 8000 to leave buffer)
            
        Returns:
            Truncated text that fits within the token limit
        """
        current_tokens = self._count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        logger.info(f"Text too long ({current_tokens} tokens), truncating to {max_tokens} tokens")
        
        # Binary search approach for precise truncation
        left, right = 0, len(text)
        best_text = text[:max_tokens * 3]  # Initial rough estimate
        
        # Try to find the longest text that fits within token limit
        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid]
            
            if self._count_tokens(candidate) <= max_tokens:
                best_text = candidate
                left = mid
            else:
                right = mid - 1
        
        # Now try to truncate at natural boundaries
        truncated = best_text
        
        # Try to end at a paragraph break
        last_paragraph = truncated.rfind('\n\n')
        if last_paragraph > len(truncated) * 0.7:  # If we can keep at least 70% of content
            candidate = truncated[:last_paragraph] + "\n\n[Content truncated for embedding...]"
            if self._count_tokens(candidate) <= max_tokens:
                return candidate
        
        # Try to end at a sentence
        last_sentence = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )
        if last_sentence > len(truncated) * 0.7:
            candidate = truncated[:last_sentence + 1] + " [Content truncated for embedding...]"
            if self._count_tokens(candidate) <= max_tokens:
                return candidate
        
        # Try to end at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > len(truncated) * 0.8:
            candidate = truncated[:last_space] + " [Content truncated for embedding...]"
            if self._count_tokens(candidate) <= max_tokens:
                return candidate
        
        # Fallback: use the binary search result with truncation notice
        final_text = truncated + " [Content truncated for embedding...]"
        
        # Make sure the final text with notice doesn't exceed limit
        while self._count_tokens(final_text) > max_tokens and len(truncated) > 100:
            truncated = truncated[:-100]  # Remove 100 chars at a time
            final_text = truncated + " [Content truncated for embedding...]"
        
        return final_text

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for the given text using OpenAI with retry logic and smart truncation."""
        if not self.openai_client:
            logger.error("OpenAI client not available.")
            return None
        try:
            # Ensure text is not empty or just whitespace
            text = text.strip()
            if not text:
                 logger.warning("Attempted to embed empty string.")
                 return None

            # Smart truncation to handle long texts
            text = self._truncate_text_smartly(text)

            # Retry logic for API calls
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        input=[text.replace("\n", " ")], # API best practice: replace newlines
                        model=OPENAI_EMBEDDING_MODEL,
                        dimensions=self.index_dim # Explicitly set dimensions if using v3 models
                    )
                    embedding = response.data[0].embedding
                    # Verify dimension just in case
                    if len(embedding) != self.index_dim:
                        logger.error(f"OpenAI returned embedding with wrong dimension! Expected {self.index_dim}, got {len(embedding)}")
                        return None
                    return embedding
                    
                except OpenAIError as e:
                    # Check if it's still a token limit error after truncation
                    if "maximum context length" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Still too long after truncation, reducing further (attempt {attempt + 1})")
                        # Reduce text further
                        current_tokens = self._count_tokens(text)
                        new_max = int(current_tokens * 0.7)  # Reduce by 30%
                        text = self._truncate_text_smartly(text, new_max)
                        continue
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"OpenAI API error after {max_retries} attempts: {e}", exc_info=True)
                        return None
                        
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}", exc_info=True)
            return None

    def load_or_initialize_index(self):
        """Loads the index and metadata from files, or initializes new ones if they don't exist."""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                logger.info(f"Loading FAISS index from: {self.index_file}")
                cpu_index = faiss.read_index(self.index_file)
                if not isinstance(cpu_index, faiss.IndexIDMap):
                     # If the loaded index isn't IndexIDMap (e.g. from older version), wrap it
                     logger.warning("Loaded index is not IndexIDMap, wrapping it.")
                     base_index = cpu_index
                     self.index = faiss.IndexIDMap(base_index)
                     # Note: This assumes sequential IDs were used before.
                     # Migration logic might be needed for existing non-IDMap indices
                     # with custom ID schemes. For simplicity, we'll assume new or compatible indices.
                else:
                    self.index = cpu_index

                # Check dimension consistency - CRUCIAL
                if self.index.d != self.index_dim:
                     logger.error(f"Dimension mismatch! Index file dimension ({self.index.d}) does not match configured FAISS_DIMENSION ({self.index_dim}).")
                     logger.warning("Re-initializing index due to dimension mismatch.")
                     # This will discard the old index and metadata
                     self._initialize_new_index()
                     return # Stop loading further as we've re-initialized
                else:
                    logger.info(f"Index dimension {self.index.d} matches configured dimension {self.index_dim}.")

                logger.info(f"Loading metadata from: {self.metadata_file}")
                with open(self.metadata_file, 'r') as f:
                    # Load metadata, converting string keys back to int
                    raw_meta = json.load(f)
                    self.metadata_store = {int(k): v for k, v in raw_meta.items()}
                logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata_store)} metadata entries.")

            except Exception as e:
                logger.error(f"Error loading index/metadata: {e}. Re-initializing.", exc_info=True)
                self._initialize_new_index()
        else:
            logger.info("Index or metadata file not found. Initializing new index.")
            self._initialize_new_index()

    def _initialize_new_index(self):
        """Initializes a new FAISS index (IndexFlatL2 wrapped in IndexIDMap) and metadata store."""
        logger.info(f"Creating new FAISS index with dimension {self.index_dim}")
        base_index = faiss.IndexFlatL2(self.index_dim) # Use configured dimension
        self.index = faiss.IndexIDMap(base_index)
        self.metadata_store = {}
        self._save_index() # Save the empty index and metadata immediately

    def _save_index(self) -> bool:
        """Saves the current index and metadata to their respective files."""
        if self.index is None:
             logger.error("Cannot save, index is not initialized.")
             return False # Indicate failure
        try:
            logger.info(f"Saving FAISS index ({self.index.ntotal} vectors) to: {self.index_file}")
            cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'is_trained') and faiss.get_num_gpus() > 0 and hasattr(faiss, 'GpuResources') else self.index
            faiss.write_index(cpu_index, self.index_file)

            logger.info(f"Saving metadata ({len(self.metadata_store)} entries) to: {self.metadata_file}")
            serializable_meta = {str(k): v for k, v in self.metadata_store.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_meta, f, indent=4)
            logger.info("Save complete.")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Error saving index/metadata: {e}", exc_info=True)
            return False # Indicate failure

    # Public method to trigger save
    def save(self) -> bool:
        """Public method to trigger saving the index and metadata."""
        return self._save_index()

    def clear_index(self) -> bool:
        """Clears the index and metadata store by re-initializing them and saving the empty state."""
        logger.warning("Clearing FAISS index and metadata store.")
        try:
            self._initialize_new_index() # This creates new empty index/metadata and saves them
            logger.info("FAISS index and metadata cleared successfully.")
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {e}", exc_info=True)
            return False

    def add_embedding(self, text: str, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Generates embedding for the text and adds it with metadata to the index. Does NOT save automatically."""
        if self.index is None:
            return False, "Index not initialized."
        if not self.openai_client:
            return False, "OpenAI client not available."

        vector = self._get_embedding(text)
        if vector is None:
            return False, "Failed to generate embedding for the provided text."

        if len(vector) != self.index.d:
             logger.error(f"Internal dimension mismatch after embedding. Expected {self.index.d}, got {len(vector)}.")
             return False, f"Internal error: Vector dimension mismatch. Expected {self.index.d}, got {len(vector)}."

        try:
            # More efficient ID generation using timestamp + random component
            timestamp = int(time.time() * 1000000)  # microseconds
            random_part = np.random.randint(0, 999999)
            new_id = np.int64(timestamp + random_part)
            
            # Fallback to original method if collision (very unlikely)
            collision_attempts = 0
            while new_id in self.metadata_store and collision_attempts < 10:
                new_id = np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64)
                collision_attempts += 1
            
            if collision_attempts >= 10:
                logger.error("Failed to generate unique ID after 10 attempts")
                return False, "Failed to generate unique ID for embedding"

            vector_np = np.array([vector], dtype=np.float32)
            ids_np = np.array([new_id], dtype=np.int64)

            self.index.add_with_ids(vector_np, ids_np)
            self.metadata_store[new_id] = metadata
            # Log addition, but do not save here
            logger.debug(f"Added text embedding with ID {new_id} to memory. Index size: {self.index.ntotal}")

            # REMOVED: self.save_index()
            return True, f"Text embedding added successfully to memory with ID {new_id}. Call save() to persist."

        except Exception as e:
            logger.error(f"Error adding embedding to FAISS in memory: {e}", exc_info=True)
            return False, f"Failed to add embedding to FAISS index in memory: {e}"

    # Modified to accept query_text instead of query_vector
    def search_embeddings(self, query_text: str, k: int = 5) -> Tuple[bool, List[Dict[str, Any]]]:
        """Generates embedding for the query text and searches the index."""
        if self.index is None:
            return False, [{"error": "Index not initialized."}]
        if not self.openai_client:
             return False, [{"error": "OpenAI client not available."}]
        if self.index.ntotal == 0:
            return True, [] # Successful search, but no results as index is empty

        # Generate query embedding
        query_vector = self._get_embedding(query_text)
        if query_vector is None:
            return False, [{"error": "Failed to generate query embedding."}]

        # Dimension check should be handled by _get_embedding
        if len(query_vector) != self.index.d:
             logger.error(f"Internal query dimension mismatch after embedding. Expected {self.index.d}, got {len(query_vector)}.")
             return False, [{"error": f"Internal error: Query vector dimension mismatch."}]

        try:
            query_np = np.array([query_vector], dtype=np.float32)
            distances, ids = self.index.search(query_np, k)

            results = []
            for i in range(len(ids[0])):
                faiss_id = ids[0][i]
                if faiss_id == -1: continue
                distance = distances[0][i]
                metadata = self.metadata_store.get(int(faiss_id), {"error": "Metadata not found for ID"})
                results.append({
                    "id": int(faiss_id),
                    "distance": float(distance),
                    "metadata": metadata
                })

            logger.info(f"Search found {len(results)} results for k={k} from query text.")
            return True, results

        except Exception as e:
            logger.error(f"Error searching FAISS embeddings: {e}", exc_info=True)
            return False, [{"error": f"Failed to search FAISS embeddings: {e}"}]

# Singleton instance for easy access
faiss_handler = FaissHandler() 