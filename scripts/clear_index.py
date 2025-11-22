import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qdrant_client import QdrantClient
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger

def clear_index():
    """
    Clears the RAG index by deleting the Qdrant collection.
    """
    logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )
    
    collection_name = settings.QDRANT_COLLECTION_NAME
    
    if client.collection_exists(collection_name):
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Successfully deleted collection '{collection_name}'.")
    else:
        logger.info(f"Collection '{collection_name}' does not exist. Nothing to delete.")

if __name__ == "__main__":
    try:
        clear_index()
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        sys.exit(1)
