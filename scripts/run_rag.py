import os
import sys
from datacom_ai.rag.pipeline import RAGPipeline
from datacom_ai.utils.logger import logger

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

def main():
    # 1. Initialize Pipeline
    rag = RAGPipeline()
    
    # 2. Index Data (Optional - check if collection exists or just force index for demo)
    # For this script, we'll assume we want to index the 'data' directory if it exists
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    if os.path.exists(data_dir):
        logger.info(f"Indexing documents from {data_dir}...")
        rag.index_directory(data_dir)
    else:
        logger.warning(f"Data directory {data_dir} not found. Skipping indexing.")

    # 3. Query
    query = "What is the main topic of the documents?"
    logger.info(f"Querying: {query}")
    
    try:
        result = rag.query(query)
        print("\n=== RAG Result ===")
        print(f"Question: {result['input']}")
        print(f"Answer: {result['answer']}")
        print("\n=== Source Documents ===")
        for i, doc in enumerate(result["context"]):
            print(f"[{i+1}] {doc.metadata.get('source', 'Unknown')} (Author: {doc.metadata.get('author', 'Unknown')})")
            print(f"    Snippet: {doc.page_content[:100]}...")
    except Exception as e:
        logger.error(f"Error during query: {e}")

if __name__ == "__main__":
    main()
