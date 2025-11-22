import os
import sys
import argparse
from datacom_ai.rag.pipeline import RAGPipeline
from datacom_ai.utils.logger import logger

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

def main():
    parser = argparse.ArgumentParser(description="Index documents for RAG pipeline.")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), "../data/raw"),
        help="Path to the directory containing documents to index."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return

    logger.info(f"Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    logger.info(f"Indexing documents from: {data_dir}")
    try:
        rag.index_directory(data_dir)
        logger.success("Indexing completed successfully!")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        logger.exception("Exception during indexing")

if __name__ == "__main__":
    main()
