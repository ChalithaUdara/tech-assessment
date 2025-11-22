import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from datacom_ai.utils.logger import logger

class DocumentLoader:
    """Loads documents from a directory and extracts metadata."""

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load_documents(self) -> List[Document]:
        """
        Load all text documents from the directory.
        
        Returns:
            List of LangChain Documents with metadata.
        """
        documents = []
        if not os.path.exists(self.directory_path):
            logger.warning(f"Directory not found: {self.directory_path}")
            return documents

        for filename in os.listdir(self.directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory_path, filename)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    # Enrich metadata
                    for doc in docs:
                        metadata = self._extract_metadata(filename)
                        doc.metadata.update(metadata)
                        documents.append(doc)
                        
                    logger.info(f"Loaded {len(docs)} documents from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        return documents

    def _extract_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from filename.
        Expected format: "Author - Novel Name.txt" or similar.
        If format doesn't match, uses filename as title.
        """
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Try to split by " - "
        parts = name.split(" - ")
        
        if len(parts) >= 2:
            author = parts[0].strip()
            novel = " - ".join(parts[1:]).strip()
            return {
                "author": author,
                "novel": novel,
                "source": filename
            }
        else:
            return {
                "author": "Unknown",
                "novel": name,
                "source": filename
            }
