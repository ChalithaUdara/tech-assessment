"""
Generate synthetic Q&A dataset using Ragas framework.

This script uses Ragas to generate high-quality test datasets from multiple
Arthur Conan Doyle novels. It generates both single-hop and multi-hop questions
for comprehensive RAG evaluation.

Features:
- Loads multiple novels from data/raw directory
- Generates single-hop questions (direct answers from one document)
- Generates multi-hop questions (requires information from multiple documents)
- Uses existing LLM and embedding configuration
- Exports dataset in compatible format for evaluation

Usage:
    uv run scripts/generate_dataset_ragas.py
"""
import os
import json
import glob
from typing import List
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper, BaseRagasEmbeddings
from ragas import RunConfig
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

from datacom_ai.clients.llm_client import create_llm_client
from datacom_ai.rag.factories import EmbeddingFactory
from datacom_ai.rag.embeddings import CustomFastEmbedEmbeddings
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger


def load_arthur_conan_doyle_novels(data_dir: str = "data/raw", limit: int = None) -> List[Document]:
    """
    Load all Arthur Conan Doyle novels from the data directory.
    
    Args:
        data_dir: Directory containing the text files
        limit: Optional limit on number of novels to load (for testing)
        
    Returns:
        List of LangChain Document objects with metadata
    """
    logger.info(f"Loading Arthur Conan Doyle novels from {data_dir}...")
    
    # Find all Arthur Conan Doyle novels
    pattern = os.path.join(data_dir, "Arthur_Conan_Doyle*.txt")
    novel_files = glob.glob(pattern)
    
    if not novel_files:
        raise FileNotFoundError(
            f"No Arthur Conan Doyle novels found in {data_dir}. "
            f"Expected files matching pattern: {pattern}"
        )
    
    # Limit number of files if specified (for testing)
    if limit:
        novel_files = novel_files[:limit]
        logger.info(f"Found {len(glob.glob(pattern))} novels, loading {limit} for testing")
    else:
        logger.info(f"Found {len(novel_files)} novels")
    
    documents = []
    for file_path in novel_files:
        try:
            # Extract book title from filename
            filename = Path(file_path).stem
            # Remove "Arthur_Conan_Doyle-" or "Arthur Conan Doyle - " prefix
            book_title = filename.replace("Arthur_Conan_Doyle-", "").replace("Arthur Conan Doyle - ", "")
            
            # Load document
            loader = TextLoader(file_path, encoding='utf-8')
            doc = loader.load()[0]
            
            # Add metadata
            doc.metadata.update({
                "source": file_path,
                "author": "Arthur Conan Doyle",
                "book_title": book_title,
                "file_name": os.path.basename(file_path)
            })
            
            documents.append(doc)
            logger.info(f"  ✓ Loaded: {book_title} ({len(doc.page_content)} chars)")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not documents:
        raise ValueError("No documents were successfully loaded")
    
    logger.success(f"Successfully loaded {len(documents)} novels")
    return documents


class FastEmbedRagasWrapper(BaseRagasEmbeddings):
    """
    Custom wrapper for FastEmbed embeddings that properly provides model name string.
    
    This fixes the issue where LangchainEmbeddingsWrapper tries to pass the
    TextEmbedding object directly, but Ragas expects a string model name.
    """
    
    def __init__(self, fastembed_embeddings: CustomFastEmbedEmbeddings):
        """
        Initialize wrapper with FastEmbed embeddings.
        
        Args:
            fastembed_embeddings: CustomFastEmbedEmbeddings instance
        """
        super().__init__()
        self._embeddings = fastembed_embeddings
        # Store model as string (Ragas expects this for telemetry)
        # Use settings since CustomFastEmbedEmbeddings doesn't store model_name
        self.model = getattr(settings, 'FASTEMBED_MODEL_NAME', 'BAAI/bge-small-en-v1.5')
        self.model_name = self.model  # Also store as model_name for compatibility
    
    # Required abstract methods from BaseRagasEmbeddings
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed single query text."""
        return self._embeddings.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed multiple document texts."""
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Sync embed single query text."""
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sync embed multiple document texts."""
        return self._embeddings.embed_documents(texts)
    
    # Additional methods that may be used
    async def aembed_text(self, text: str) -> List[float]:
        """Async embed single text (alias for aembed_query)."""
        return await self.aembed_query(text)
    
    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Async embed multiple texts (alias for aembed_documents)."""
        return await self.aembed_documents(texts)
    
    def embed_text(self, text: str) -> List[float]:
        """Sync embed single text (alias for embed_query)."""
        return self.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Sync embed multiple texts (alias for embed_documents)."""
        return self.embed_documents(texts)


def setup_ragas_components():
    """
    Setup LLM and embeddings for Ragas, wrapping existing LangChain components.
    
    Returns:
        Tuple of (generator_llm, generator_embeddings)
    """
    logger.info("Setting up Ragas components...")
    
    # Get existing LLM client (Azure OpenAI via LangChain)
    langchain_llm = create_llm_client()
    generator_llm = LangchainLLMWrapper(langchain_llm)
    logger.info("✓ LLM wrapped for Ragas")
    
    # Get existing embeddings (FastEmbed or Azure OpenAI)
    langchain_embeddings = EmbeddingFactory.create()
    
    # Use appropriate wrapper based on embedding type
    if isinstance(langchain_embeddings, CustomFastEmbedEmbeddings):
        # Use custom wrapper for FastEmbed to fix model name issue
        logger.info("Using custom FastEmbed wrapper for Ragas")
        generator_embeddings = FastEmbedRagasWrapper(langchain_embeddings)
        logger.info(f"✓ FastEmbed wrapped for Ragas (model: {generator_embeddings.model_name})")
    else:
        # Use standard wrapper for Azure OpenAI embeddings
        generator_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        logger.info("✓ Azure OpenAI embeddings wrapped for Ragas")
    
    return generator_llm, generator_embeddings


def create_query_distribution(generator_llm, max_hops: int = 3, optimize_for_speed: bool = False):
    """
    Create a query distribution that includes both single-hop and multi-hop questions.
    
    Distribution (default):
    - 50% Single-hop specific queries (direct answers from one document)
    - 25% Multi-hop abstract queries (requires reasoning across documents, max 3 hops)
    - 25% Multi-hop specific queries (requires specific facts from multiple documents, max 3 hops)
    
    Distribution (speed-optimized):
    - 75% Single-hop specific queries (faster)
    - 12.5% Multi-hop abstract queries
    - 12.5% Multi-hop specific queries
    
    Args:
        generator_llm: The LLM wrapper for Ragas
        max_hops: Maximum number of hops for multi-hop queries (default: 3)
        optimize_for_speed: If True, use speed-optimized distribution (more single-hop)
        
    Returns:
        List of tuples (synthesizer, weight)
    """
    if optimize_for_speed:
        logger.info(f"Creating SPEED-OPTIMIZED query distribution (max {max_hops} hops)...")
        single_hop_weight = 0.75
        multi_hop_weight = 0.125
    else:
        logger.info(f"Creating query distribution with single-hop and multi-hop questions (max {max_hops} hops)...")
        single_hop_weight = 0.5
        multi_hop_weight = 0.25
    
    # Create multi-hop synthesizers with custom prompts that limit to max_hops
    from ragas.testset.synthesizers.multi_hop.prompts import QueryAnswerGenerationPrompt
    
    # Custom prompt class that limits to max_hops
    class LimitedHopsQueryPrompt(QueryAnswerGenerationPrompt):
        """Custom prompt that limits multi-hop queries to max_hops."""
        def __init__(self, max_hops: int = 3, **kwargs):
            super().__init__(**kwargs)
            # Custom instruction that explicitly limits hops
            custom_instruction = f"""Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) and the provided context. The themes represent a set of phrases either extracted or generated from the context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query explicitly incorporates these themes.

### Instructions:
1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining information from multiple segments (e.g., `<1-hop>`, `<2-hop>`, `<3-hop>`). Ensure the query explicitly incorporates one or more themes and reflects their relevance to the context.
2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to the query. Avoid adding information that is not directly present or inferable from the given context.
3. **Multi-Hop Context Tags**:
   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, `<3-hop>`, etc.
   - Ensure the query uses information from at least two segments and connects them meaningfully.
   - **IMPORTANT: Do not exceed {max_hops} hops. Maximum allowed hops: {max_hops}.**"""
            # Set the instruction using object.__setattr__ to bypass Pydantic validation
            object.__setattr__(self, 'instruction', custom_instruction)
    
    # Create custom prompts for multi-hop synthesizers
    multi_hop_prompt = LimitedHopsQueryPrompt(max_hops=max_hops, language="english")
    
    # Create synthesizers with custom prompts
    multi_hop_abstract = MultiHopAbstractQuerySynthesizer(
        llm=generator_llm,
        generate_query_reference_prompt=multi_hop_prompt
    )
    
    multi_hop_specific = MultiHopSpecificQuerySynthesizer(
        llm=generator_llm,
        generate_query_reference_prompt=multi_hop_prompt
    )
    
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), single_hop_weight),
        (multi_hop_abstract, multi_hop_weight),
        (multi_hop_specific, multi_hop_weight),
    ]
    
    logger.info(f"  ✓ {single_hop_weight*100:.0f}% Single-hop specific queries")
    logger.info(f"  ✓ {multi_hop_weight*100:.1f}% Multi-hop abstract queries (max {max_hops} hops)")
    logger.info(f"  ✓ {multi_hop_weight*100:.1f}% Multi-hop specific queries (max {max_hops} hops)")
    
    return distribution


def save_partial_results(output_data: List[dict], output_file: str, error_message: str = None):
    """
    Save partial results to file, even if generation was incomplete.
    
    Args:
        output_data: List of Q&A pairs to save
        output_file: Path to save the data
        error_message: Optional error message to log
    """
    if not output_data:
        logger.warning("No data to save.")
        return
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Add metadata about partial save
        metadata = {
            "partial": True,
            "total_pairs": len(output_data),
            "error": error_message
        }
        
        # Save with metadata comment in JSON
        output_with_metadata = {
            "_metadata": metadata,
            "data": output_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_with_metadata, f, indent=2, ensure_ascii=False)
        
        logger.warning(f"⚠ Saved {len(output_data)} partial Q&A pairs to {output_file}")
        if error_message:
            logger.warning(f"  Error: {error_message}")
    except Exception as e:
        logger.error(f"Failed to save partial results: {e}")


def convert_ragas_dataset_to_format(dataset, output_file: str, is_partial: bool = False):
    """
    Convert Ragas dataset to the format expected by the evaluation pipeline.
    
    Ragas format has: question, contexts, ground_truth
    Our format needs: input, retrieval_context, expected_output, context
    
    Args:
        dataset: Ragas dataset object
        output_file: Path to save the converted dataset
    """
    logger.info(f"Converting Ragas dataset to evaluation format...")
    
    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()
    
    # Debug: Print column names to understand the structure
    logger.info(f"Dataset columns: {list(df.columns)}")
    logger.info(f"Dataset shape: {df.shape}")
    if len(df) > 0:
        logger.info(f"Sample row keys: {list(df.iloc[0].keys())}")
        logger.info(f"Sample row data: {df.iloc[0].to_dict()}")
    
    output_data = []
    for idx, row in df.iterrows():
        # Ragas provides different column names - check what's actually available
        # Common Ragas column names: 'question', 'contexts', 'ground_truth'
        # But might also be: 'user_input', 'reference_contexts', 'reference', etc.
        
        # Try different possible column names
        question = (
            row.get('question', '') or 
            row.get('user_input', '') or 
            row.get('query', '') or
            str(row.get('input', ''))
        )
        
        # Get contexts - could be 'contexts', 'reference_contexts', 'context'
        contexts = (
            row.get('contexts', []) or 
            row.get('reference_contexts', []) or 
            row.get('context', []) or
            []
        )
        
        # Get ground truth - could be 'ground_truth', 'reference', 'answer'
        ground_truth = (
            row.get('ground_truth', '') or 
            row.get('reference', '') or 
            row.get('answer', '') or
            str(row.get('expected_output', ''))
        )
        
        # Handle contexts
        if isinstance(contexts, list) and len(contexts) > 0:
            # Use first context as primary retrieval_context (string)
            retrieval_context = contexts[0] if isinstance(contexts[0], str) else str(contexts[0])
            # Keep all contexts in context list
            context_list = [str(c) for c in contexts if c]
        elif contexts:
            # If contexts is not a list but has value
            retrieval_context = str(contexts)
            context_list = [retrieval_context]
        else:
            retrieval_context = ""
            context_list = []
        
        # Convert to strings and clean
        question = str(question).strip() if question else ""
        ground_truth = str(ground_truth).strip() if ground_truth else ""
        
        output_data.append({
            "input": question,
            "retrieval_context": retrieval_context,
            "expected_output": ground_truth,
            "actual_output": None,  # Will be filled during evaluation
            "context": context_list,
        })
    
    # Save to file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Add metadata if partial
        if is_partial:
            output_with_metadata = {
                "_metadata": {
                    "partial": True,
                    "total_pairs": len(output_data)
                },
                "data": output_data
            }
        else:
            output_with_metadata = output_data
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_with_metadata, f, indent=2, ensure_ascii=False)
        
        if is_partial:
            logger.warning(f"⚠ Saved {len(output_data)} partial Q&A pairs to {output_file}")
        else:
            logger.success(f"✓ Saved {len(output_data)} Q&A pairs to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save dataset to {output_file}: {e}")
        raise
    
    # Print summary
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total questions: {len(output_data)}")
    logger.info(f"  Questions with contexts: {sum(1 for item in output_data if item['retrieval_context'])}")
    logger.info(f"  Questions with answers: {sum(1 for item in output_data if item['expected_output'])}")


def main():
    """Main function to generate testset using Ragas."""
    # Configuration
    DATA_DIR = "data/raw"
    TEST_MODE = False  # Set to True for quick testing (3 Q&A pairs), False for full dataset
    TESTSET_SIZE = 25  # Target number of Q&A pairs
    NOVEL_LIMIT = 2  # Limit to 2 novels
    OUTPUT_FILE = "data/processed/synthetic_dataset_ragas_test.json" if TEST_MODE else "data/processed/synthetic_dataset_ragas.json"
    
    # Performance Optimization Settings
    OPTIMIZE_FOR_SPEED = True  # Set to True for faster generation (more single-hop queries)
    MAX_WORKERS = 32  # Parallel workers (default: 16, increase for speed, decrease if memory issues)
    MAX_RETRIES = 3  # Retry attempts (default: 10, reduce for speed)
    TIMEOUT = 120  # Timeout in seconds (default: 180, reduce for faster failure)
    
    logger.info("=" * 60)
    logger.info("Ragas Testset Generation")
    logger.info(f"Configuration: {TESTSET_SIZE} Q&A pairs from {NOVEL_LIMIT} novels")
    if OPTIMIZE_FOR_SPEED:
        logger.info("⚡ Speed optimizations: ENABLED")
    logger.info("=" * 60)
    
    dataset = None
    generated_data = []
    error_occurred = False
    error_message = None
    
    try:
        # 1. Load multiple novels (limited for testing)
        logger.info("\n[Step 1/4] Loading documents...")
        try:
            documents = load_arthur_conan_doyle_novels(DATA_DIR, limit=NOVEL_LIMIT)
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
        
        # 2. Setup Ragas components
        logger.info("\n[Step 2/4] Setting up Ragas components...")
        try:
            generator_llm, generator_embeddings = setup_ragas_components()
        except Exception as e:
            logger.error(f"Failed to setup Ragas components: {e}")
            raise
        
        # 3. Create query distribution (single-hop + multi-hop, max 3 hops)
        logger.info("\n[Step 3/4] Configuring query distribution...")
        try:
            MAX_HOPS = 3  # Maximum number of hops for multi-hop queries
            query_distribution = create_query_distribution(
                generator_llm, 
                max_hops=MAX_HOPS,
                optimize_for_speed=OPTIMIZE_FOR_SPEED
            )
        except Exception as e:
            logger.error(f"Failed to create query distribution: {e}")
            raise
        
        # 4. Generate testset
        logger.info("\n[Step 4/4] Generating testset with Ragas...")
        logger.info(f"  Target size: {TESTSET_SIZE} Q&A pairs")
        logger.info(f"  This may take several minutes...")
        
        try:
            # Create optimized RunConfig for performance
            run_config = RunConfig(
                max_workers=MAX_WORKERS,  # Parallel processing
                max_retries=MAX_RETRIES,  # Fewer retries for speed
                timeout=TIMEOUT,          # Shorter timeout
                max_wait=30,              # Shorter wait time
            )
            
            if OPTIMIZE_FOR_SPEED:
                logger.info(f"  Performance optimizations enabled:")
                logger.info(f"    - Max workers: {MAX_WORKERS} (parallel processing)")
                logger.info(f"    - Max retries: {MAX_RETRIES} (faster failure)")
                logger.info(f"    - Timeout: {TIMEOUT}s (shorter timeout)")
                logger.info(f"    - Speed-optimized query distribution (more single-hop)")
            
            generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=generator_embeddings
            )
            
            # Generate testset from LangChain documents with optimized config
            dataset = generator.generate_with_langchain_docs(
                documents,
                testset_size=TESTSET_SIZE,
                query_distribution=query_distribution,
                run_config=run_config  # Use optimized RunConfig
            )
            
            if len(dataset) == 0:
                logger.warning("No test cases were generated. Check your documents and LLM configuration.")
                return
            
            logger.success(f"\n✓ Generated {len(dataset)} test cases!")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠ Generation interrupted by user (Ctrl+C)")
            error_occurred = True
            error_message = "Generation interrupted by user"
            if dataset and len(dataset) > 0:
                logger.info(f"Attempting to save {len(dataset)} generated pairs...")
                try:
                    convert_ragas_dataset_to_format(dataset, OUTPUT_FILE, is_partial=True)
                    logger.info("Partial results saved successfully.")
                except Exception as save_error:
                    logger.error(f"Failed to save partial results: {save_error}")
            return
        
        except Exception as e:
            logger.error(f"Error during testset generation: {e}")
            error_occurred = True
            error_message = str(e)
            # Try to save whatever was generated
            if dataset and len(dataset) > 0:
                logger.warning(f"Attempting to save {len(dataset)} generated pairs before error...")
                try:
                    convert_ragas_dataset_to_format(dataset, OUTPUT_FILE, is_partial=True)
                    logger.info("Partial results saved successfully.")
                except Exception as save_error:
                    logger.error(f"Failed to save partial results: {save_error}")
            raise
        
        # 5. Convert and save
        logger.info("\n[Final Step] Converting and saving dataset...")
        try:
            convert_ragas_dataset_to_format(dataset, OUTPUT_FILE, is_partial=False)
        except Exception as e:
            logger.error(f"Error converting dataset: {e}")
            error_occurred = True
            error_message = f"Conversion error: {str(e)}"
            # Try to save raw data if conversion fails
            if dataset and len(dataset) > 0:
                logger.warning("Attempting to save raw dataset...")
                try:
                    df = dataset.to_pandas()
                    raw_output_file = OUTPUT_FILE.replace('.json', '_raw.json')
                    df.to_json(raw_output_file, orient='records', indent=2)
                    logger.info(f"Raw dataset saved to {raw_output_file}")
                except Exception as save_error:
                    logger.error(f"Failed to save raw dataset: {save_error}")
            raise
        
        # 6. Display sample
        try:
            logger.info("\n" + "=" * 60)
            logger.info("Sample from generated dataset:")
            logger.info("=" * 60)
            df = dataset.to_pandas()
            for i, row in df.head(3).iterrows():
                logger.info(f"\nQuestion {i+1}:")
                question = row.get('user_input', '') or row.get('question', 'N/A')
                logger.info(f"  Q: {str(question)[:100]}...")
                contexts = row.get('reference_contexts', []) or row.get('contexts', [])
                if contexts:
                    logger.info(f"  Contexts: {len(contexts)} document(s)")
                answer = row.get('reference', '') or row.get('ground_truth', 'N/A')
                logger.info(f"  A: {str(answer)[:100]}...")
        except Exception as e:
            logger.warning(f"Error displaying samples: {e}")
            # Don't fail the whole process for display errors
        
        if not error_occurred:
            logger.success("\n" + "=" * 60)
            logger.success("Testset generation completed successfully!")
            logger.success("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Process interrupted by user (Ctrl+C)")
        if dataset and len(dataset) > 0:
            logger.info(f"Attempting to save {len(dataset)} generated pairs...")
            try:
                convert_ragas_dataset_to_format(dataset, OUTPUT_FILE, is_partial=True)
                logger.info("Partial results saved successfully.")
            except Exception as save_error:
                logger.error(f"Failed to save partial results: {save_error}")
        return
    
    except Exception as e:
        logger.error(f"\n❌ Error during testset generation: {e}")
        logger.exception("Full traceback:")
        
        # Final attempt to save any generated data
        if dataset and len(dataset) > 0:
            logger.warning(f"\n⚠ Attempting final save of {len(dataset)} generated pairs...")
            try:
                convert_ragas_dataset_to_format(dataset, OUTPUT_FILE, is_partial=True)
                logger.info("✓ Partial results saved successfully.")
            except Exception as save_error:
                logger.error(f"❌ Failed to save partial results: {save_error}")
        
        # Don't raise - exit gracefully
        logger.error("\n" + "=" * 60)
        logger.error("Generation failed, but partial results may have been saved.")
        logger.error("=" * 60)
        return


if __name__ == "__main__":
    main()

