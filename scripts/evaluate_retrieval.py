#!/usr/bin/env python3
"""
Evaluate RAG retrieval system using DeepEval metrics.

This script evaluates only the retrieval component of the RAG pipeline
against a synthetic dataset. Documents should already be indexed.

For Azure OpenAI configuration with deepeval, set the following environment variables:
- OPENAI_API_KEY: Your Azure OpenAI API key
- AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
- AZURE_OPENAI_DEPLOYMENT: Your Azure OpenAI deployment name
- AZURE_OPENAI_API_VERSION: Your Azure OpenAI API version

Alternatively, deepeval will use OpenAI's API by default if OPENAI_API_KEY is set.
To use Azure OpenAI with deepeval, you may need to configure it separately.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import AzureOpenAIModel

from datacom_ai.rag.pipeline import RAGPipeline
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger, setup_logging
from datacom_ai.utils.structured_logging import log_evaluation_run


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the evaluation dataset from JSON file."""
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test cases")
    return data


def retrieve_documents(retriever, query: str, k: int = 4) -> List[str]:
    """
    Retrieve documents for a given query.
    
    Args:
        retriever: The retriever instance
        query: The query string
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved document contents as strings
    """
    try:
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Error retrieving documents for query '{query}': {e}")
        return []


def create_test_cases(
    dataset: List[Dict[str, Any]], 
    retriever,
    k: int = 4
) -> List[LLMTestCase]:
    """
    Create DeepEval test cases from dataset.
    
    Args:
        dataset: List of dataset entries
        retriever: The retriever instance
        k: Number of documents to retrieve
        
    Returns:
        List of LLMTestCase objects
    """
    test_cases = []
    
    for i, entry in enumerate(dataset):
        logger.debug(f"Processing test case {i+1}/{len(dataset)}")
        
        # Get the query
        query = entry.get("input", "")
        if not query:
            logger.warning(f"Skipping entry {i+1}: missing 'input' field")
            continue
        
        # Get expected output (ground truth)
        expected_output = entry.get("expected_output", "")
        if not expected_output:
            logger.warning(f"Skipping entry {i+1}: missing 'expected_output' field")
            continue
        
        # Retrieve documents using the retriever
        retrieval_context = retrieve_documents(retriever, query, k)
        
        if not retrieval_context:
            logger.warning(f"No documents retrieved for query {i+1}: '{query[:50]}...'")
            # Still create test case with empty retrieval_context for evaluation
        
        # Create test case
        # Note: actual_output is not needed for retrieval-only evaluation
        # but DeepEval requires it, so we'll use an empty string
        test_case = LLMTestCase(
            input=query,
            actual_output="",  # Not used for retrieval evaluation
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        
        test_cases.append(test_case)
    
    logger.info(f"Created {len(test_cases)} test cases")
    return test_cases


def main():
    """Main evaluation function."""
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/evaluation.log")
    log_format = os.getenv("LOG_FORMAT", "both")  # json, text, or both
    json_log_file = os.getenv("LOG_JSON_FILE", "logs/evaluation.jsonl")
    
    setup_logging(
        level=log_level, 
        log_file=log_file if log_file else None,
        log_format=log_format,
        json_log_file=json_log_file if log_format in ("json", "both") else None
    )
    
    logger.info("Starting RAG retrieval evaluation")
    
    # Get dataset path
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "../data/processed/synthetic_dataset_ragas.json"
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Load dataset
    dataset = load_dataset(str(dataset_path))
    
    if not dataset:
        logger.error("Dataset is empty")
        sys.exit(1)
    
    # Get k value from environment or use default
    k = int(os.getenv("RETRIEVAL_K", "4"))
    logger.info(f"Using k={k} for retrieval")
    
    # Initialize RAG pipeline to get retriever
    logger.info("Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline()
        retriever = rag_pipeline.retriever.get_retriever(k=k)
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create test cases
    logger.info("Creating test cases...")
    test_cases = create_test_cases(dataset, retriever, k=k)
    
    if not test_cases:
        logger.error("No test cases created")
        sys.exit(1)
    
    # Initialize metrics
    logger.info("Initializing evaluation metrics...")
    
    # Configure Azure OpenAI for DeepEval if using Azure
    use_azure = os.getenv("DEEPEVAL_USE_AZURE", "true").lower() == "true"
    evaluation_model = None
    
    if use_azure:
        try:
            logger.info("Configuring DeepEval to use Azure OpenAI...")
            # Normalize endpoint (remove trailing slash if present)
            endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
            if not endpoint.endswith("/"):
                endpoint += "/"
            
            # Use correct parameter names according to DeepEval documentation
            # https://deepeval.com/integrations/models/azure-openai
            evaluation_model = AzureOpenAIModel(
                model_name=settings.MODEL_NAME,  # e.g., "gpt-4o"
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,  # Deployment name
                azure_openai_api_key=settings.OPENAI_API_KEY,  # API key
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,  # API version
                azure_endpoint=endpoint,  # Endpoint URL
                temperature=0,  # Default temperature
            )
            logger.info("Azure OpenAI configured for DeepEval metrics")
        except Exception as e:
            logger.warning(f"Failed to configure Azure OpenAI for DeepEval: {e}")
            logger.warning("Falling back to default OpenAI API (requires OPENAI_API_KEY)")
            import traceback
            traceback.print_exc()
            evaluation_model = None
    else:
        logger.info("Using default OpenAI API for DeepEval (set DEEPEVAL_USE_AZURE=false to disable Azure)")
    
    # Configure metrics with thresholds
    # These thresholds can be adjusted based on your requirements
    precision_threshold = float(os.getenv("PRECISION_THRESHOLD", "0.5"))
    recall_threshold = float(os.getenv("RECALL_THRESHOLD", "0.5"))
    relevancy_threshold = float(os.getenv("RELEVANCY_THRESHOLD", "0.5"))
    
    # Initialize metrics with optional custom model
    metric_kwargs = {
        "threshold": precision_threshold,
        "include_reason": True,
    }
    if evaluation_model:
        metric_kwargs["model"] = evaluation_model
    
    contextual_precision = ContextualPrecisionMetric(**metric_kwargs)
    
    metric_kwargs["threshold"] = recall_threshold
    contextual_recall = ContextualRecallMetric(**metric_kwargs)
    
    metric_kwargs["threshold"] = relevancy_threshold
    contextual_relevancy = ContextualRelevancyMetric(**metric_kwargs)
    
    metrics = [
        contextual_precision,
        contextual_recall,
        contextual_relevancy,
    ]
    
    # Run evaluation with rate limit handling
    logger.info("Running evaluation...")
    
    # Get batch size from environment (default: process sequentially to avoid rate limits)
    batch_size = int(os.getenv("EVAL_BATCH_SIZE", "1"))
    max_retries = int(os.getenv("EVAL_MAX_RETRIES", "3"))
    
    if batch_size == 1:
        logger.info("Processing test cases sequentially to avoid rate limits...")
    else:
        logger.info(f"Processing test cases in batches of {batch_size}...")
    
    try:
        # Process test cases in batches to avoid rate limits
        if batch_size >= len(test_cases):
            # Process all at once (original behavior)
            logger.info("Processing all test cases in parallel...")
            evaluate(
                test_cases=test_cases,
                metrics=metrics,
            )
        else:
            # Process in batches
            logger.info(f"Processing {len(test_cases)} test cases in batches of {batch_size}...")
            for i in range(0, len(test_cases), batch_size):
                batch = test_cases[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(test_cases) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} test cases)...")
                
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        evaluate(
                            test_cases=batch,
                            metrics=metrics,
                        )
                        logger.info(f"Batch {batch_num} completed successfully")
                        break
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        if "429" in error_msg or "RateLimit" in error_msg or "rate limit" in error_msg.lower():
                            if retry_count < max_retries:
                                wait_time = 10 * retry_count  # Exponential backoff: 10s, 20s, 30s
                                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                                import time
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Rate limit error after {max_retries} retries. Skipping batch {batch_num}.")
                                raise
                        else:
                            # Non-rate-limit error, don't retry
                            logger.error(f"Error in batch {batch_num}: {e}")
                            raise
                
                # Small delay between batches to avoid hitting rate limits
                if i + batch_size < len(test_cases):
                    import time
                    time.sleep(2)  # 2 second delay between batches
        
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Don't exit - try to show partial results
        logger.warning("Attempting to show partial results...")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("=== Evaluation Summary ===")
    logger.info("="*60)
    logger.info(f"Total test cases evaluated: {len(test_cases)}")
    logger.info(f"Retrieval k value: {k}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"\n--- Contextual Precision ---")
    if contextual_precision.score is not None:
        logger.info(f"  Score: {contextual_precision.score:.4f}")
        logger.info(f"  Threshold: {precision_threshold}")
        logger.info(f"  Passed: {contextual_precision.score >= precision_threshold}")
        if contextual_precision.reason:
            logger.info(f"  Reason: {contextual_precision.reason}")
    else:
        logger.info("  Score: N/A (evaluated in batches - check individual batch results above)")
    
    logger.info(f"\n--- Contextual Recall ---")
    if contextual_recall.score is not None:
        logger.info(f"  Score: {contextual_recall.score:.4f}")
        logger.info(f"  Threshold: {recall_threshold}")
        logger.info(f"  Passed: {contextual_recall.score >= recall_threshold}")
        if contextual_recall.reason:
            logger.info(f"  Reason: {contextual_recall.reason}")
    else:
        logger.info("  Score: N/A (evaluated in batches - check individual batch results above)")
    
    logger.info(f"\n--- Contextual Relevancy ---")
    if contextual_relevancy.score is not None:
        logger.info(f"  Score: {contextual_relevancy.score:.4f}")
        logger.info(f"  Threshold: {relevancy_threshold}")
        logger.info(f"  Passed: {contextual_relevancy.score >= relevancy_threshold}")
        if contextual_relevancy.reason:
            logger.info(f"  Reason: {contextual_relevancy.reason}")
    else:
        logger.info("  Score: N/A (evaluated in batches - check individual batch results above)")
    
    logger.info("="*60)
    logger.info("\n✅ Evaluation completed successfully!")
    logger.info("Note: When processing in batches, individual batch results are shown above.")
    logger.info("Each batch shows pass rates and scores for that subset of test cases.")
    logger.info("="*60)
    
    # Log evaluation results for analytics dashboard
    metrics_dict = {}
    if contextual_precision.score is not None:
        metrics_dict["contextual_precision"] = contextual_precision.score
    if contextual_recall.score is not None:
        metrics_dict["contextual_recall"] = contextual_recall.score
    if contextual_relevancy.score is not None:
        metrics_dict["contextual_relevancy"] = contextual_relevancy.score
    
    # Calculate pass rates
    if contextual_precision.score is not None:
        metrics_dict["contextual_precision_passed"] = contextual_precision.score >= precision_threshold
    if contextual_recall.score is not None:
        metrics_dict["contextual_recall_passed"] = contextual_recall.score >= recall_threshold
    if contextual_relevancy.score is not None:
        metrics_dict["contextual_relevancy_passed"] = contextual_relevancy.score >= relevancy_threshold
    
    # Add thresholds to metrics for reference
    metrics_dict["precision_threshold"] = precision_threshold
    metrics_dict["recall_threshold"] = recall_threshold
    metrics_dict["relevancy_threshold"] = relevancy_threshold
    
    if metrics_dict:
        log_evaluation_run(
            evaluation_type="rag_retrieval",
            metrics=metrics_dict,
            test_cases_count=len(test_cases),
            k_value=k,
            batch_size=batch_size
        )
        logger.info("Evaluation results logged for analytics dashboard")


if __name__ == "__main__":
    main()

