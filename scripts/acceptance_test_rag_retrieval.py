#!/usr/bin/env python3
"""
Acceptance test for Task 3.2: High-Performance Retrieval-Augmented QA

This script evaluates both Retrieval and Generation components of the RAG pipeline
on ≥20 graded questions from data/processed/synthetic_dataset_ragas.json.

Retrieval Metrics (top-5):
- ContextualPrecisionMetric: Evaluates reranker quality and ranking order
- ContextualRecallMetric: Evaluates embedding model accuracy  
- ContextualRelevancyMetric: Evaluates chunk size and top-K parameter tuning

Generation Metrics:
- AnswerRelevancyMetric: Evaluates prompt template quality
- FaithfulnessMetric: Evaluates LLM output quality (no hallucinations/contradictions)

Acceptance Criteria: Based on Answer Relevancy metric only

Based on: https://deepeval.com/guides/guides-rag-evaluation
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import AzureOpenAIModel

from datacom_ai.rag.pipeline import RAGPipeline
from datacom_ai.config.settings import settings
from datacom_ai.utils.logger import logger, setup_logging


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the evaluation dataset from JSON file."""
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test cases")
    return data


def create_test_cases(
    dataset: List[Dict[str, Any]], 
    rag_pipeline: RAGPipeline,
    k: int = 5
) -> List[LLMTestCase]:
    """
    Create DeepEval test cases with both retrieval context and actual RAG output.
    
    Args:
        dataset: List of dataset entries
        rag_pipeline: The RAG pipeline instance
        k: Number of documents to retrieve (top-k)
        
    Returns:
        List of LLMTestCase objects
    """
    test_cases = []
    retriever = rag_pipeline.retriever.get_retriever(k=k)
    
    for i, entry in enumerate(dataset):
        logger.info(f"Processing test case {i+1}/{len(dataset)}: {entry.get('input', '')[:50]}...")
        
        query = entry.get("input", "")
        if not query:
            logger.warning(f"Skipping entry {i+1}: missing 'input' field")
            continue
        
        expected_output = entry.get("expected_output", "")
        if not expected_output:
            logger.warning(f"Skipping entry {i+1}: missing 'expected_output' field")
            continue
        
        # Retrieve documents (for retrieval metrics)
        try:
            docs = retriever.invoke(query)
            retrieval_context = [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents for query {i+1}: {e}")
            retrieval_context = []
        
        # Generate actual output using RAG pipeline (for generation metrics)
        try:
            response = rag_pipeline.query(query)
            actual_output = response.get("answer", "") if isinstance(response, dict) else str(response)
            if not actual_output:
                actual_output = ""
        except Exception as e:
            logger.error(f"Error generating RAG output for query {i+1}: {e}")
            actual_output = ""
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)
    
    logger.info(f"Created {len(test_cases)} test cases")
    return test_cases
    

def main():
    """Main evaluation function."""
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"), 
        log_file=os.getenv("LOG_FILE", "logs/acceptance_rag_evaluation.log"),
        log_format=os.getenv("LOG_FORMAT", "text"),
        json_log_file=None
    )
    
    logger.info("Starting RAG acceptance test (Retrieval + Generation)")
    
    # Load dataset
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "data" / "processed" / "synthetic_dataset_ragas.json"
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    dataset = load_dataset(str(dataset_path))
    if not dataset:
        print("ERROR: Dataset is empty", file=sys.stderr)
        sys.exit(1)
    
    if len(dataset) < 20:
        print(f"WARNING: Dataset has only {len(dataset)} questions, expected ≥20", file=sys.stderr)
    
    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create test cases
    k = 5
    logger.info(f"Creating test cases with k={k}...")
    test_cases = create_test_cases(dataset, rag_pipeline, k=k)
    
    if not test_cases:
        print("ERROR: No test cases created", file=sys.stderr)
        sys.exit(1)
    
    # Configure Azure OpenAI for DeepEval
    evaluation_model = None
    if os.getenv("DEEPEVAL_USE_AZURE", "true").lower() == "true":
        try:
            endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
            if not endpoint.endswith("/"):
                endpoint += "/"
            
            evaluation_model = AzureOpenAIModel(
                model_name=settings.MODEL_NAME,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
                azure_openai_api_key=settings.OPENAI_API_KEY,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=endpoint,
                temperature=0,
            )
            logger.info("Azure OpenAI configured for DeepEval metrics")
        except Exception as e:
            logger.warning(f"Failed to configure Azure OpenAI: {e}")
    
    # Initialize metrics
    threshold = 0.5
    metric_kwargs = {"threshold": threshold, "include_reason": False}
    if evaluation_model:
        metric_kwargs["model"] = evaluation_model
    
    contextual_precision = ContextualPrecisionMetric(**metric_kwargs)
    contextual_recall = ContextualRecallMetric(**metric_kwargs)
    contextual_relevancy = ContextualRelevancyMetric(**metric_kwargs)
    answer_relevancy = AnswerRelevancyMetric(**metric_kwargs)
    faithfulness = FaithfulnessMetric(**metric_kwargs)
    
    metrics = [
        contextual_precision,
        contextual_recall,
        contextual_relevancy,
        answer_relevancy,
        faithfulness,
    ]
    
    # Run evaluation iteratively (one test case at a time)
    logger.info(f"Running E2E RAG evaluation on {len(test_cases)} test cases (one at a time)...")
    
    # Track scores for aggregation
    metric_scores = {
        "Contextual Precision": [],
        "Contextual Recall": [],
        "Contextual Relevancy": [],
        "Answer Relevancy": [],
        "Faithfulness": [],
    }
    
    successful_evaluations = 0
    failed_evaluations = 0
    
    max_retries = int(os.getenv("EVAL_MAX_RETRIES", "3"))
    
    for idx, test_case in enumerate(test_cases):
        logger.info(f"Evaluating test case {idx+1}/{len(test_cases)}: {test_case.input[:50]}...")
        
        retry_count = 0
        test_case_success = False
        
        while retry_count < max_retries:
            try:
                evaluation_results = evaluate(test_cases=[test_case], metrics=metrics)
                
                # Extract scores from metrics_data
                if evaluation_results.test_results:
                    for metric_data in evaluation_results.test_results[0].metrics_data:
                        metric_name = metric_data.name
                        if metric_name in metric_scores and metric_data.score is not None:
                            metric_scores[metric_name].append(metric_data.score)
                
                successful_evaluations += 1
                test_case_success = True
                break
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "429" in error_msg or "RateLimit" in error_msg or "rate limit" in error_msg.lower():
                    if retry_count < max_retries:
                        wait_time = 5 * retry_count
                        logger.warning(f"Rate limit hit for test case {idx+1}. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit error after {max_retries} retries for test case {idx+1}. Skipping...")
                        failed_evaluations += 1
                        break
                elif "invalid JSON" in error_msg or "JSONDecodeError" in error_msg:
                    if retry_count < max_retries:
                        wait_time = 3 * retry_count
                        logger.warning(f"JSON parsing error for test case {idx+1}. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"JSON parsing error after {max_retries} retries for test case {idx+1}. Skipping...")
                        failed_evaluations += 1
                        break
                else:
                    if retry_count < max_retries:
                        wait_time = 2 * retry_count
                        logger.warning(f"Error evaluating test case {idx+1}: {error_msg[:100]}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Error after {max_retries} retries for test case {idx+1}: {error_msg[:100]}")
                        failed_evaluations += 1
                        break
        
        # Small delay to avoid rate limits
        if idx < len(test_cases) - 1 and test_case_success:
            time.sleep(1)
    
    logger.info(f"Evaluation completed: {successful_evaluations} successful, {failed_evaluations} failed")
    
    # Calculate average scores
    avg_scores = {}
    for metric_name, scores in metric_scores.items():
        if scores:
            avg_scores[metric_name] = sum(scores) / len(scores)
        else:
            avg_scores[metric_name] = None
    
    # Update metric objects with aggregated scores
    if avg_scores.get("Contextual Precision") is not None:
        contextual_precision.score = avg_scores["Contextual Precision"]
    if avg_scores.get("Contextual Recall") is not None:
        contextual_recall.score = avg_scores["Contextual Recall"]
    if avg_scores.get("Contextual Relevancy") is not None:
        contextual_relevancy.score = avg_scores["Contextual Relevancy"]
    if avg_scores.get("Answer Relevancy") is not None:
        answer_relevancy.score = avg_scores["Answer Relevancy"]
    if avg_scores.get("Faithfulness") is not None:
        faithfulness.score = avg_scores["Faithfulness"]
    
    # Print summary report
    print("\n" + "="*70)
    print("=== RAG E2E Evaluation Report (Top-5) ===")
    print("="*70)
    print(f"Total questions evaluated: {len(test_cases)}")
    print(f"Retrieval k value: {k} (top-5)")
    print()
    
    # Retrieval Metrics
    print("--- RETRIEVAL METRICS ---")
    precision_score = contextual_precision.score
    recall_score = contextual_recall.score
    relevancy_score = contextual_relevancy.score
    
    precision_str = f"{precision_score:.4f}" if precision_score is not None else "N/A"
    recall_str = f"{recall_score:.4f}" if recall_score is not None else "N/A"
    relevancy_str = f"{relevancy_score:.4f}" if relevancy_score is not None else "N/A"
    
    print(f"\n  Contextual Precision@5: {precision_str} (threshold: {threshold})")
    print(f"  Contextual Recall@5: {recall_str} (threshold: {threshold})")
    print(f"  Contextual Relevancy@5: {relevancy_str} (threshold: {threshold})")
    
    # Generation Metrics
    print("\n--- GENERATION METRICS ---")
    answer_relevancy_score = answer_relevancy.score
    faithfulness_score = faithfulness.score
    
    answer_relevancy_str = f"{answer_relevancy_score:.4f}" if answer_relevancy_score is not None else "N/A"
    faithfulness_str = f"{faithfulness_score:.4f}" if faithfulness_score is not None else "N/A"
    
    print(f"\n  Answer Relevancy: {answer_relevancy_str} (threshold: {threshold})")
    print(f"  Faithfulness: {faithfulness_str} (threshold: {threshold})")
    
    print()
    print("="*70)
    
    # Acceptance criteria: Based on Answer Relevancy only
    print("\n--- ACCEPTANCE CRITERIA ---")
    print("Acceptance is based on Answer Relevancy metric")
    
    if answer_relevancy_score is None:
        print("\n❌ Acceptance test FAILED: Answer Relevancy score not available")
        sys.exit(1)
    
    passed = answer_relevancy_score >= threshold
    
    if passed:
        print(f"\n✅ Acceptance test PASSED: Answer Relevancy meets threshold")
        print(f"   Answer Relevancy: {answer_relevancy_score:.4f} (threshold: {threshold})")
        sys.exit(0)
    else:
        print(f"\n❌ Acceptance test FAILED: Answer Relevancy below threshold")
        print(f"   Answer Relevancy: {answer_relevancy_score:.4f} (threshold: {threshold})")
        sys.exit(1)


if __name__ == "__main__":
    main()
