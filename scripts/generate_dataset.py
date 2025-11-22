"""
Generate synthetic Q&A dataset from Arthur Conan Doyle books using DeepEval Synthesizer.

⚠️  NOTE: This script is SLOWER than the fast alternative.
   For much faster generation, use: scripts/generate_dataset_fast.py
   (2-5 min vs 15-30+ min for 25 Q&A pairs)

Performance optimizations applied:
1. Disabled complex evolutions (num_evolutions=0) - evolutions add multiple LLM calls per golden
2. Using pre-chunked contexts approach for reliable generation
3. Increased max_concurrent to 10 for parallel processing
4. Simplified evolution config to only IN_BREADTH (if evolutions enabled)
5. Manual chunking with RecursiveCharacterTextSplitter for better control
6. Answer generation enabled by default (set GENERATE_ANSWERS=False to skip for speed)

For faster generation:
- Keep num_evolutions=0
- Set GENERATE_ANSWERS=False (skip answers, but dataset won't have expected_output)
- Or use scripts/generate_dataset_fast.py instead

Usage:
    uv run scripts/generate_dataset.py
"""
import os
import glob
import json

from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Evolution
from deepeval.synthesizer.config import EvolutionConfig

from datacom_ai.clients.llm_client import create_llm_client

# Wrapper for LangChain AzureChatOpenAI to be compatible with DeepEval
class AzureOpenAIWrapper(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Azure OpenAI"

def main():
    # Configuration
    DATA_DIR = "data/raw"
    OUTPUT_FILE = "data/processed/synthetic_dataset.json"
    TARGET_GOLDENS = 25  # Target number of Q&A pairs
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Get document file paths (let DeepEval handle chunking)
    print("Finding documents...")
    search_path = os.path.join(DATA_DIR, "Arthur_Conan_Doyle*.txt")
    document_files = glob.glob(search_path)
    
    if not document_files:
        print("No documents found. Exiting.")
        return
    
    print(f"Found {len(document_files)} files. Using first file for generation.")
    # Use just the first file to speed things up - adjust if needed
    document_paths = document_files[:1]

    # 2. Setup LLM
    print("Setting up LLM...")
    azure_llm = create_llm_client()
    deepeval_model = AzureOpenAIWrapper(azure_llm)

    # 3. Initialize Synthesizer with optimized settings
    print("Initializing Synthesizer...")
    # Use minimal evolutions - only IN_BREADTH for faster generation
    # Set num_evolutions=0 to disable evolutions entirely for maximum speed
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.IN_BREADTH: 1.0,  # Only use simple breadth expansion
        },
        num_evolutions=0  # Disable evolutions for speed - set to 1-2 if you want some variation
    )
    
    # Initialize Synthesizer with optimized settings
    # Chunking parameters can be set here if supported by your DeepEval version
    synthesizer = Synthesizer(
        model=deepeval_model, 
        evolution_config=evolution_config,
        max_concurrent=10  # Increased concurrency for parallel processing
        # Note: If your DeepEval version supports it, you can add:
        # chunk_size=1024,
        # chunk_overlap=0,
    )

    # 4. Generate Goldens using pre-chunked contexts
    # This approach works reliably with DeepEval's current API
    print(f"Generating {TARGET_GOLDENS} synthetic Q&A pairs...")
    print("This may take a few minutes...")
    print(f"Processing document: {document_paths[0]}")
    
    # Check document size
    doc_size = os.path.getsize(document_paths[0])
    print(f"Document size: {doc_size:,} bytes")
    
    # Read and chunk the document manually for reliable processing
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    with open(document_paths[0], 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from document")
    
    # Use chunks to generate goldens - spread across multiple chunks
    # Calculate how many chunks we need: TARGET_GOLDENS / max_goldens_per_context
    max_goldens_per_context = 1
    num_contexts_needed = min(TARGET_GOLDENS, len(chunks))
    contexts = [[chunk] for chunk in chunks[:num_contexts_needed]]
    print(f"Using {len(contexts)} contexts to generate goldens...")
    
    # Generate expected_output (answers) for the synthetic questions
    # Set to False to skip answer generation for faster processing
    GENERATE_ANSWERS = True  # Set to False to skip answers (faster but no expected outputs)
    
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        max_goldens_per_context=max_goldens_per_context,
        include_expected_output=GENERATE_ANSWERS  # Skip answers for speed
    )

    # 5. Save Dataset
    print(f"Generated {len(goldens)} goldens.")
    
    if len(goldens) == 0:
        print("ERROR: No goldens were generated. This might be due to:")
        print("  - Document too small or empty")
        print("  - Chunking issues")
        print("  - Quality filtering rejecting all generated inputs")
        print("Try increasing the document size or adjusting parameters.")
        return
    
    # Limit to exactly TARGET_GOLDENS if we got more
    if len(goldens) > TARGET_GOLDENS:
        print(f"Limiting to {TARGET_GOLDENS} goldens (got {len(goldens)})")
        goldens = goldens[:TARGET_GOLDENS]
    
    print(f"Saving {len(goldens)} goldens to {OUTPUT_FILE}...")
    dataset = EvaluationDataset(goldens=goldens)
    dataset.save_as(file_type='json', directory=os.path.dirname(OUTPUT_FILE))
    
    # Also save as a simple JSON for inspection
    output_data = []
    for golden in goldens:
        output_data.append({
            "input": golden.input,
            "actual_output": golden.actual_output,
            "expected_output": golden.expected_output,
            "context": golden.context
        })
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Done! Saved {len(goldens)} Q&A pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
