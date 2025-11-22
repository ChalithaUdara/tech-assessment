import os
import glob
import json
from typing import List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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

def load_documents(data_dir: str, file_pattern: str = "Arthur_Conan_Doyle*.txt") -> List[Document]:
    """Load and split documents from the specified directory."""
    documents = []
    search_path = os.path.join(data_dir, file_pattern)
    files = glob.glob(search_path)
    
    print(f"Found {len(files)} files matching {file_pattern}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Create a document for the whole file, or split it
                # For DeepEval synthesizer, passing chunks is often better
                chunks = text_splitter.create_documents([text], metadatas=[{"source": file_path}])
                documents.extend(chunks)
                print(f"Loaded {len(chunks)} chunks from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents

def main():
    # Configuration
    DATA_DIR = "data/raw"
    OUTPUT_FILE = "data/processed/synthetic_dataset.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Load Documents
    print("Loading documents...")
    documents = load_documents(DATA_DIR)
    
    if not documents:
        print("No documents found. Exiting.")
        return

    # Limit documents for testing purposes if needed, or use all
    # For this task, we'll use a subset of chunks to keep generation time reasonable
    # or maybe just the first few chunks of each book?
    # Let's use a subset of 20 chunks for now to demonstrate functionality
    selected_documents = documents[:20] 
    print(f"Selected {len(selected_documents)} chunks for generation.")

    # 2. Setup LLM
    print("Setting up LLM...")
    azure_llm = create_llm_client()
    deepeval_model = AzureOpenAIWrapper(azure_llm)

    # 3. Initialize Synthesizer
    print("Initializing Synthesizer...")
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.REASONING: 0.1,
            Evolution.MULTICONTEXT: 0.2,
            Evolution.CONCRETIZING: 0.1,
            Evolution.CONSTRAINED: 0.1,
            Evolution.COMPARATIVE: 0.1,
            Evolution.HYPOTHETICAL: 0.1,
            Evolution.IN_BREADTH: 0.3,
        }
    )
    synthesizer = Synthesizer(
        model=deepeval_model, 
        evolution_config=evolution_config,
        max_concurrent=3
    )

    # 4. Generate Goldens
    print("Generating synthetic dataset...")
    
    contexts = [[doc.page_content] for doc in selected_documents]
    
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        max_goldens_per_context=1, # 1 question per chunk
        include_expected_output=True
    )

    # 5. Save Dataset
    print(f"Generated {len(goldens)} goldens. Saving to {OUTPUT_FILE}...")
    dataset = EvaluationDataset(goldens=goldens)
    dataset.save_as(file_type='json', directory=os.path.dirname(OUTPUT_FILE))
    
    # Also save as a simple JSON for inspection if needed, 
    # but dataset.save_as should handle it.
    # Let's also manually save to ensure we have the format we want if save_as is tricky
    
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
        
    print("Done!")

if __name__ == "__main__":
    main()
