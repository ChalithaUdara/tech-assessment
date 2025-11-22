from typing import Dict, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from datacom_ai.utils.logger import logger

class RetrievalPipeline:
    """Handles retrieval and generation using RAG."""
    
    # System prompt for RAG - extracted to avoid duplication
    SYSTEM_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for RAG."""
        return ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

    def __init__(self, vector_store: VectorStore, llm: BaseChatModel):
        self.vector_store = vector_store
        self.llm = llm

    def get_retriever(self, k: int = 4):
        """Get the retriever from the vector store."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the RAG pipeline for a given query.
        """
        logger.info(f"Running RAG for query: {query}")
        
        retriever = self.get_retriever()
        prompt = self._create_prompt()
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        return response

    def stream(self, query: str):
        """
        Stream the RAG pipeline response.
        """
        logger.info(f"Streaming RAG for query: {query}")
        
        retriever = self.get_retriever()
        prompt = self._create_prompt()
        
        # Stream the response
        # First, retrieve documents
        docs = retriever.invoke(query)
        yield {"context": docs}
        
        # Then generate answer using the retrieved documents
        # Then generate answer using the retrieved documents
        # We manually construct the chain to ensure we get the raw LLM output (with usage metadata)
        # instead of just the string content which create_stuff_documents_chain might return
        
        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)
            
        formatted_context = format_docs(docs)
        
        # Create messages directly
        messages = prompt.format_messages(context=formatted_context, input=query)
        
        # Stream directly from LLM
        for chunk in self.llm.stream(messages):
            # Yield content
            if chunk.content:
                yield {"answer": chunk.content}
                
            # Yield usage metadata if present
            if chunk.usage_metadata:
                yield {"usage": chunk.usage_metadata}
