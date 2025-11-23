"""Chunk parsers for different engine types."""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document

from datacom_ai.chat.models import StreamUpdate
from datacom_ai.telemetry.metrics import TelemetryMetrics


class RAGChunkParser:
    """Parser for RAG pipeline chunk formats."""

    @staticmethod
    def parse(chunk: Dict[str, Any], metrics: TelemetryMetrics) -> Optional[StreamUpdate]:
        """
        Parse RAG chunk into StreamUpdate.

        Args:
            chunk: RAG pipeline chunk (dict with "context", "answer", or "usage")
            metrics: TelemetryMetrics instance to update

        Returns:
            StreamUpdate with content or metadata, or None if chunk should be skipped
        """
        if not isinstance(chunk, dict):
            # Fallback if pipeline yields strings directly (legacy support)
            return StreamUpdate(content=str(chunk))

        if "context" in chunk:
            return RAGChunkParser._parse_context(chunk["context"])

        elif "answer" in chunk:
            return StreamUpdate(content=chunk["answer"])

        elif "usage" in chunk:
            RAGChunkParser._parse_usage(chunk["usage"], metrics)
            return None  # Usage metadata is tracked, no need to yield

        return None

    @staticmethod
    def _parse_context(docs: List[Document]) -> StreamUpdate:
        """
        Parse context documents into citations.

        Args:
            docs: List of retrieved documents

        Returns:
            StreamUpdate with citations metadata
        """
        citations = []
        for doc in docs:
            # Extract relevant metadata
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            citations.append(f"Source: {source}, Page: {page}")

        # Yield citations as metadata
        return StreamUpdate(metadata={"citations": citations})

    @staticmethod
    def _parse_usage(usage: Dict[str, Any], metrics: TelemetryMetrics) -> None:
        """
        Parse usage metadata and update metrics.

        Args:
            usage: Usage metadata dictionary
            metrics: TelemetryMetrics instance to update
        """
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        metrics.set_token_usage(prompt_tokens, completion_tokens, total_tokens)


