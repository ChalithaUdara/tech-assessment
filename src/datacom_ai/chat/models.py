"""Data models for the chat application."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Source:
    """
    Represents a source used in RAG generation.
    """
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class StreamUpdate:
    """
    Represents a single update during the streaming response.
    
    Attributes:
        content: The text content chunk (or accumulated content)
        metadata: Optional metadata (e.g., usage stats, metrics)
        sources: Optional list of sources used (for RAG)
        thought: Optional reasoning trace (for Agents)
        error: Optional error message
    """
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
    sources: Optional[List[Source]] = None
    thought: Optional[str] = None
    error: Optional[str] = None
