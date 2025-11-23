"""Custom exception hierarchy for the Datacom AI platform."""


class DatacomAIError(Exception):
    """Base exception for Datacom AI platform errors."""
    pass


class ChatError(DatacomAIError):
    """Chat-related errors."""
    pass


class RAGError(DatacomAIError):
    """RAG pipeline errors."""
    pass


class ConfigurationError(DatacomAIError):
    """Configuration errors."""
    pass


class EngineError(DatacomAIError):
    """Chat engine errors."""
    pass


class VectorStoreError(DatacomAIError):
    """Vector store operation errors."""
    pass

