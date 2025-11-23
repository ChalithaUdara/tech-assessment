"""Logging configuration using loguru."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from loguru import logger


def serialize_record(record: Dict[str, Any]) -> str:
    """
    Serialize log record to JSON format.
    
    This function is used as a custom serializer for loguru JSON logging.
    It receives a loguru record dict and converts it to JSON.
    
    Args:
        record: Log record dictionary from loguru
        
    Returns:
        JSON string representation of the log record
    """
    # Extract relevant fields from loguru record
    try:
        # Handle timestamp
        time_obj = record.get("time")
        if time_obj:
            timestamp = time_obj.isoformat() if hasattr(time_obj, "isoformat") else str(time_obj)
        else:
            timestamp = ""
    except (AttributeError, KeyError, TypeError):
        timestamp = str(record.get("time", ""))
    
    # Handle level
    level_obj = record.get("level")
    if level_obj and hasattr(level_obj, "name"):
        level_name = level_obj.name
    else:
        level_name = str(level_obj) if level_obj else "INFO"
    
    log_data = {
        "timestamp": timestamp,
        "level": level_name,
        "message": record.get("message", ""),
        "module": record.get("name", ""),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
    }
    
    # Add exception info if present
    exception = record.get("exception")
    if exception:
        try:
            log_data["exception"] = {
                "type": exception.type.__name__ if exception.type else None,
                "value": str(exception.value) if exception.value else None,
            }
        except (AttributeError, TypeError):
            pass
    
    # Add extra fields (structured logging data)
    extra = record.get("extra")
    if extra:
        # Filter out loguru internal fields
        for key, value in extra.items():
            if key not in ("id", "pid", "tid", "elapsed"):
                log_data[key] = value
    
    return json.dumps(log_data, default=str) + "\n"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: str | None = None,
    log_format: str = "both",
    json_log_file: str | Path | None = None,
) -> None:
    """
    Configure loguru logger with sensible defaults and optional JSON logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to text log file. If None, logs only to console.
        rotation: Log rotation size (e.g., "10 MB", "1 day")
        retention: Log retention period (e.g., "7 days", "1 month")
        format_string: Custom format string. If None, uses a structured format.
        log_format: Log format mode - "json", "text", or "both" (default: "both")
        json_log_file: Optional path to JSON log file (JSONL format). 
                      If None and log_format includes JSON, uses default path.
    """
    # Remove default handler
    logger.remove()

    # Determine which formats to use
    use_json = log_format in ("json", "both")
    use_text = log_format in ("text", "both")
    
    if not use_json and not use_text:
        # Fallback to both if invalid option
        use_json = True
        use_text = True

    # Console handler - always human-readable
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    if use_text:
        # Add console handler with colors
        logger.add(
            sys.stderr,
            format=format_string,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Text file handler
    if use_text and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File format without colors
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe logging
        )

    # JSON file handler
    if use_json:
        json_path = Path(json_log_file) if json_log_file else Path("logs/chat.jsonl")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use loguru's built-in serialize=True for automatic JSON serialization
        # This handles rotation, retention, and compression automatically
        # The structured data from logger.bind() in structured_logging.py will be included
        # in the "extra" field of the serialized JSON
        logger.add(
            str(json_path),  # Convert Path to string for loguru
            format="{message}",  # Simple format - serialize handles JSON conversion
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe logging
            serialize=True,  # Enable automatic JSON serialization
        )


# Initialize logger with default settings on import (text only, no JSON)
# Users can call setup_logging() again to reconfigure (it will remove existing handlers first)
setup_logging(log_format="text")

# Export logger for easy import
__all__ = ["logger", "setup_logging"]

