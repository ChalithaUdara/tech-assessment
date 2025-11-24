"""Log parsing utilities for analytics dashboard."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_log_file(path: str) -> List[Dict]:
    """
    Parse JSONL log file and extract structured events.
    
    Args:
        path: Path to the JSONL log file
        
    Returns:
        List of event dictionaries with structured data
        
    Raises:
        FileNotFoundError: If the log file doesn't exist
        ValueError: If the log file is malformed
    """
    log_path = Path(path)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    
    events = []
    relevant_event_types = {
        "chat_request",
        "chat_response",
        "llm_call",
        "rag_query",
        "rag_retrieval",
        "agent_execution",
        "agent_step",
    }
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_entry = json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip malformed lines but continue processing
                    continue
                
                # Extract structured event data from record.extra
                record = log_entry.get("record", {})
                extra = record.get("extra", {})
                
                # Only process events with event_type in our relevant set
                event_type = extra.get("event_type")
                if event_type not in relevant_event_types:
                    continue
                
                # Parse timestamp
                timestamp_iso = extra.get("timestamp_iso")
                timestamp = None
                if timestamp_iso:
                    try:
                        # Handle timezone-aware timestamps
                        if timestamp_iso.endswith("+00:00") or timestamp_iso.endswith("Z"):
                            timestamp = datetime.fromisoformat(
                                timestamp_iso.replace("Z", "+00:00")
                            )
                        else:
                            # Try parsing without timezone info
                            timestamp = datetime.fromisoformat(timestamp_iso)
                    except (ValueError, AttributeError):
                        # If timestamp parsing fails, use unix timestamp as fallback
                        timestamp_unix = extra.get("timestamp_unix")
                        if timestamp_unix:
                            try:
                                timestamp = datetime.fromtimestamp(timestamp_unix)
                            except (ValueError, OSError):
                                pass
                
                # Create event dictionary with all extra fields plus parsed timestamp
                event = {
                    **extra,
                    "timestamp": timestamp,
                    "timestamp_iso": timestamp_iso,
                }
                
                events.append(event)
    
    except Exception as e:
        raise ValueError(f"Error parsing log file at line {line_num}: {str(e)}")
    
    return events


def filter_events_by_type(events: List[Dict], event_type: str) -> List[Dict]:
    """
    Filter events by event type.
    
    Args:
        events: List of event dictionaries
        event_type: Event type to filter by
        
    Returns:
        Filtered list of events
    """
    return [e for e in events if e.get("event_type") == event_type]


def filter_events_by_date_range(
    events: List[Dict],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict]:
    """
    Filter events by date range.
    
    Args:
        events: List of event dictionaries
        start_date: Start date (inclusive)
        end_date: End date (inclusive, set to end of day)
        
    Returns:
        Filtered list of events
    """
    filtered = events
    
    if start_date:
        filtered = [
            e for e in filtered
            if e.get("timestamp") and e["timestamp"] >= start_date
        ]
    
    if end_date:
        # Set end_date to end of day to make it inclusive
        from datetime import time as dt_time
        end_date_inclusive = datetime.combine(end_date.date(), dt_time.max)
        filtered = [
            e for e in filtered
            if e.get("timestamp") and e["timestamp"] <= end_date_inclusive
        ]
    
    return filtered

