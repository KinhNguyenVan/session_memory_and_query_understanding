import os
import json
from datetime import datetime
from typing import Optional, Dict, Any


class ConversationLogger:
    """
    Simple JSONL conversation logger.
    Each line is a JSON object with at least: role, content, timestamp.
    Compatible with the format in test_data/conversation_*.jsonl.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a single message to the log file as JSON."""
        entry: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": timestamp or datetime.now().isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

