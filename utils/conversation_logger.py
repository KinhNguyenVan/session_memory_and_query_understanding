import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class ConversationLogger:
    """
    Simple JSONL conversation logger.
    Each line is a JSON object with at least: role, content, timestamp.
    Compatible with the format in test_data/conversation_*.jsonl.
    Use seed_from_history() after loading a conversation so the log file contains
    loaded + new messages and can be used as the single source for 20 recent context.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def seed_from_history(self, messages: List[Dict[str, Any]]) -> None:
        """
        Overwrite the log file with existing conversation history (e.g. after loading test_data).
        Subsequent log_message() calls will append. Use overwrite (not append) to avoid
        duplicate history when seed is called multiple times (e.g. Streamlit reruns).
        """
        if not messages:
            return
        with open(self.log_path, "w", encoding="utf-8") as f:
            for msg in messages:
                entry: Dict[str, Any] = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp") or datetime.now().isoformat(),
                }
                if msg.get("metadata"):
                    entry["metadata"] = msg["metadata"]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

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

