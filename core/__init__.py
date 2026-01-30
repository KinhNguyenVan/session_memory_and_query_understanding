"""
Core modules for Chat Assistant.

This package exposes a clean public API while the actual implementation
is organized into subpackages:

- core.schema:     Pydantic schemas (SessionMemory, QueryUnderstanding)
- core.memory:     Session memory & summarization
- core.chatbot:    LLM client, query understanding pipeline, chat assistant
"""

# Core schemas (session memory + query understanding)
from .schema.core_schema import SessionMemory, UserContext, CoreQueryUnderstanding

# Core classes
from .chatbot.llm_client import LLMClient
from .memory.session_memory import SessionMemoryManager
from .chatbot.query_understanding import QueryUnderstandingPipeline
from .chatbot.chat_assistant import ChatAssistant

__all__ = [
    "SessionMemory",
    "UserContext",
    "CoreQueryUnderstanding",
    "LLMClient",
    "SessionMemoryManager",
    "QueryUnderstandingPipeline",
    "ChatAssistant",
]
