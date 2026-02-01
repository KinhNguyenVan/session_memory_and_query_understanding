"""
Core Schema Definitions for Chat Assistant

Session memory is designed to store only stable, reusable context
that improves future query understanding.

Pipeline:
  (1) 20 recent messages → Summarize → SessionMemory (long-term)
  (2) User query + session memory + 5 recent messages → Query Understanding
  (3) Answer uses final_context only
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# SESSION MEMORY (Long-term – by role, not by data type)
# ============================================================================

class UserContext(BaseModel):
    """Stable user-related information: preferences, constraints, goals."""
    preferences: List[str] = Field(default_factory=list, description="User preferences mentioned")
    constraints: List[str] = Field(default_factory=list, description="Limitations or constraints")
    goals: List[str] = Field(default_factory=list, description="User goals inferred from conversation")


class SessionMemory(BaseModel):
    """
    Session memory: stable, reusable context for many future queries.
    Fields are grouped by role (conversation state, user, shared facts, open loops).
    """
    memory_id: str = Field(..., description="Unique identifier for this memory")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    conversation_state: str = Field(
        ...,
        description="Overall understanding of what the conversation is about"
    )

    user_context: Optional[UserContext] = Field(
        None,
        description="Long-lived user preferences, constraints, or goals"
    )

    shared_context: List[str] = Field(
        default_factory=list,
        description="Facts or assumptions both user and assistant agree on"
    )

    open_threads: List[str] = Field(
        default_factory=list,
        description="Unresolved topics that may affect future queries"
    )

    scope: Dict[str, int] = Field(
        ...,
        description="Which messages were summarized: {\"from\": int, \"to\": int}"
    )


# ============================================================================
# QUERY UNDERSTANDING
# ============================================================================

class CoreQueryUnderstandingLLMOutput(BaseModel):
    """
    LLM fills this in one shot. final_context is NOT filled by LLM — it is built in code
    from query (clarified or original), conversation_state, selected_memory, and 5 recent messages.
    """
    is_ambiguous: bool = Field(
        ...,
        description="Whether the query is ambiguous in current context"
    )
    clarified_query: str = Field(
        ...,
        description="Always required. Best-effort clarified/rewritten query using context; use original query verbatim if no disambiguation possible."
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Questions to ask user if ambiguity remains"
    )
    selected_memory: List[str] = Field(
        default_factory=list,
        description="Relevant memory snippets selected for this query (from conversation_state, user_context, shared_context, open_threads)"
    )


class CoreQueryUnderstanding(CoreQueryUnderstandingLLMOutput):
    """Full query understanding result (used in pipeline and demos)."""
    query_id: str = Field(..., description="Unique identifier for this query analysis")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    original_query: str = Field(..., description="The user's original query")
    final_context: str = Field(
        ...,
        description="Built in code: ORIGINAL QUERY + CLARIFIED QUERY + CONVERSATION STATE + SELECTED MEMORY + RECENT MESSAGES"
    )
