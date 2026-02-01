"""
Query Understanding Pipeline

Input: user query + session memory (if any) + recent messages (up to 20).
Output: CoreQueryUnderstanding
  - is_ambiguous: whether the query is ambiguous in context
  - clarified_query: ALWAYS required; best-effort clarified/rewritten query (used in final_context)
  - clarifying_questions: optional follow-up questions (no rigid template; answer step uses clarified_query)
  - selected_memory: relevant snippets (from memory; not full dump)
  - final_context: built in code: ORIGINAL QUERY + CLARIFIED QUERY + conversation_state + selected_memory + recent messages
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from ..schema.core_schema import CoreQueryUnderstanding, CoreQueryUnderstandingLLMOutput


class QueryUnderstandingPipeline:
    """Pipeline for understanding and refining user queries (Core schema)."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def process_query(
        self,
        query: str,
        recent_messages: List[Dict[str, Any]],
        session_memory_context: Dict[str, Any],
        has_session_summary: bool = True,
    ) -> CoreQueryUnderstanding:
        """
        Process a user query: detect ambiguity, optionally rewrite,
        select relevant memory, build final_context.

        When has_session_summary: final_context uses 5 recent messages (memory covers the rest).
        When not: no memory → final_context uses up to 20 recent messages.
        """
        llm_output = self._run_core_llm(
            query=query,
            recent_messages=recent_messages,
            session_memory_context=session_memory_context,
        )

        # Build final_context: 5 recent if we have memory; up to 20 recent if we don't
        final_context = self._build_final_context(
            query=query,
            clarified_query=llm_output.clarified_query,
            session_memory_context=session_memory_context,
            selected_memory=llm_output.selected_memory,
            recent_messages=recent_messages,
            has_session_summary=has_session_summary,
        )

        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return CoreQueryUnderstanding(
            query_id=query_id,
            original_query=query,
            is_ambiguous=llm_output.is_ambiguous,
            clarified_query=llm_output.clarified_query,
            clarifying_questions=llm_output.clarifying_questions,
            selected_memory=llm_output.selected_memory,
            final_context=final_context,
        )

    def _run_core_llm(
        self,
        query: str,
        recent_messages: List[Dict[str, Any]],
        session_memory_context: Dict[str, Any],
    ) -> CoreQueryUnderstandingLLMOutput:
        """Single LLM call to fill is_ambiguous, clarified_query, clarifying_questions, selected_memory."""
        # Use all recent_messages (up to 20) for the prompt
        convo_lines: List[str] = []
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "")[:300]
            convo_lines.append(f"{role}: {content}")
        recent_conversation = "\n".join(convo_lines) if convo_lines else "N/A"

        try:
            import json
            session_memory_json = json.dumps(
                session_memory_context, indent=2, ensure_ascii=False
            )
        except Exception:
            session_memory_json = str(session_memory_context)

        system_prompt = (
            "You are a query understanding engine for a conversational assistant.\n\n"
            "You receive: the current user query, recent conversation turns, and session memory:\n"
            "  - conversation_state: overall understanding of the conversation\n"
            "  - user_context: preferences, constraints, goals\n"
            "  - shared_context: facts or assumptions both sides agree on\n"
            "  - open_threads: unresolved topics\n\n"
            "Your tasks (all fields required except clarifying_questions):\n\n"
            "1) is_ambiguous: Is the query ambiguous in this context? Use conversation_state, shared_context, "
            "open_threads to decide. If memory is enough to understand (e.g. 'cái này' = current schema), set false.\n\n"
            "2) clarified_query: REQUIRED. Always output a single clarified/rewritten query.\n"
            "   - When context is enough: rewrite the query explicitly using that context (e.g. 'DL' in a ML chat → 'deep learning').\n"
            "   - When context is unclear: give your best-effort interpretation or repeat the original query verbatim.\n"
            "   Never leave this empty; the answer step always uses this.\n\n"
            "3) clarifying_questions: Optional. Only when is_ambiguous=true and it would help to list 1–3 short follow-up questions. "
            "Otherwise return an empty list.\n\n"
            "4) selected_memory: Select ONLY the memory snippets relevant to this query. Draw from: "
            "conversation_state, user_context, shared_context, open_threads. Return a list of short sentences. Do NOT dump the full memory.\n\n"
            "Return ONLY the fields of CoreQueryUnderstandingLLMOutput (final_context is built in code)."
        )

        user_template = (
            "USER QUERY:\n\"\"\"{query}\"\"\"\n\n"
            "RECENT CONVERSATION (most recent last):\n{recent_conversation}\n\n"
            "SESSION MEMORY:\n{session_memory_json}\n\n"
            "Fill the CoreQueryUnderstandingLLMOutput fields."
        )

        return self.llm_client.generate_structured(
            CoreQueryUnderstandingLLMOutput,
            system_prompt=system_prompt,
            user_template=user_template,
            variables={
                "query": query,
                "recent_conversation": recent_conversation,
                "session_memory_json": session_memory_json,
            },
            temperature=0.3,
        )

    def _build_final_context(
        self,
        query: str,
        clarified_query: str,
        session_memory_context: Dict[str, Any],
        selected_memory: List[str],
        recent_messages: List[Dict[str, Any]],
        has_session_summary: bool = True,
    ) -> str:
        """
        Build final_context with exactly 5 parts (required):
        1. ORIGINAL QUERY: user's raw query
        2. CLARIFIED QUERY: LLM's best-effort clarified/rewritten query (always present)
        3. CONVERSATION STATE: from session memory
        4. SELECTED MEMORY: list of snippets (or "None.")
        5. RECENT MESSAGES: 5 recent when has_session_summary; up to 20 when no memory.
        """
        original_block = f"ORIGINAL QUERY:\n{query}\n\nCLARIFIED QUERY:\n{clarified_query or query}"
        conversation_state = session_memory_context.get("conversation_state", "") or "(none)"
        selected_block = "\n".join(f"- {s}" for s in selected_memory) if selected_memory else "None."

        if has_session_summary:
            # Have memory → use last 5 recent only; pad if fewer
            recent_slice = list(recent_messages[-5:])
            while len(recent_slice) < 5:
                recent_slice.insert(0, {"role": "(none)", "content": "(no earlier message)"})
            recent_label = "RECENT MESSAGES (last 5)"
        else:
            # No memory (under threshold) → use all recent (up to 20), enforce chronological order.
            # Sort by timestamp so display is always oldest-first even if caller passed wrong order.
            # Secondary key: original index, so messages without timestamp keep relative order.
            def _sort_key(item: tuple) -> tuple:
                i, m = item
                ts = (m.get("timestamp") or "")[:26]
                return (ts, i)
            recent_slice = [m for _, m in sorted(enumerate(recent_messages), key=_sort_key)]
            recent_label = "RECENT MESSAGES"

        recent_lines = []
        for msg in recent_slice:
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "")[:400]
            recent_lines.append(f"{role}: {content}")
        recent_block = "\n".join(recent_lines)

        return (
            f"{original_block}\n\n"
            f"CONVERSATION STATE:\n{conversation_state}\n\n"
            f"SELECTED MEMORY:\n{selected_block}\n\n"
            f"{recent_label}:\n{recent_block}"
        )
