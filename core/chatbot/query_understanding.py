"""
Query Understanding Pipeline

Input: user query + session memory (conversation_state, user_context, shared_context, open_threads) + 5 recent messages
Output: CoreQueryUnderstanding
  - is_ambiguous → answer now vs rewrite vs ask clarifying questions
  - clarified_query → rewritten query when memory is enough
  - clarifying_questions → when memory is not enough
  - selected_memory → relevant snippets (from memory; not full dump)
  - final_context → passed to answer step only
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
        session_memory_context: Dict[str, Any]
    ) -> CoreQueryUnderstanding:
        """
        Process a user query: detect ambiguity, optionally rewrite,
        select relevant memory, build final_context.
        """
        llm_output = self._run_core_llm(
            query=query,
            recent_messages=recent_messages,
            session_memory_context=session_memory_context,
        )

        # Build final_context in code so it ALWAYS has: query, conversation_state, selected_memory, 5 recent messages
        final_context = self._build_final_context(
            query=query,
            clarified_query=llm_output.clarified_query,
            session_memory_context=session_memory_context,
            selected_memory=llm_output.selected_memory,
            recent_messages=recent_messages,
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
        """Single LLM call to fill is_ambiguous, clarified_query, clarifying_questions, selected_memory, final_context."""
        # 5 recent messages (abbreviated for prompt)
        convo_lines: List[str] = []
        for msg in recent_messages[-5:]:
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
            "Your tasks:\n"
            "1) is_ambiguous: Is the query ambiguous in this context? Use conversation_state, shared_context, "
            "open_threads to decide. If memory is enough to understand (e.g. 'cái này' = current schema), set false.\n"
            "2) clarified_query: If you can rewrite the query using memory (e.g. 'schema này có ổn không?' → "
            "'Is the current session memory schema well-designed?'), put the rewritten query here. Otherwise null.\n"
            "3) clarifying_questions: Only when is_ambiguous=true AND memory is NOT enough to clarify. "
            "List 1–3 short questions. Use open_threads and user_context.constraints to avoid redundant questions.\n"
            "4) selected_memory: Select ONLY the memory snippets relevant to this query. Draw from: "
            "conversation_state (short sentence), user_context (constraints/goals), shared_context (facts), "
            "open_threads (open issues). Return a list of short sentences. Do NOT dump the full memory.\n\n"
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
        clarified_query: Optional[str],
        session_memory_context: Dict[str, Any],
        selected_memory: List[str],
        recent_messages: List[Dict[str, Any]],
    ) -> str:
        """
        Build final_context with exactly 4 parts (required):
        1. USER QUERY: clarified_query if present, else original query
        2. CONVERSATION STATE: from session memory
        3. SELECTED MEMORY: list of snippets (or "None.")
        4. RECENT MESSAGES: exactly 5 recent messages
        """
        query_text = clarified_query if clarified_query else query
        conversation_state = session_memory_context.get("conversation_state", "") or "(none)"
        selected_block = "\n".join(f"- {s}" for s in selected_memory) if selected_memory else "None."
        # Exactly 5 recent messages: take last 5, pad with placeholder if fewer
        recent_5 = list(recent_messages[-5:])
        while len(recent_5) < 5:
            recent_5.insert(0, {"role": "(none)", "content": "(no earlier message)"})
        recent_lines = []
        for msg in recent_5:
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "")[:400]
            recent_lines.append(f"{role}: {content}")
        recent_block = "\n".join(recent_lines)

        return (
            f"USER QUERY:\n{query_text}\n\n"
            f"CONVERSATION STATE:\n{conversation_state}\n\n"
            f"SELECTED MEMORY:\n{selected_block}\n\n"
            f"RECENT MESSAGES (last 5):\n{recent_block}"
        )
