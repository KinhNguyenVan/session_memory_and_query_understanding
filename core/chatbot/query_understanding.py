"""
Query Understanding Pipeline

Input: user query + session memory (if any) + recent messages (up to 20).
Output: CoreQueryUnderstanding
  - is_ambiguous: whether the query is ambiguous in context
  - clarified_query: ALWAYS required; best-effort clarified/rewritten query (used in final_context)
  - clarifying_questions: optional follow-up questions (no rigid template; answer step uses clarified_query)
  - selected_memory: relevant snippets (from memory; not full dump)
  - final_context: built in code: ORIGINAL QUERY + CLARIFIED QUERY + (CLARIFYING_QUESTIONS when ambiguous) + conversation_state + selected_memory + recent messages
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

        # Build final_context: includes clarifying_questions when ambiguous (for answer step)
        final_context = self._build_final_context(
            query=query,
            clarified_query=llm_output.clarified_query,
            session_memory_context=session_memory_context,
            selected_memory=llm_output.selected_memory,
            recent_messages=recent_messages,
            has_session_summary=has_session_summary,
            clarifying_questions=llm_output.clarifying_questions or [],
            is_ambiguous=llm_output.is_ambiguous,
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
            "You receive: the current user query, recent conversation turns, and session memory (conversation_state, user_context, shared_context, open_threads).\n\n"
            "Before filling the output fields, reason STEP BY STEP:\n\n"
            "STEP 1 – Topic link: Does the query mention or clearly refer to the conversation topic? (e.g. 'pip install', 'Python import errors', 'evaluation metrics', 'that algorithm'). If YES → the query is tied to context; if NO → may be ambiguous.\n\n"
            "STEP 2 – Number of referents: In the conversation so far, are there TWO OR MORE different things the user could be referring to? (e.g. two algorithms discussed → 'that algorithm' is ambiguous; only one topic like 'Python import error' → single referent, NOT ambiguous). If only ONE topic or referent → not ambiguous.\n\n"
            "STEP 3 – Sufficiency: Can you answer the question or rewrite it into a clear, answerable form with current context? (e.g. 'What is the pip install command for the package that fixes Python import errors?' in a Python-import-error conversation: you can answer with a general template and note that the exact package name may be needed; context is sufficient → NOT ambiguous). If you can answer or rewrite → not ambiguous.\n\n"
            "Apply the outcome: If STEP 1 = tied to topic, STEP 2 = single referent (or query already names the topic), STEP 3 = context sufficient → set is_ambiguous=FALSE and clarifying_questions=[].\n"
            "Set is_ambiguous=TRUE only when: query has no link to topic, OR there are 2+ possible referents and you cannot choose, OR context is truly insufficient to answer or rewrite.\n\n"
            "Examples:\n"
            "- Query: 'What is the exact pip install command for the package that fixes Python import errors?' in a conversation about Python import error → topic and intent clear (pip, import errors); one topic; can answer (general command + note about package name) → is_ambiguous=FALSE, clarifying_questions=[].\n"
            "- Query: 'How do I fix it?' when only 'Python import error' was discussed → single referent ('it' = import error) → is_ambiguous=FALSE.\n"
            "- Query: 'Which one should I use?' when both Adam and SGD were discussed → two referents → is_ambiguous=TRUE, add clarifying_questions.\n\n"
            "Your tasks:\n\n"
            "1) is_ambiguous: Result of the step-by-step reasoning above. FALSE when topic is clear, single referent, and context sufficient to answer or rewrite.\n\n"
            "2) clarified_query: REQUIRED. One clarified/rewritten query. When context is enough, rewrite with that context; when already specific, use same or slight refinement. Never leave empty.\n\n"
            "3) clarifying_questions: Only when is_ambiguous=TRUE (e.g. multiple referents). When is_ambiguous=FALSE, return an empty list [].\n\n"
            "4) selected_memory: Relevant snippets from conversation_state, user_context, shared_context, open_threads. Short sentences; do NOT dump full memory.\n\n"
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
        clarifying_questions: Optional[List[str]] = None,
        is_ambiguous: bool = False,
    ) -> str:
        """
        Build final_context: ORIGINAL QUERY, CLARIFIED QUERY, (optional CLARIFYING_QUESTIONS when ambiguous),
        CONVERSATION STATE, SELECTED MEMORY, RECENT MESSAGES.
        When is_ambiguous and clarifying_questions are present, include them so the answer step can use them.
        """
        clarifying_questions = clarifying_questions or []
        original_block = f"ORIGINAL QUERY:\n{query}\n\nCLARIFIED QUERY:\n{clarified_query or query}"
        if is_ambiguous and clarifying_questions:
            q_block = "\n".join(f"- {q}" for q in clarifying_questions)
            original_block += f"\n\nCLARIFYING_QUESTIONS (query was ambiguous; use these to ask the user naturally, not as a rigid list):\n{q_block}"
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
