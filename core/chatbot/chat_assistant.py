"""
Chat Assistant: session memory + query understanding + answer generation.

Pipeline: user input → (summarize if needed) → query understanding → answer.
Uses Gemini for all LLM calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import datetime

from ..schema.core_schema import SessionMemory, CoreQueryUnderstanding
from .llm_client import LLMClient
from ..memory.session_memory import SessionMemoryManager
from .query_understanding import QueryUnderstandingPipeline


console = Console()


class ChatAssistant:
    """Main chat assistant with session memory and query understanding."""

    def __init__(
        self,
        llm_provider: str = "gemini",
        llm_model: Optional[str] = None,
        token_threshold: int = 1000,
        use_tokenizer: bool = True,
        recent_messages_window: Optional[int] = None,
        keep_recent_after_summary: int = 5,
    ) -> None:
        """
        Args:
            llm_provider: Must be "gemini" (Gemini-only).
            llm_model: Optional Gemini model name (default: gemini-2.0-flash).
            token_threshold: Token threshold for summarization.
            use_tokenizer: Use tiktoken for token counting.
            recent_messages_window: Number of recent messages for threshold (e.g. 20).
            keep_recent_after_summary: Messages to keep after summarization.
        """
        self.llm_client = LLMClient(provider=llm_provider, model=llm_model)
        self.memory_manager = SessionMemoryManager(
            token_threshold=token_threshold,
            use_tokenizer=use_tokenizer,
            recent_messages_window=recent_messages_window,
            keep_recent_after_summary=keep_recent_after_summary,
        )
        self.query_pipeline = QueryUnderstandingPipeline(self.llm_client)

        console.print("[bold green]Chat Assistant initialized![/bold green]")
        console.print(f"Token threshold: {token_threshold}")
        console.print(f"Gemini model: {llm_model or 'gemini-2.0-flash'}\n")

    # ---------------------------------------------------------------------
    # Main entrypoint
    # ---------------------------------------------------------------------
    def process_user_message(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message: summarize if needed, run query understanding, generate answer.
        Returns dict with response, query_understanding, summary info, context size.
        """
        # IMPORTANT FLOW:
        # 1) First, work only with existing history → decide whether to summarize.
        #    The *current* user_input should NOT be part of the summary that is used
        #    as long-term memory for this turn.
        # 2) After summarization (if any), use:
        #       - summarized memory
        #       - recent raw messages BEFORE this turn
        #    to understand and possibly rewrite the current query.
        # 3) Generate the assistant response based on the rewritten query + context.
        # 4) Finally, append BOTH the user_input and the assistant response into
        #    conversation_history so they become part of future memory.

        # 1) Check if summarization is needed on *existing* history
        summary_triggered = False
        summary = None
        if self.memory_manager.should_summarize():
            console.print(
                "[yellow]⚠ Context threshold exceeded. Triggering summarization...[/yellow]"
            )
            summary = self.memory_manager.summarize_conversation(self.llm_client)
            summary_triggered = True
            console.print("[green]✓ Summarization complete![/green]\n")

        # 2) Build recent_messages: last N (e.g. 20) in chronological order (oldest first, newest last).
        #    Query understanding and final_context rely on this order; do not reverse.
        base_history = list[Dict[str, Any]](self.memory_manager.conversation_history)
        temp_history = base_history + [
            {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()}
        ]
        window = self.memory_manager.recent_messages_window or 20
        recent_messages = temp_history[-window:] if len(temp_history) > window else temp_history

        # Session memory context is derived from the latest summary only
        session_memory_context = self.memory_manager.get_memory_context()
        # If we have a summary (already exceeded threshold), final_context uses only 5 recent; else uses full 20 recent
        has_session_summary = self.memory_manager.current_summary is not None

        # Process query through understanding pipeline (always receives up to 20 recent for LLM)
        query_understanding = self.query_pipeline.process_query(
            query=user_input,
            recent_messages=recent_messages,
            session_memory_context=session_memory_context,
            has_session_summary=has_session_summary,
        )

        # 3) Generate response using augmented context and ambiguity signal
        response = self._generate_response(query_understanding)

        # 4) Now that we have the full turn, append both user and assistant messages
        #    to the conversation history to be used in future summaries.
        self.memory_manager.add_message("user", user_input)
        self.memory_manager.add_message("assistant", response)

        return {
            "response": response,
            "query_understanding": query_understanding,
            "summary_triggered": summary_triggered,
            "summary": summary,
            "context_size": self.memory_manager.get_context_size(),
        }

    # ---------------------------------------------------------------------
    # Response generation
    # ---------------------------------------------------------------------
    def _generate_response(self, query_understanding: CoreQueryUnderstanding) -> str:
        """
        Step 3: Answer using final_context only.
        final_context includes ORIGINAL QUERY + CLARIFIED QUERY + conversation state + selected memory + recent messages.
        Always generate a natural response (no fixed clarification template).
        """
        context = query_understanding.final_context
        prompt = f"""You are a helpful AI assistant. Use the context below to answer.

{context}

Respond naturally. Answer based on the CLARIFIED QUERY. If the query was ambiguous and you need more detail, you may briefly ask for clarification in a natural, conversational way—do not use a rigid list or template. Otherwise give a clear, helpful answer."""

        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            system_prompt=(
                "You are a helpful, knowledgeable AI assistant. "
                "Provide clear, natural responses. Avoid rigid or templated phrasing."
            ),
        )
        return response

    # ---------------------------------------------------------------------
    # Loading logs
    # ---------------------------------------------------------------------
    def load_conversation_log(self, log_path: str) -> None:
        """Load a conversation log from file (JSON or JSONL). Replaces current conversation history."""
        try:
            self.memory_manager.conversation_history.clear()
            self.memory_manager.current_summary = None
            with open(log_path, "r", encoding="utf-8") as f:
                if log_path.endswith(".jsonl"):
                    # JSONL-like format. Be tolerant of multi-line JSON objects
                    # and newlines inside "content" strings.
                    text = f.read()

                    chunks: List[str] = []
                    brace_depth = 0
                    in_string = False
                    escape = False
                    start_idx: Optional[int] = None

                    for i, ch in enumerate(text):
                        if escape:
                            escape = False
                            continue

                        if ch == "\\":
                            escape = True
                            continue

                        if ch == '"' and not escape:
                            in_string = not in_string
                            continue

                        if in_string:
                            continue

                        if ch == "{":
                            if brace_depth == 0:
                                start_idx = i
                            brace_depth += 1
                        elif ch == "}":
                            brace_depth -= 1
                            if brace_depth == 0 and start_idx is not None:
                                chunks.append(text[start_idx : i + 1])
                                start_idx = None

                    if brace_depth != 0:
                        console.print(
                            "[yellow]Warning: Unbalanced braces in conversation log; some entries may be skipped.[/yellow]"
                        )

                    loaded = 0
                    for raw_chunk in chunks:
                        # Replace control characters (including newlines) with spaces
                        sanitized = "".join(
                            " " if ord(c) < 32 and c not in ("\t",) else c
                            for c in raw_chunk
                        )
                        try:
                            msg = json.loads(sanitized)
                        except json.JSONDecodeError:
                            console.print(
                                "[yellow]Warning: Skipped an invalid JSONL entry while loading log.[/yellow]"
                            )
                            continue

                        self.memory_manager.add_message(
                            msg.get("role", "user"),
                            msg.get("content", ""),
                            msg.get("timestamp"),
                        )
                        loaded += 1

                    if loaded == 0:
                        console.print(
                            "[yellow]Warning: No valid messages were loaded from the JSONL log.[/yellow]"
                        )
                else:
                    # Assume JSON array
                    messages = json.load(f)
                    for msg in messages:
                        self.memory_manager.add_message(
                            msg.get("role", "user"),
                            msg.get("content", ""),
                            msg.get("timestamp"),
                        )

            console.print(
                f"[green]✓ Loaded conversation log from {log_path}[/green]"
            )
            console.print(
                f"  Messages loaded: {len(self.memory_manager.conversation_history)}"
            )
            console.print(
                f"  Current context size: {self.memory_manager.get_context_size()} tokens\n"
            )
        except Exception as e:
            console.print(f"[red]Error loading conversation log: {e}[/red]")

    # ---------------------------------------------------------------------
    # Display helpers (CLI)
    # ---------------------------------------------------------------------
    def display_query_understanding(self, query_understanding: CoreQueryUnderstanding) -> None:
        """Display query understanding analysis (Core schema)."""
        console.print("\n[bold cyan]Query Understanding Analysis[/bold cyan]")
        console.print("=" * 60)

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Original Query", query_understanding.original_query)
        table.add_row(
            "Is Ambiguous",
            "Yes" if query_understanding.is_ambiguous else "No",
        )
        if query_understanding.clarified_query:
            table.add_row("Clarified Query", query_understanding.clarified_query)
        console.print(table)

        if query_understanding.selected_memory:
            console.print("\n[bold blue]Selected Memory:[/bold blue]")
            for snippet in query_understanding.selected_memory:
                console.print(f"  • {snippet}")

        if query_understanding.final_context:
            console.print("\n[bold blue]Final Context:[/bold blue]")
            ctx = query_understanding.final_context
            max_len = 800
            ctx_display = ctx[:max_len] + "... [truncated]" if len(ctx) > max_len else ctx
            console.print(ctx_display)

        if query_understanding.clarifying_questions:
            console.print("\n[bold magenta]Clarifying Questions:[/bold magenta]")
            for i, q in enumerate(query_understanding.clarifying_questions, 1):
                console.print(f"  {i}. {q}")

    def display_summary(self, summary: SessionMemory) -> None:
        """Display session memory (conversation_state, user_context, shared_context, open_threads)."""
        console.print("\n[bold green]Session Memory[/bold green]")
        console.print("=" * 60)
        console.print(f"[bold]Conversation state:[/bold] {summary.conversation_state}\n")

        if summary.user_context and (
            summary.user_context.preferences
            or summary.user_context.constraints
            or summary.user_context.goals
        ):
            console.print("[bold]User context:[/bold]")
            for p in (summary.user_context.preferences or [])[:5]:
                console.print(f"  • preference: {p}")
            for c in (summary.user_context.constraints or [])[:5]:
                console.print(f"  • constraint: {c}")
            for g in (summary.user_context.goals or [])[:5]:
                console.print(f"  • goal: {g}")
            console.print()

        if summary.shared_context:
            console.print("[bold]Shared context:[/bold]")
            for item in summary.shared_context[:5]:
                console.print(f"  • {item}")

        if summary.open_threads:
            console.print("\n[bold]Open threads:[/bold]")
            for q in summary.open_threads[:5]:
                console.print(f"  • {q}")

