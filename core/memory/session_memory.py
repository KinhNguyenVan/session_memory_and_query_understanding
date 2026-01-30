"""
Session Memory Manager

Stores only stable, reusable context that improves future query understanding.
Uses SessionMemory schema (conversation_state, user_context, shared_context, open_threads, scope).
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken

from ..schema.core_schema import SessionMemory, UserContext


class SessionMemoryManager:
    """Manages session memory with automatic summarization"""
    
    def __init__(
        self,
        token_threshold: int = 1000,
        use_tokenizer: bool = True,
        memory_storage_path: str = "memory_storage",
        recent_messages_window: Optional[int] = None,
        keep_recent_after_summary: int = 5
    ):
        """
        Initialize the session memory manager (Core schema only).
        
        Args:
            token_threshold: Maximum tokens before triggering summarization
            use_tokenizer: Whether to use tiktoken for accurate token counting
            memory_storage_path: Directory to store memory files
            recent_messages_window: Number of recent messages to check for threshold.
                                   If None, checks all messages. Only these messages
                                   will be summarized when threshold is exceeded.
            keep_recent_after_summary: Number of recent messages to keep after summarization
                                      for conversational continuity. Set to 0 to remove
                                      all summarized messages and rely only on summary.
        """
        self.token_threshold = token_threshold
        self.use_tokenizer = use_tokenizer
        self.memory_storage_path = memory_storage_path
        self.recent_messages_window = recent_messages_window  # None = check all
        self.keep_recent_after_summary = keep_recent_after_summary
        self.current_summary: Optional[SessionMemory] = None
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize tokenizer if needed
        if self.use_tokenizer:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.use_tokenizer = False
                print("Warning: tiktoken not available, using character-based counting")
        
        # Create memory storage directory
        os.makedirs(self.memory_storage_path, exist_ok=True)
    
    def add_message(self, role: str, content: str, timestamp: Optional[str] = None):
        """Add a message to the conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        self.conversation_history.append(message)
    
    def get_context_size(self, messages: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Calculate context size in tokens for given messages.
        
        Args:
            messages: Messages to calculate size for. If None, uses recent messages
                     based on recent_messages_window, or all if window is None.
        
        Returns:
            Token count
        """
        if messages is None:
            # Use recent messages window if configured
            if self.recent_messages_window is not None:
                messages = self.conversation_history[-self.recent_messages_window:]
            else:
                messages = self.conversation_history
        
        if not messages:
            return 0
        
        # Combine messages into a single string
        full_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        if self.use_tokenizer:
            return len(self.tokenizer.encode(full_context))
        else:
            # Fallback: approximate 1 token = 4 characters
            return len(full_context) // 4
    
    def should_summarize(self) -> bool:
        """
        Check if summarization should be triggered.
        Only checks recent messages (based on recent_messages_window) if configured.
        """
        return self.get_context_size() >= self.token_threshold
    
    def get_recent_messages_for_summarization(self) -> List[Dict[str, Any]]:
        """
        Get the messages that should be summarized.
        Returns recent N messages if window is set, otherwise all messages.
        """
        if self.recent_messages_window is not None:
            return self.conversation_history[-self.recent_messages_window:]
        return self.conversation_history
    
    def summarize_conversation(self, llm_client):
        """
        Generate a comprehensive session summary using LLM.
        Only summarizes recent messages (based on recent_messages_window) if configured.
        
        Args:
            llm_client: LLM client with a generate method
            
        Returns:
            SessionMemory
        """
        # Get messages to summarize (recent N or all)
        messages_to_summarize = self.get_recent_messages_for_summarization()
        
        if not messages_to_summarize:
            raise ValueError("No messages to summarize")
        
        # Calculate range of messages being summarized
        start_idx = len(self.conversation_history) - len(messages_to_summarize)
        end_idx = len(self.conversation_history) - 1
        
        # Prepare conversation text from messages to summarize
        conversation_text = self._format_conversation_for_summarization(messages_to_summarize)
        
        # Generate summary using LLM (Core schema)
        summary_dict = self._generate_summary_with_llm(
            llm_client, 
            conversation_text,
            message_range=(start_idx, end_idx)
        )
        summary = SessionMemory(**summary_dict)
        self.current_summary = summary
        
        # Save to disk
        self._save_summary(summary)
        
        # Clear summarized messages (keep recent ones for context continuity if configured)
        if self.keep_recent_after_summary > 0:
            if len(self.conversation_history) > len(messages_to_summarize) + self.keep_recent_after_summary:
                # Remove summarized messages but keep recent ones
                self.conversation_history = (
                    self.conversation_history[:start_idx] + 
                    self.conversation_history[-self.keep_recent_after_summary:]
                )
            elif len(messages_to_summarize) > self.keep_recent_after_summary:
                # If we summarized recent messages, keep last few
                self.conversation_history = self.conversation_history[-self.keep_recent_after_summary:]
            else:
                # If keep_recent >= messages_to_summarize, keep all (shouldn't happen normally)
                pass
        else:
            # Remove all summarized messages, rely only on summary for context
            if len(self.conversation_history) > len(messages_to_summarize):
                # Keep messages before the summarized range
                self.conversation_history = self.conversation_history[:start_idx]
            else:
                # All messages were summarized, clear history
                self.conversation_history = []
        
        return summary
    
    def _format_conversation_for_summarization(self, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format conversation messages for LLM summarization.
        
        Args:
            messages: Messages to format. If None, uses all conversation history.
        """
        if messages is None:
            messages = self.conversation_history
        
        lines = []
        for i, msg in enumerate(messages):
            lines.append(f"Message {i}: [{msg['role'].upper()}]\n{msg['content']}\n")
        return "\n".join(lines)
    
    def _generate_summary_with_llm(
        self, 
        llm_client, 
        conversation_text: str,
        message_range: Optional[tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Use LLM to generate structured SessionMemory (long-term, by role)."""
        if message_range:
            msg_from = message_range[0]
            msg_to = message_range[1]
        else:
            msg_from = 0
            msg_to = len(self.conversation_history) - 1
        
        system_prompt = (
            "You are an expert at summarizing conversations into a SessionMemory. "
            "Session memory stores only stable, reusable context that improves future query understanding.\n\n"
            "RULES:\n"
            "1. memory_id: short unique id (e.g. mem_001).\n"
            "2. conversation_state: 2–4 sentences – overall understanding of what the conversation is about. "
            "Base context for every query.\n"
            "3. user_context: optional. Use null if none inferred. Otherwise:\n"
            "   - preferences: explicit preferences (e.g. 'prefers dark mode').\n"
            "   - constraints: limitations (e.g. 'budget under 100', 'limited LLM calls').\n"
            "   - goals: user goals inferred (e.g. 'pass take-home test', 'junior-friendly design').\n"
            "4. shared_context: list of facts or assumptions both sides agree on. "
            "One short phrase per item (e.g. 'System uses session summarization', 'Query rewriting reduces ambiguity').\n"
            "5. open_threads: list of unresolved topics that may affect future queries (open questions).\n"
            "6. scope: object with 'from' and 'to' (integers) – which message indices were summarized.\n\n"
            "Output ONLY a valid SessionMemory. Use empty lists [] where nothing applies; do not omit fields."
        )
        user_template = (
            "Summarize the following conversation into a SessionMemory.\n\n"
            "Message range: {msg_from} to {msg_to}.\n\n"
            "Conversation:\n{conversation_text}"
        )
        
        try:
            summary_obj = llm_client.generate_structured(
                SessionMemory,
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "conversation_text": conversation_text,
                    "msg_from": msg_from,
                    "msg_to": msg_to,
                },
            )
            summary_dict = summary_obj.model_dump()
            summary_dict.setdefault("scope", {"from": msg_from, "to": msg_to})
            summary_dict.setdefault(
                "memory_id",
                f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            summary_dict.setdefault("created_at", datetime.now().isoformat())
            return summary_dict
        except Exception as e:
            print(f"Error generating structured summary: {e}")
            return self._create_minimal_summary(message_range)
    
    def _create_minimal_summary(self, message_range: Optional[tuple[int, int]] = None) -> Dict[str, Any]:
        """Create a minimal valid SessionMemory if LLM parsing fails."""
        if message_range:
            msg_from, msg_to = message_range[0], message_range[1]
        else:
            msg_from, msg_to = 0, max(0, len(self.conversation_history) - 1)
        return {
            "memory_id": f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "conversation_state": "Conversation summary (parsing error occurred)",
            "user_context": {"preferences": [], "constraints": [], "goals": []},
            "shared_context": [],
            "open_threads": [],
            "scope": {"from": msg_from, "to": msg_to},
        }
    
    def _save_summary(self, summary: SessionMemory) -> None:
        """Save session memory to disk."""
        filename = f"{self.memory_storage_path}/summary_{summary.memory_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)
    
    def get_memory_context(self) -> Dict[str, Any]:
        """
        Return session memory for query understanding.
        Query-understanding step uses this to fill is_ambiguous, clarified_query,
        selected_memory, final_context (no full dump; select only what is needed).
        """
        if not self.current_summary:
            return {}
        s = self.current_summary
        return {
            "conversation_state": getattr(s, "conversation_state", "") or "",
            "user_context": (
                {
                    "preferences": s.user_context.preferences,
                    "constraints": s.user_context.constraints,
                    "goals": s.user_context.goals,
                }
                if s.user_context else {}
            ),
            "shared_context": list(getattr(s, "shared_context", []) or []),
            "open_threads": list(getattr(s, "open_threads", []) or []),
        }
    
    def get_recent_messages(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent N messages"""
        return self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
