"""
Streamlit Demo Application
Web-based demo of the chat assistant.
"""

import streamlit as st
import json
import sys
import os
from datetime import datetime

# Project root (parent of demos/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

# Default: save conversation logs under conversation_logger/streamlit/ for easy debug
_DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, "conversation_logger", "streamlit")

from core import ChatAssistant
from utils.conversation_logger import ConversationLogger

# Page config
st.set_page_config(
    page_title="Chat Assistant Demo",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "logger" not in st.session_state:
    st.session_state.logger = None

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    provider = st.selectbox("LLM Provider", ["openai", "anthropic", "gemini"], index=0)
    model = st.text_input("Model (optional)", value="", help="Leave empty for default")
    threshold = st.slider("Token Threshold", 1000, 50000, 10000, 1000)

    st.subheader("Conversation Logging")
    log_path_input = st.text_input(
        "Log file path (directory or file, JSONL)",
        value="conversation_logger/streamlit",
        help="Default: conversation_logger/streamlit — logs are saved here for easy debug. Leave as is or change. If directory, a timestamped file is created inside."
    )

    if st.button("Initialize Assistant"):
        try:
            st.session_state.assistant = ChatAssistant(
                llm_provider=provider,
                llm_model=model if model else None,
                token_threshold=threshold,
                use_tokenizer=True,
                recent_messages_window=20,
                keep_recent_after_summary=5,
            )
            # Always log: use input path or default conversation_logger/streamlit
            raw_path = (log_path_input or "").strip() or "conversation_logger/streamlit"
            log_path = os.path.join(_PROJECT_ROOT, raw_path) if not os.path.isabs(raw_path) else raw_path
            if os.path.isdir(log_path) or not log_path.lower().endswith(".jsonl"):
                os.makedirs(log_path, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = os.path.join(log_path, f"streamlit_conversation_{ts}.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            st.session_state.logger = ConversationLogger(log_path)
            st.success(f"Conversation log: {log_path}")

            st.success("Assistant initialized!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you have set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in .env file")
    
    st.divider()
    
    # Load conversation log
    uploaded_file = st.file_uploader("Load Conversation Log", type=["json", "jsonl"])
    if uploaded_file and st.session_state.assistant:
        content = uploaded_file.read().decode("utf-8")
        # Save to temp file with UTF-8 so Unicode (e.g. → in content) does not raise on Windows
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.jsonl', encoding='utf-8'
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            st.session_state.assistant.load_conversation_log(temp_path)
            st.success("Conversation log loaded!")
        except Exception as e:
            st.error(f"Error loading log: {e}")
        finally:
            os.unlink(temp_path)
    
    st.divider()
    st.session_state.show_analysis = st.checkbox("Show Query Analysis", value=False)

# Main interface
st.title("💬 Chat Assistant Demo")
st.markdown("Chat assistant with **session memory** and **query understanding**")

if not st.session_state.assistant:
    st.warning("⚠️ Please initialize the assistant in the sidebar first.")
    st.info("""
    **Setup Instructions:**
    1. Create a `.env` file in the project root
    2. Add your API key: `OPENAI_API_KEY=your_key_here`, `ANTHROPIC_API_KEY=your_key_here`, or `GEMINI_API_KEY=your_key_here`
    3. Click "Initialize Assistant" in the sidebar
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message and st.session_state.show_analysis:
                with st.expander("🔍 Query Analysis"):
                    metadata = message["metadata"]
                    if "query_understanding" in metadata:
                        qu = metadata["query_understanding"]
                        st.write("**Original Query:**", qu.get("original_query", "N/A"))
                        st.write("**Is Ambiguous:**", "Yes" if qu.get("is_ambiguous", False) else "No")
                        if qu.get("clarified_query"):
                            st.write("**Clarified Query:**", qu["clarified_query"])
                        if qu.get("selected_memory"):
                            st.write("**Selected Memory:**")
                            for snippet in qu["selected_memory"]:
                                st.write(f"- {snippet}")
                        final_ctx = qu.get("final_context")
                        if final_ctx:
                            st.write("**Final Context:**")
                            max_len = 1500
                            preview = final_ctx[:max_len] + (" ... _[truncated]_" if len(final_ctx) > max_len else "")
                            # Insert line breaks before section labels so fields display on separate lines
                            for label in ("CONVERSATION STATE:", "SELECTED MEMORY:", "RECENT MESSAGES:"):
                                preview = preview.replace(f" {label}", f"\n{label}\n")
                            preview = preview.replace("USER QUERY:", "USER QUERY:\n", 1)
                            st.markdown(f"<div style='white-space: pre-wrap; line-height: 1.2;'>{preview}</div>", unsafe_allow_html=True)
                        if qu.get("clarifying_questions"):
                            st.info("💡 Clarifying Questions:")
                            for q_text in qu["clarifying_questions"]:
                                st.write(f"• {q_text}")
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Log user message
        if st.session_state.logger:
            st.session_state.logger.log_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with assistant
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.assistant.process_user_message(prompt)
                    
                    # Display response
                    st.markdown(result["response"])
                    
                    # Add to messages with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "metadata": {
                            "query_understanding": result["query_understanding"].model_dump(),
                            "summary_triggered": result["summary_triggered"],
                            "context_size": result["context_size"]
                        }
                    })

                    # Log assistant response with metadata
                    if st.session_state.logger:
                        metadata = {
                            "query_understanding": result["query_understanding"].model_dump(),
                            "summary_triggered": result["summary_triggered"],
                            "context_size": result["context_size"],
                        }
                        if result["summary"]:
                            metadata["summary_id"] = getattr(result["summary"], "memory_id", None)
                        st.session_state.logger.log_message("assistant", result["response"], metadata=metadata)
                    
                    # Show summary if triggered
                    if result["summary_triggered"] and result["summary"]:
                        st.success("📝 Session memory generated!")
                        with st.expander("View Session Memory"):
                            summary = result["summary"]
                            st.write("**Conversation state:**")
                            st.write(summary.conversation_state)
                            if summary.user_context and (
                                summary.user_context.preferences
                                or summary.user_context.constraints
                                or summary.user_context.goals
                            ):
                                st.write("**User context:**")
                                for p in (summary.user_context.preferences or [])[:5]:
                                    st.write(f"- preference: {p}")
                                for c in (summary.user_context.constraints or [])[:5]:
                                    st.write(f"- constraint: {c}")
                                for g in (summary.user_context.goals or [])[:5]:
                                    st.write(f"- goal: {g}")
                            if getattr(summary, "shared_context", None) and summary.shared_context:
                                st.write("**Shared context:**")
                                for item in summary.shared_context[:5]:
                                    st.write(f"- {item}")
                            if getattr(summary, "open_threads", None) and summary.open_threads:
                                st.write("**Open threads:**")
                                for q in summary.open_threads[:3]:
                                    st.write(f"- {q}")
                    
                    # Show context stats
                    context_size = result["context_size"]
                    threshold = st.session_state.assistant.memory_manager.token_threshold
                    progress = min(context_size / threshold, 1.0)
                    st.progress(progress, text=f"Context: {context_size}/{threshold} tokens ({progress*100:.1f}%)")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check your API key and network connection.")
    
    # Stats sidebar
    if st.session_state.assistant:
        with st.sidebar:
            st.divider()
            st.subheader("📊 Statistics")
            context_size = st.session_state.assistant.memory_manager.get_context_size()
            threshold = st.session_state.assistant.memory_manager.token_threshold
            st.metric("Context Size", f"{context_size:,} tokens")
            st.metric("Threshold", f"{threshold:,} tokens")
            st.metric("Usage", f"{context_size/threshold*100:.1f}%")
            st.metric("Messages", len(st.session_state.assistant.memory_manager.conversation_history))
            
            if st.session_state.assistant.memory_manager.current_summary:
                st.success("✓ Has session summary")
            else:
                st.info("No summary yet")
