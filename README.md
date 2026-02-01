# Chat Assistant with Session Memory and Query Understanding

A chat assistant backend with **session memory** (conversation summarization) and **query understanding** (ambiguity detection, query rewriting, context selection). Uses **Google Gemini** as the sole LLM and Pydantic schemas (SessionMemory, CoreQueryUnderstanding).

---

## 1. Setup

### Prerequisites

- Python 3.10+
- **Google Gemini API key** (required)

### Installation

1. **Clone or enter the project directory**
   ```bash
   cd chatbot-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env` (or create `.env`).
   - Set your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   ```
   Alternatively, `GOOGLE_API_KEY` is also accepted.

4. **Verify setup (optional)**
   ```bash
   python utils/validate_setup.py
   ```
   This checks that `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set and that required packages are installed.

---

## 2. How to Run

The project runs **Gemini only**. Use the CLI or Streamlit demo.

### 2.1 Run Modes: Three Cases

| Case | Purpose | CLI command | What to expect |
|------|---------|-------------|-----------------|
| **1. Fresh chat** | Start from scratch, no prior conversation | `python demos/demo_cli.py` | Normal chat; logs saved to `conversation_logger/cli/` by default. When context exceeds the token threshold, summarization runs and session memory is created. |
| **2. Session memory** | Load a long conversation so summarization is triggered | `python demos/demo_cli.py --load-log test_data/conversation_1_long.jsonl --verbose` | After load, context is large; next message can trigger summarization. You see Session Memory (conversation_state, user_context, shared_context, open_threads). `final_context` uses **5 recent messages** (memory covers the rest). |
| **3. Query understanding** | Load a conversation with ambiguous queries | `python demos/demo_cli.py --load-log test_data/conversation_2_ambiguous.jsonl --verbose` | No summarization yet (under threshold). You see query analysis: is_ambiguous, clarified_query, clarifying_questions, selected_memory. `final_context` uses **up to 20 recent messages** in chronological order (no memory). |

For scenarios 2 and 3, use `--load-log` with the test file and `--verbose` to see query analysis and session memory. In-session: type a message to send; `exit` / `quit` / `q` to quit; `summary` to trigger summarization; `stats` for context size and token count.

---

### 2.2 Streamlit Demo

```bash
streamlit run demos/demo_streamlit.py
```

1. In the sidebar: set **Gemini model** (optional) and **token threshold**.
2. **Log file path** defaults to `conversation_logger/streamlit/`. Click **Initialize Assistant** to start; logs are written there (e.g. `streamlit_conversation_YYYYMMDD_HHMMSS.jsonl`).
3. Optionally **Load Conversation Log** (upload a JSON/JSONL file) to reproduce Case 2 or Case 3.
4. Chat; enable **Show Query Analysis** to see query understanding and final context.

---

## 3. Conversation Logging

Both CLI and Streamlit save conversation history (each turn + metadata) to `conversation_logger/`.

### Layout

```
conversation_logger/
├── cli/          # From demos/demo_cli.py
│   └── cli_YYYYMMDD_HHMMSS.jsonl
└── streamlit/    # From demos/demo_streamlit.py
    └── streamlit_conversation_YYYYMMDD_HHMMSS.jsonl
```

### Log format (JSONL)

Each line is a JSON object:

- `role`: `"user"` or `"assistant"`.
- `content`: message text.
- `timestamp`: ISO timestamp.
- `metadata` (optional, on assistant messages): query_understanding, summary_triggered, context_size, summary_id (memory_id).

Use `--no-log` (CLI) or a custom path to disable or redirect logs.

---

## 4. Test Data

| File | Purpose |
|------|---------|
| `test_data/conversation_1_long.jsonl` | Long conversation → triggers summarization (Case 2). |
| `test_data/conversation_2_ambiguous.jsonl` | Ambiguous queries → query understanding, rewrite, clarifying questions (Case 3). |
| `test_data/conversation_3_mixed.jsonl` | Mixed topics. |
| `test_data/conversation_4_technical.jsonl` | Technical discussion. |

Format: one JSON per line with `role`, `content`, `timestamp`. Load via `--load-log` (CLI) or **Load Conversation Log** (Streamlit).

---

## 5. Design Overview

### Pipeline

```
20 recent messages (chronological)
        ↓
If over token threshold → Summarize (Gemini) → SessionMemory
        ↓
User query + session memory (if any) + recent messages
        ↓
Query Understanding (Gemini) → is_ambiguous, clarified_query, selected_memory, final_context
        ↓
final_context → Answer (Gemini)
```

### Session Memory (SessionMemory)

- **conversation_state**: Overall understanding of the conversation.
- **user_context**: preferences, constraints, goals.
- **shared_context**: Facts or assumptions both sides agree on.
- **open_threads**: Unresolved topics.
- **scope**: Range of messages summarized.

Session memory is created when the **recent message window** (e.g. 20 messages) exceeds the **token threshold**. After summarization, only the last **5 recent messages** are kept in the conversation buffer; the rest are represented by the summary.

### Query Understanding (CoreQueryUnderstanding)

- **is_ambiguous**: Whether the query is ambiguous in the current context.
- **clarified_query**: Rewritten query when memory is enough to disambiguate.
- **clarifying_questions**: Questions to ask when the query stays ambiguous.
- **selected_memory**: Memory snippets relevant to this query (no full dump).
- **final_context**: Built in code (not by the LLM). Always includes:
  - USER QUERY (clarified or original)
  - CONVERSATION STATE (from session memory)
  - SELECTED MEMORY
  - RECENT MESSAGES: **5** when there is session memory; **up to 20** when there is no memory (chronological order, enforced by timestamp when no memory).

### Code layout

```
core/
├── schema/core_schema.py    # SessionMemory, UserContext, CoreQueryUnderstanding
├── memory/session_memory.py # Summarization, get_memory_context
├── chatbot/
│   ├── llm_client.py        # Gemini client
│   ├── query_understanding.py # Query understanding + final_context build
│   └── chat_assistant.py    # Orchestration: memory → query understanding → answer
demos/
├── demo_cli.py              # CLI (Gemini only)
└── demo_streamlit.py        # Streamlit (Gemini only)
utils/
├── conversation_logger.py   # JSONL writer
└── validate_setup.py        # Check API key and dependencies
```

---

## 6. Assumptions and Limitations

### Assumptions

- **Gemini only**: No OpenAI or Anthropic; set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env`.
- Token counting: tiktoken when available; character-based fallback otherwise.
- Session memory can be persisted to disk (`memory_storage/`).
- Augmented context fits within the model’s context window.

### Limitations

- Quality depends on the LLM (ambiguity, rewrite, summary).
- Structured output (Pydantic): the LLM may occasionally return invalid format; minimal fallbacks exist.
- Memory retrieval is simple (no embeddings).
- Single-session oriented; multi-session would require session_id and per-session storage.

---

## 7. Quick Reference (test scenarios)

| Scenario | CLI command |
|----------|-------------|
| 1. Fresh chat | `python demos/demo_cli.py` |
| 2. Session memory | `python demos/demo_cli.py --load-log test_data/conversation_1_long.jsonl --verbose` |
| 3. Query understanding | `python demos/demo_cli.py --load-log test_data/conversation_2_ambiguous.jsonl --verbose` |

---

**Note:** Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env` before running the demos.
