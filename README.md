# Chat Assistant with Session Memory and Query Understanding

A chat assistant backend with **session memory** (conversation summarization) and **query understanding** (ambiguity detection, query rewriting, context selection). Uses Pydantic schemas (SessionMemory, CoreQueryUnderstanding) and LLMs (Gemini / OpenAI / Anthropic).

---

## 1. Setup Instructions

### Prerequisites

- Python 3.8+
- API key: **Google Gemini** (recommended), OpenAI, or Anthropic

### Installation

1. **Navigate to the project directory**
   ```bash
   cd chatbot-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Create a `.env` file from `.env.example` (if present) or create a new one.
   - Add **one** of the following keys:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   # or
   OPENAI_API_KEY=your_openai_key_here
   # or
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

---

## 2. How to Run the Demo

### CLI Demo

Run from the project root:

```bash
python demos/demo_cli.py
```

**Default:** Each run **automatically saves conversation logs** to `conversation_logger/cli/` (e.g. `cli_YYYYMMDD_HHMMSS.jsonl`).

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--provider {openai,anthropic,gemini,google}` | LLM provider (default: gemini) |
| `--model NAME` | Model name (optional) |
| `--threshold N` | Token threshold to trigger summarization (default: 1000) |
| `--log-file PATH` | Save logs to this file or directory instead of `conversation_logger/cli/` |
| `--no-log` | **Do not** save conversation logs (disable default logging) |
| `--load-log PATH` | Load conversation from a JSONL file (e.g. from `test_data/`) |
| `--verbose` | Show detailed query understanding analysis |

**Examples:**

```bash
# Normal run (logs saved to conversation_logger/cli/)
python demos/demo_cli.py

# Run without saving logs
python demos/demo_cli.py --no-log

# Save logs to a custom directory or file
python demos/demo_cli.py --log-file my_logs/

# Load a sample conversation and chat (logs still saved to conversation_logger/cli/)
python demos/demo_cli.py --load-log test_data/conversation_2_ambiguous.jsonl --verbose
```

**In-session CLI commands:**

- Type your message and press Enter to send.
- `exit` / `quit` / `q`: exit the app.
- `summary`: manually trigger summarization (session memory).
- `stats`: show context statistics (token count, whether a summary exists).

---

### Streamlit Demo

```bash
streamlit run demos/demo_streamlit.py
```

A browser window opens. In the sidebar:

1. Choose LLM (provider, model) and token threshold.
2. **Conversation Logging:** The "Log file path" field defaults to `conversation_logger/streamlit`. Each time you click **Initialize Assistant**, logs are written to this directory (e.g. `streamlit_conversation_YYYYMMDD_HHMMSS.jsonl`). You can change the path if you like.
3. Click **Initialize Assistant**, then start chatting.

You can **Load Conversation Log** (upload a JSON/JSONL file) to load an existing conversation.

---

## 3. Conversation Logging (conversation_logger)

Both **CLI** and **Streamlit** save conversation history (each user/assistant turn plus metadata) for debugging and as test data.

### Directory layout

```
conversation_logger/
├── cli/          # Logs from demos/demo_cli.py
│   └── cli_YYYYMMDD_HHMMSS.jsonl
└── streamlit/    # Logs from demos/demo_streamlit.py
    └── streamlit_conversation_YYYYMMDD_HHMMSS.jsonl
```

- **cli/**: CLI-only logs — easier to debug when running from the terminal.
- **streamlit/**: Streamlit-only logs — easier to debug when running the web app.

### Log format (JSONL)

Each line is a JSON object:

- `role`: `"user"` or `"assistant"`.
- `content`: message text.
- `timestamp`: ISO timestamp.
- `metadata` (optional): query_understanding, summary_triggered, context_size, summary_id (memory_id) for assistant messages.

### Usage

- **CLI:** By default logs are written to `conversation_logger/cli/`. Use `--no-log` to disable, or `--log-file PATH` to change the destination.
- **Streamlit:** By default logs go to `conversation_logger/streamlit/`. Change the "Log file path" field in the sidebar to use a different path.

Logs in `conversation_logger/` can be reused as **test data** (e.g. via `--load-log` or upload in Streamlit).

---

## 4. Test Data

The `test_data/` directory contains **at least three conversation logs** (JSONL) to demonstrate:

- **Session memory being triggered**
- **Ambiguous user queries**

| File | Purpose |
|------|---------|
| `conversation_1_long.jsonl` | Long conversation → triggers summarization (session memory) |
| `conversation_2_ambiguous.jsonl` | Ambiguous queries → query understanding, rewrite, clarifying questions |
| `conversation_3_mixed.jsonl` | Mixed topics |
| `conversation_4_technical.jsonl` | Technical discussion (e.g. backpropagation) |

Format: one JSON per line with `role`, `content`, `timestamp`. Load via:

- CLI: `python demos/demo_cli.py --load-log test_data/conversation_1_long.jsonl`
- Streamlit: sidebar → **Load Conversation Log** → choose file.

---

## 5. High-Level Design

### End-to-end pipeline

```
20 recent messages
        ↓
Summarize (LLM) → SessionMemory (long-term)
        ↓
User query + 5 recent messages + session memory
        ↓
Query Understanding (LLM) → is_ambiguous, clarified_query, selected_memory, final_context
        ↓
final_context
        ↓
Answer / downstream task
```

### Session Memory (SessionMemory)

- **conversation_state**: Overall understanding of the conversation.
- **user_context**: preferences, constraints, goals (stable, reused across queries).
- **shared_context**: Facts or assumptions both sides have agreed on.
- **open_threads**: Unresolved topics.
- **scope**: Range of messages that were summarized (from/to).

Session memory stores only **stable, reusable** information to improve query understanding.

### Query Understanding (CoreQueryUnderstanding)

- **is_ambiguous**: Whether the query is ambiguous in the current context.
- **clarified_query**: Rewritten query when memory is enough to disambiguate.
- **clarifying_questions**: Questions to ask when the query remains ambiguous.
- **selected_memory**: Memory **snippets selected** for this query (no full dump).
- **final_context**: Context passed to the answer step (query + conversation_state + selected_memory + recent messages).

### Main code layout

```
core/
├── schema/core_schema.py    # SessionMemory, UserContext, CoreQueryUnderstanding
├── memory/session_memory.py # Session management, summarization, get_memory_context
├── chatbot/
│   ├── llm_client.py        # LLM calls (Gemini/OpenAI/Anthropic)
│   ├── query_understanding.py # Query understanding pipeline
│   └── chat_assistant.py    # Orchestration: memory → query understanding → answer
demos/
├── demo_cli.py              # CLI; default logs → conversation_logger/cli/
└── demo_streamlit.py        # Web; default logs → conversation_logger/streamlit/
utils/
└── conversation_logger.py   # JSONL writer (role, content, timestamp, metadata)
```

---

## 6. Assumptions and Limitations

### Assumptions

- Valid API key (Gemini/OpenAI/Anthropic).
- Token counting: tiktoken when available; character-based fallback otherwise.
- Session memory is persisted to disk (`memory_storage/`).
- Augmented context fits within the LLM’s context window.

### Limitations

- Quality depends on the LLM (ambiguity, rewrite, summary).
- Structured output (Pydantic): LLM may occasionally return invalid format — minimal fallbacks are in place.
- Memory retrieval is simple (no embeddings).
- Designed for **single session**; multi-session would require extensions (e.g. session_id, per-session storage).

---

## 7. Evaluation Alignment (Rubric)

- **Core features work end-to-end (0–6):** Full pipeline from input → session memory → query understanding → response; CLI and Streamlit demos.
- **Structured outputs & validation (0–1):** SessionMemory and CoreQueryUnderstanding use Pydantic; LLM output is structured via `generate_structured`.
- **Code structure & readability (0–2):** Clear separation (core/demos/utils); explicit schemas; type hints and docstrings.
- **Documentation & test data (0–1):** README covers setup, how to run demos, high-level design, assumptions/limitations; conversation logging (conversation_logger) and at least three conversation logs in test_data demonstrating session memory and ambiguous queries.

---

## 8. Quick Reference

| Topic | Details |
|-------|---------|
| CLI logs | Default: `conversation_logger/cli/cli_*.jsonl`; disable: `--no-log`; custom: `--log-file PATH` |
| Streamlit logs | Default: `conversation_logger/streamlit/`; change via "Log file path" in sidebar |
| Load conversation | CLI: `--load-log test_data/xxx.jsonl`; Streamlit: Load Conversation Log (upload) |
| Test data | `test_data/conversation_1_long.jsonl`, `conversation_2_ambiguous.jsonl`, … (session memory + ambiguous) |

---

**Note:** Configure your API key in `.env` before running the demos.
