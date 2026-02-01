# Chat Assistant

Session memory and query understanding for a conversational assistant. **Google Gemini** only; Pydantic schemas (SessionMemory, CoreQueryUnderstanding).

---

## Setup

**Prerequisites:** Python 3.10+, Google Gemini API key

1. Enter project and install:
   ```bash
   cd chatbot-assistant
   pip install -r requirements.txt
   ```

2. Configure `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   ```
   (`GOOGLE_API_KEY` is also accepted.)

3. Verify (optional):
   ```bash
   python utils/validate_setup.py
   ```

---

## How to Run

### Three test scenarios (CLI)

Log is **always on**. Each run creates a **new** log file: `conversation_logger/cli/cli_YYYYMMDD_HHMMSS.jsonl` (date + time). Refresh or run again = new file.

| Scenario | Command | Expectation |
|----------|---------|-------------|
| **1. Fresh session** | `python demos/demo_cli.py` | New log file each run; all messages are logged. |
| **2. Session memory** | `python demos/demo_cli.py --load-log test_data/conversation_1_long.jsonl --verbose` | Load long conversation; next message can trigger summarization. `final_context` uses 5 recent (memory covers rest). |
| **3. Query understanding** | `python demos/demo_cli.py --load-log test_data/conversation_2_ambiguous.jsonl --verbose` | Load ambiguous conversation; see is_ambiguous, clarified_query, selected_memory. `final_context` uses up to 20 recent (chronological). |

Use `--load-log PATH` to load a specific file (e.g. test data) and seed the new log with that history. Use `--log-file PATH` (file or directory) to change where the log is saved.

**In-session:** type message → send; `exit` / `quit` / `q` → quit; `summary` → trigger summarization; `stats` → context size and token count.

### Streamlit

```bash
streamlit run demos/demo_streamlit.py
```

1. Sidebar: set Gemini model (optional), token threshold, log path (default `conversation_logger/streamlit/`).
2. Click **Initialize Assistant**, then chat.
3. Optional: **Load Conversation Log** (upload JSON/JSONL) for scenarios 2 or 3.
4. Enable **Show Query Analysis** to see query understanding and final context.

---

## Conversation logging

- **CLI:** always logs; each run = new file `conversation_logger/cli/cli_YYYYMMDD_HHMMSS.jsonl`. Optional: `--log-file PATH`, `--load-log PATH` (load a file and seed the new log).
- **Streamlit:** logs to `conversation_logger/streamlit/` (or custom path in sidebar).

**Format (JSONL):** one JSON per line — `role`, `content`, `timestamp`; assistant lines may include `metadata` (query_understanding, summary_triggered, context_size, summary_id).

When you load a log, the session log file is seeded with that history so the same data backs both 20 recent messages and persistence (no duplicate seeding).

---

## Test data

| File | Use |
|------|-----|
| `test_data/conversation_1_long.jsonl` | Scenario 2: long → summarization |
| `test_data/conversation_2_ambiguous.jsonl` | Scenario 3: ambiguous → query understanding |
| `test_data/conversation_3_mixed.jsonl` | Mixed topics |
| `test_data/conversation_4_technical.jsonl` | Technical (e.g. backpropagation) |

Load via `--load-log` (CLI) or **Load Conversation Log** (Streamlit).

### Example questions for testers

Use after loading each file. Under step-by-step logic: **ambiguous** = 2+ possible referents or context insufficient to answer/rewrite; **clear** = single referent or query names topic, context sufficient. Expect: 2 questions → is_ambiguous Yes + clarifying_questions; 1 question → is_ambiguous No + empty clarifying_questions.

**conversation_1_long.jsonl** (recommendation system: algorithms, metrics, cold start, hybrid)

- Ambiguous: *Which one should I use?* (algorithm? metric? approach? — multiple referents)
- Ambiguous: *What would you recommend for that?* (“that” could be cold start, evaluation, or hybrid — multiple referents)
- Clear: *What evaluation metrics should I use for my recommendation system?* (topic and intent explicit)

**conversation_2_ambiguous.jsonl** (Python import error; several fixes mentioned: pip install, path, directory)

- Ambiguous: *Which method should I try first?* (pip install vs check path vs directory — multiple referents)
- Ambiguous: *Is that enough or do I need something else?* (“that” = which suggestion? — multiple referents)
- Clear: *What is the exact pip install command for the package that fixes Python import errors?* (topic and intent explicit; single topic in context)

**conversation_3_mixed.jsonl** (Japan: Tokyo, Kyoto, JR Pass, Suica, hotels, food)

- Ambiguous: *Should I get one?* (one = JR Pass? Suica? — multiple referents)
- Ambiguous: *What about that?* (“that” = transport? hotel? food? — multiple referents)
- Clear: *How many days should I spend in Tokyo versus Kyoto for a first-time visit?* (topic explicit)

**conversation_4_technical.jsonl** (backpropagation, Adam, SGD, learning rate, batch size)

- Ambiguous: *Which one should I use?* (Adam vs SGD — two referents)
- Ambiguous: *Will my system be optimized if I choose this one?* (“this one” = Adam? SGD? — multiple referents)
- Clear: *What is the difference between Adam and SGD in how they use backpropagation gradients?* (topic explicit)

---

## Design

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

### Session memory

conversation_state, user_context, shared_context, open_threads, scope. Created when recent window exceeds token threshold; after summarization, only last 5 recent kept in buffer.

### Query understanding

is_ambiguous, clarified_query (always required), clarifying_questions, selected_memory. **final_context** is built in code: ORIGINAL QUERY + CLARIFIED QUERY + CONVERSATION STATE + SELECTED MEMORY + RECENT MESSAGES (5 if memory exists, up to 20 if not; chronological when no memory).

### Code layout

```
core/
├── schema/core_schema.py
├── memory/session_memory.py
├── chatbot/
│   ├── llm_client.py
│   ├── query_understanding.py
│   └── chat_assistant.py
demos/
├── demo_cli.py
└── demo_streamlit.py
utils/
├── conversation_logger.py
└── validate_setup.py
```

---

## Assumptions and limitations

- **Assumptions:** Gemini only; `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env`; tiktoken for token count when available; session memory persisted under `memory_storage/`.
- **Limitations:** Quality depends on LLM; Pydantic output may occasionally need fallbacks; no embeddings; single-session (multi-session would need session_id and per-session storage).

---

## Quick reference

| Scenario | CLI |
|----------|-----|
| Fresh session (new log file each run) | `python demos/demo_cli.py` |
| Session memory (load test file) | `python demos/demo_cli.py --load-log test_data/conversation_1_long.jsonl --verbose` |
| Query understanding (load test file) | `python demos/demo_cli.py --load-log test_data/conversation_2_ambiguous.jsonl --verbose` |

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env` before running.
