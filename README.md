# ResolveAI – Intelligent Customer Support Platform

A Streamlit customer support demonstration combining knowledge-grounded AI answers with explicit escalation, local support-ticket creation, ticket lookup, and status management.

> ResolveAI uses fictional **CloudSync Pro** demonstration data. It has no authentication or automatic human-agent notifications and is not production-ready. Conversation memory is deferred; questions are processed independently.

## 1. Project Overview

Customers often need answers spread across FAQs, product manuals, and previous support resolutions. Support teams also need a way to retain issues the chatbot cannot confidently answer.

ResolveAI retrieves relevant knowledge-base documents, generates cited answers, and routes insufficient-context questions to an escalation response. Customers can then explicitly save a local SQLite support ticket. The interface provides three views: **AI Support Chat**, **My Tickets**, and **Support Dashboard**.

The application extends The Gen Academy's CloudSync Pro support-bot reference implementation. The hybrid RAG backend is inherited; the ResolveAI interface and SQLite ticket workflows are independent enhancements described below.

## 2. Business Use Case

| User | Demonstration workflow |
| --- | --- |
| Customer | Ask a support question or select a Quick Help suggestion, inspect cited sources, and create a ticket if the bot escalates. Save the returned ticket ID and use My Tickets to check its status. |
| Support agent | Open the Support Dashboard, filter tickets, select an issue, review its customer details and escalation reason, and explicitly update its status. |
| Support manager | Review total, Open, In Progress, and Resolved counts to understand the current local queue. These are current-state counts, not SLA, response-time, or historical performance analytics. |

These are intended workflows, not enforced roles. Anyone accessing the application can open the administrative dashboard. Creating a ticket does not assign an agent, send an email, or schedule a response.

## 3. Features

### AI Support Chat

- Navy, teal, and light-background Streamlit interface with consistent sidebar navigation.
- Six Quick Help categories: Account & Sign In, Security & Privacy, File Sharing, Billing & Plans, Troubleshooting, and Talk to Support.
- Browse Help Topics during a conversation without clearing previous messages; selecting a suggestion collapses the topic browser and appends a new exchange.
- Session-based display history, inline chat input, and **+ New Chat** to clear chat history, pending suggestions, and topic expansion state.
- Switching pages preserves the chat within the current Streamlit session. Display history is **not** supplied to the model as conversational memory.

### Sources and confidence-based escalation

Answers are prompted to cite source IDs inline, such as `[faq-003]`. Expandable panels expose retrieved documents, cosine similarity, route decisions, judge usage, and escalation reasons. Retrieved candidates are not necessarily sufficient evidence for an answer.

The inherited routing policy uses the best candidate's similarity:

| Similarity | Action |
| --- | --- |
| At least `0.55` | Generate an answer from retrieved context. |
| At least `0.35`, below `0.55` | Ask GPT-4o-mini whether the best candidate directly answers the question; answer on YES, otherwise escalate. |
| Below `0.35` | Escalate without answer generation. |

The answer model can also return `INSUFFICIENT_CONTEXT`, triggering escalation. Similarity is a context-match signal, not an answer-accuracy percentage. These checks reduce unsupported responses but cannot guarantee correctness.

### SQLite support tickets

An escalated response offers **Create Support Ticket**; no ticket is created automatically. A form requires a nonempty customer name and shows the original question as a read-only issue description. Successful submission stores the name, original question, escalation reason, UTC timestamp, UUID ticket ID, and initial status **Open**.

The saved ID replaces that escalation's form with a confirmation. Each escalation retains its own question and confirmation in session history, preventing ordinary reruns from creating duplicate tickets for that turn. Storage failures display a helpful error rather than a success claim.

### My Tickets

Customers enter one ticket ID to view its full ID, customer name, issue description, UTC creation date, and current status. Missing IDs receive a friendly message. **Clear Search** resets the input and displayed result without modifying SQLite. This page does not list every customer's tickets or allow status changes.

Ticket-ID lookup is a local demonstration, not secure customer authentication.

### Support Dashboard

The dashboard provides four overview cards, a status filter, an abbreviated ticket table, a full-ID selector, complete ticket details, and an explicit **Update Status** action. Valid statuses are **Open**, **In Progress**, and **Resolved**. Successful updates reload the ticket list and counts. Abbreviations affect presentation only; storage operations use full IDs.

## 4. Architecture

```text
Fictional JSON records
        |
        v
LangChain Documents (title + content; id/source/title metadata)
        |
        +--> HuggingFace embeddings --> in-memory FAISS semantic index
        |
        +--> BM25 keyword index
                         |
Customer question --> weighted reciprocal rank fusion
                         |
                  cosine similarity scoring
                         |
                  LangGraph grade node
                    /             \
                 answer         escalate
                    |               |
           GPT-4o-mini +       explicit customer
           source citations    ticket submission
                                    |
                               local SQLite
                                    |
                        My Tickets / Dashboard
```

1. **Document loading:** `knowledge_base.py` loads `data/faqs.json`, `data/tickets.json`, and `data/manuals.json`. Each record becomes one document; title and content are concatenated. There is no separate chunking pipeline.
2. **Embeddings and semantic retrieval:** `HuggingFaceEmbeddings` uses `sentence-transformers/all-MiniLM-L6-v2`. `FAISS.from_documents()` builds the vector index locally in memory.
3. **Keyword retrieval:** `BM25Retriever` supplies exact-token matching alongside semantic retrieval. Each base retriever requests four candidates.
4. **Fusion:** LangChain's `EnsembleRetriever` combines rankings using weighted reciprocal rank fusion: BM25 weight `0.4`, semantic weight `0.6`. The merged result is not necessarily limited to four documents.
5. **Scoring:** `score_query()` embeds the query and candidate document contents, explicitly normalizes vectors, and computes cosine similarity with NumPy. Candidates are then sorted by similarity. The routing score is separate from the rank-fusion score.
6. **Routing and generation:** LangGraph runs `START → retrieve → grade → answer/escalate → END`. Both the answer model and the borderline-context judge use GPT-4o-mini with temperature `0`. The answer node can reroute to escalation if it flags insufficient context.
7. **Ticket storage:** Streamlit calls the separate storage module only through explicit ticket actions and ticket-page reads. Tickets are not added to the JSON knowledge base or retrieval indexes.

**Caching:** `app.py` wraps `get_bot()` with `st.cache_resource`. This reuses the constructed bot, graph, embedding model, and retriever/index resources across Streamlit reruns and sessions in the running process. Chat history remains in per-session state. There is no application-level on-disk FAISS persistence; clearing the resource cache or restarting the process rebuilds the index. Candidate documents are re-embedded during similarity scoring. Ticket pages route independently of bot initialization and do not require OpenAI calls.

## 5. Technology Stack

| Library / component | Responsibility |
| --- | --- |
| Python | Application and storage logic; Python 3.10+ syntax is used. Python 3.12 is a reasonable local starting point. |
| `streamlit` | Chat UI, forms, navigation, tables, metrics, session state, and resource caching. |
| `langchain` / `langchain-core` | Ensemble retrieval, document and message abstractions; core is installed transitively. |
| `langchain-community` | BM25 retriever and FAISS vector-store integrations. |
| `langchain-huggingface`, `sentence-transformers` | Local sentence embeddings. |
| `faiss-cpu` | Semantic vector search. |
| `rank-bm25` | Keyword ranking used by the BM25 integration. |
| `numpy` | Vector normalization and cosine similarity calculations. |
| `langgraph` | Retrieve, grade, answer, and escalation workflow. |
| `langchain-openai` | GPT-4o-mini answer generation and context judging. |
| Standard-library `sqlite3`, `uuid`, `datetime` | Persistent tickets, unique IDs, and UTC timestamps. |
| Standard-library `unittest`, `tempfile`, `unittest.mock` | Isolated automated storage tests. |

Dependency ranges are defined in `requirements.txt`; it is not an exact-version lock file.

## 6. Project Structure

```text
starter-repo/
├── README.md                    # Project documentation
├── app.py                       # ResolveAI's three Streamlit views
├── bot.py                       # Reference LangGraph workflow and LLM prompts
├── retriever.py                 # Reference FAISS + BM25 retrieval and scoring
├── knowledge_base.py            # Reference JSON-to-document loader
├── ticket_store.py              # SQLite creation, lookup, listing, status updates
├── test_ticket_store.py         # 16 isolated storage unit tests
├── requirements.txt             # Python dependency ranges
├── data/
│   ├── faqs.json                # Fictional product FAQs
│   ├── manuals.json             # Fictional product manuals
│   └── tickets.json             # Fictional historical resolutions used for RAG
└── .gitignore                   # Excludes databases, environments, and secrets
```

At runtime, `resolveai_tickets.db` is created beside `ticket_store.py` when database initialization is needed. Its absolute path is based on the module location, not the terminal's current directory. It is local application data and must not be submitted with real ticket records. The JSON historical tickets and this runtime database serve different purposes.

## 7. Local Setup — Windows PowerShell

Prerequisites: a compatible Python installation (Python 3.12 is a reasonable starting point), internet access for dependencies and the initial embedding-model download, Git if cloning from the terminal, and an OpenAI API key with access to the configured model. The Windows `py` launcher is optional. API requests may incur charges.

Clone the ResolveAI repository and enter the cloned directory:

```powershell
git clone https://github.com/Suhasini0812/ResolveAI-Customer-Support-RAG.git
Set-Location .\ResolveAI-Customer-Support-RAG
```

The following commands assume this directory contains `app.py` and `requirements.txt`. In a checkout that instead contains these files under `starter-repo`, first run `Set-Location .\starter-repo`. Run all remaining setup, testing, and launch commands from that same application directory. Create `.venv` there, inside the cloned repository, rather than in its parent directory.

Choose **one** way to create the virtual environment:

**Python available as `python` (no launcher required):**

```powershell
python --version
python -m venv .venv
```

Ensure this resolves to a compatible installed Python interpreter. If `python` opens the Microsoft Store, is not found, or selects the wrong interpreter, use its executable path directly instead.

**Installed Python executable by full path:**

```powershell
$pythonExecutable = Read-Host "Full path to your installed python.exe"
& $pythonExecutable --version
& $pythonExecutable -m venv .venv
```

PowerShell's `&` call operator supports executable paths containing spaces. Enter the actual path without surrounding quotation marks at the prompt.

**Python managed by uv, if uv is already installed:**

Python may be installed through uv rather than the traditional Windows installer. Such an interpreter may not provide the `py` launcher or a `python` command on PATH. To install/select Python 3.12 and invoke that interpreter directly:

```powershell
uv python install 3.12
$pythonExecutable = uv python find 3.12
& $pythonExecutable -m venv .venv
```

Alternatively, if the Windows launcher is available, `py -3.12 -m venv .venv` creates the same environment location. Do not run every creation option; use the one appropriate to your Python installation.

After creating `.venv`, use its interpreter for every package, test, and application command:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the repository-local virtual environment directly, so activation and execution-policy changes are unnecessary. The testing and Streamlit commands below use this same `.\.venv\Scripts\python.exe`; no parent-directory environment or global pip is required.

Set the API key for the current terminal session using a masked prompt:

```powershell
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "OpenAI API key" -AsSecureString)).Password
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The application reads `OPENAI_API_KEY` from its environment; it does not automatically load a `.env` file. Do not print or commit the key. The project directory must be writable for SQLite ticket operations. No separate database service or manual schema migration is required.

To remove the key from that terminal after stopping the app:

```powershell
Remove-Item Env:OPENAI_API_KEY
```

## 8. Testing

### Automated SQLite tests

From the same application directory used for setup (containing `app.py`, `requirements.txt`, and `.venv`):

```powershell
.\.venv\Scripts\python.exe -m unittest test_ticket_store.py -v
```

The storage suite contains **16 tests**, previously reported passing during development. They cover initialization and idempotency, ticket creation and UUID uniqueness, retrieval and unknown IDs, newest-first listing, initial Open status, UTC timestamps, persistence across connections, allowed status updates, invalid statuses, preservation of unrelated data, and safe database-error handling.

Each test patches `ticket_store.DB_PATH` to a database inside `TemporaryDirectory`. Tests never open or modify the application ticket database, and they require no OpenAI API calls. Run the command above to confirm results in the final checkout; tests were not rerun while preparing this README.

### Manual chatbot and UI verification

Development feedback reported the chatbot, ticket creation, My Tickets, and dashboard functioning. This is manual verification, not automated end-to-end coverage. Use the following final submission checklist to reconfirm the current checkout with fictional inputs:

- Ask a knowledge-base question, such as enabling 2FA; inspect the answer's source IDs and expandable evidence.
- Ask an unsupported question; confirm escalation rather than an invented answer, and confirm the UI makes no automatic-agent-notification promise.
- Create a ticket explicitly, verify its ID and initial status, and rerun/navigate to confirm the same escalation retains its confirmation.
- Browse Help Topics mid-chat, select another question, and verify earlier messages remain.
- Switch among all pages; return to chat and verify history remains. Use New Chat and verify the initial view returns.
- Look up a saved ticket, try an unknown ID, and use Clear Search; verify clearing does not delete the ticket.
- Filter dashboard tickets, inspect full details, update a status explicitly, and verify refreshed counts and My Tickets status.

No evaluation score, accuracy claim, latency benchmark, or performance improvement is asserted here.

## 9. Original Contributions

| Inherited academy reference functionality | Independently implemented ResolveAI enhancements |
| --- | --- |
| Fictional CloudSync Pro JSON data and document loading | ResolveAI branding, responsive UI styling, and typography |
| HuggingFace embeddings, FAISS/BM25 hybrid retrieval, and reciprocal rank fusion | Six-category Quick Help, Browse Help Topics, and improved inline conversation layout |
| Cosine scoring, confidence thresholds, LangGraph routing, and GPT-4o-mini answer/judge prompts | Three-page navigation that preserves chat state and clearer source/confidence presentation |
| Basic Streamlit chat, session display history, sample questions, source traces, and reset behavior | Explicit per-escalation ticket forms, local SQLite persistence, original-question association, and session duplicate protection |
| No corresponding ticket-management feature in the starter | My Tickets lookup and Clear Search; Support Dashboard metrics, filtering, full details, and explicit status updates |
| Textual escalation response | Accurate UI explanation that no agent is automatically notified; isolated storage tests |

The inherited retrieval and routing design is not presented as a newly invented ResolveAI algorithm. CloudSync Pro names within the reference data remain unchanged. Conversation memory was explored and reverted; it is not part of this submission.

## 10. Known Limitations and Future Improvements

- **No authentication or authorization:** Ticket-ID lookup is not identity verification. The dashboard exposes local tickets and status controls to anyone with application access. Authentication, customer ownership checks, and role-based permissions are prerequisites for public deployment.
- **No automatic agent notifications:** Saving a ticket only writes to local SQLite. Agent assignment, notifications, and service-level workflows are future work. The inherited backend's raw escalation text still contains a handoff promise; the ResolveAI UI replaces it with accurate local-ticket guidance.
- **Conversation memory is deferred:** `Bot.ask(query)` receives only the current question. Visible session history does not make pronoun-based follow-ups context-aware. Users should ask self-contained questions.
- **Session-only chat state:** New Chat clears displayed exchanges, not persisted tickets. Session loss removes chat history and ticket confirmations. Duplicate prevention is per escalation within the session, not a durable idempotency guarantee across crashes or reconnects.
- **Demonstration knowledge base:** Fictional policies, URLs, and historical resolutions are not live product guidance. Newly created SQLite tickets are not automatically indexed into RAG.
- **Grounding has limits:** Similarity thresholds, a context judge, and citation prompts do not prove factual correctness. More representative evaluation and careful threshold review would be needed for a real support domain.
- **Optional persistence and performance work:** FAISS is rebuilt on process restart. Persisting indexes, caching candidate embeddings, and measuring startup/query costs are possible improvements, not implemented optimizations.
- **Local storage and operations:** SQLite is suitable for this small local demonstration. Backups, retention, audit history, concurrent update handling, and larger-scale storage need further design. The schema stores current status, not a status-change timeline.
- **UI compatibility:** Some styling targets Streamlit's internal element selectors and may need adjustment as versions change. Dependency ranges and model availability also require final environment verification.

## 11. Attribution

This project builds on **The Gen Academy** starter repository:

[The Gen Academy — Customer Support Knowledge Base with Hybrid Search](https://github.com/The-Gen-Academy/2D-Customer-Support-KB-with-Hybrid-Search)

The original CloudSync Pro knowledge base, hybrid retrieval pipeline, LangGraph confidence routing, and prompts are credited to that reference implementation. ResolveAI adds the interface and ticket-management enhancements identified above. The Gen Academy has confirmed permission to publish this implementation. Cohort materials are excluded from this submission and must not be publicly redistributed. This permission does not imply unrestricted ownership of inherited material, and this README does not establish a new license.
