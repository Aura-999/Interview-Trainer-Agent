# Interview Trainer Agent

AI-powered interview preparation using IBM Granite (watsonx.ai) + RAG (ChromaDB).

Generates personalised interview questions, model answers, and scored feedback from a candidate's resume and target role.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | IBM Granite 13B Chat v2 via watsonx.ai |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector DB | ChromaDB (persistent local) |
| Object Storage | IBM Cloud Object Storage (Lite) |
| Backend | FastAPI + Python 3.11 |
| Frontend | Streamlit |
| Resume Parsing | pdfplumber + regex |
| Response Cache | diskcache (SQLite-backed) |

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd interview-trainer
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Fill in WATSONX_API_KEY, WATSONX_PROJECT_ID — get them from cloud.ibm.com
```

### 3. Build the Knowledge Base Index (run once)

```bash
python backend/kb_builder.py
```

Output should show ~42+ documents indexed into `./chroma_db/`.

### 4. Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Start the Frontend (new terminal)

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Token Minimisation

All strategies are active by default:

| Strategy | Where | Effect |
|---|---|---|
| Per-task `max_new_tokens` caps | `watsonx_client.py` | Questions≤400, Answers≤300, Feedback≤280 |
| Disk response cache | `rag_engine.py` + `watsonx_client.py` | Identical inputs skip Granite entirely (0 tokens) |
| Compressed prompt templates | `prompts/*.txt` | No filler text; ~40% fewer input tokens vs verbose prompts |
| Context hard-cap | `rag_engine.py:format_chunks_for_prompt()` | Retrieved chunks capped at 1200 chars total |
| Skills cap | `prompt_builder.py` | Max 6 skills injected into prompt |

Estimated Lite tier usage per full session (8 questions + 4 answers + 2 feedback):
~6,000–8,000 tokens → ~50 sessions per month on the free 50k limit.

---

## Project Structure

```
interview-trainer/
├── backend/
│   ├── main.py              # FastAPI app + endpoints
│   ├── rag_engine.py        # ChromaDB retrieval + response cache
│   ├── watsonx_client.py    # Granite wrapper with token controls
│   ├── resume_parser.py     # PDF/DOCX → CandidateProfile
│   ├── prompt_builder.py    # Template filler
│   └── kb_builder.py        # Offline index builder (run once)
├── frontend/
│   └── app.py               # Streamlit UI
├── prompts/
│   ├── question_gen.txt     # Compact question generation prompt
│   ├── model_answer.txt     # Model answer prompt
│   └── feedback.txt         # Scoring + feedback prompt
├── knowledge_base/
│   ├── technical/
│   │   ├── software_engineer.jsonl
│   │   └── data_scientist.jsonl
│   ├── behavioral/
│   │   └── star_scenarios.json
│   └── hr_guidelines/
│       └── evaluation_rubric.txt
├── chroma_db/               # Auto-created by kb_builder.py
├── cache/                   # Auto-created — diskcache response store
├── .env.example
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/parse-resume` | Upload PDF/DOCX → parsed profile JSON |
| `POST` | `/api/generate-questions` | Profile JSON → list of questions |
| `POST` | `/api/generate-answer` | Question → model answer |
| `POST` | `/api/get-feedback` | Question + user answer → scores + feedback |

---

## Adding More Knowledge Base Content

Technical questions follow JSONL format — one JSON object per line:
```json
{"question": "...", "answer": "...", "difficulty": "easy|medium|hard", "skill_tag": "..."}
```

Behavioral scenarios follow JSON array format in `star_scenarios.json`:
```json
{"competency": "...", "question": "...", "situation": "...", "task": "...", "action": "...", "result": "...", "level": "junior|mid|senior"}
```

After adding data, re-run `python backend/kb_builder.py` — it skips already-indexed items.

---

## IBM Cloud Lite Setup

1. Create account: [cloud.ibm.com](https://cloud.ibm.com) (free, no credit card)
2. Create a **watsonx.ai** project → copy Project ID
3. Generate an **API key**: Manage → Access → API keys
4. Create an **IBM Cloud Object Storage** Lite instance → create bucket `interview-kb`
5. Paste all values into `.env`
