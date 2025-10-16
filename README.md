# Smart Support Platform

This repository delivers a full-stack question-answering assistant for contact-center operators. It combines:
- a FastAPI backend that orchestrates retrieval and LLM calls,
- a React single-page application for the operator experience,
- data preparation utilities that build and maintain a FAISS-powered knowledge base.

---

## Architecture Overview
- **Backend (`backend/`)** – FastAPI app exposing `POST /dialog`. It loads the FAISS index, classifies questions with `qwen2.5-72b-h100`, performs hybrid retrieval, and returns a recommended answer plus metadata.
- **Data layer (`data/`)** – Prepares embeddings via the Scibox API (`bge-m3`), writes the FAISS index and metadata JSON, and implements classifier/verification helpers shared with the backend.
- **Frontend (`frontend/`)** – Vite + React interface for operators. It sends questions to the backend, displays answers, handles loading/errors, and stores the session transcript client-side.
- **Artifacts (`data/data/`)** – Generated FAISS index (`knowledge_base.index`) and metadata (`knowledge_base_metadata.json`) consumed at runtime.

External dependencies: Docker (optional), Python 3.10+, Node.js 18+, internet access to `https://llm.t1v.scibox.tech/v1`, and a valid Scibox API key.

---

## Repository Layout
```
.
├── backend/               # FastAPI service
│   ├── main.py            # Application entry point
│   ├── models/            # Pydantic schemas
│   ├── point/dialog_user.py   # /dialog endpoint
│   └── support/support.py     # Retrieval + response pipeline
├── data/                  # Knowledge-base and LLM utilities
│   ├── embeddings.py      # Excel ingestion, embeddings, FAISS builder
│   ├── classifier.py      # qwen2.5-72b-h100 classification helpers
│   ├── search.py          # Vector search + LLM verification
│   ├── config.py          # API endpoints, model names, batch sizes
│   ├── data/              # Generated FAISS index + metadata JSON
│   └── smart_support_...xlsx  # Source FAQ spreadsheet
├── frontend/              # React client (Vite)
│   ├── src/App.jsx        # Chat UI logic
│   ├── src/styles.css     # Styling
│   └── Dockerfile         # Static build container
├── docker-compose.yml     # Full stack launcher (backend + frontend)
└── README.md              # Project documentation (this file)
```

---

## Configuration
Environment variables you are likely to override:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCIBOX_API_KEY` | `sk-dgKROD7rG4yPAo7bTOtatA` | API key for Scibox embeddings and chat completions. Replace for production. |
| `VITE_API_URL` | `http://localhost:8000/dialog` | Backend URL used by the React client; set via `.env.local` or shell. |

Other defaults (see `data/config.py`):
- Embedding model: `bge-m3`
- Classifier/verification model: `qwen2.5-72b-h100`
- Batch size: `128`
- Data directory: `data/data`

---

## Running with Docker Compose
```bash
# build and start services
docker compose up --build

# optional: run detached
docker compose up --build -d

# shutdown
docker compose down
```

Endpoints:
- Backend API – `http://localhost:8000`
- Swagger UI – `http://localhost:8000/docs`
- Frontend UI – `http://localhost:5173`

Ensure `data/data/knowledge_base.index` and `data/data/knowledge_base_metadata.json` exist before launching. Regenerate them if the FAQ spreadsheet changes (see *Knowledge Base Preparation*).

---

## Local Development
### Backend
```bash
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# or: source venv/bin/activate # Unix-like shells
pip install -r backend/requirements.txt

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI lives at `http://localhost:8000/docs`. Example request:
```http
POST /dialog
Content-Type: application/json

{ "text": "How do I reset the remote banking password?" }
```

### Frontend
```bash
cd frontend
npm install        # once
npm run dev        # Vite dev server on http://localhost:5173
```

To build a production bundle:
```bash
npm run build      # outputs to frontend/dist
npm run preview    # static preview server
```

Set `VITE_API_URL` if the backend is not running on `http://localhost:8000/dialog`.

---

## Knowledge Base Preparation
The assistant relies on a FAISS index built from the FAQ spreadsheet. Rebuild it whenever the source data changes.

1. Place the updated spreadsheet (default: `smart_support_vtb_belarus_faq_final.xlsx`) in `data/`.
2. Activate the Python environment with dependencies from `backend/requirements.txt`.
3. Run the preparation script:
   ```bash
   python - <<'PY'
from data.embeddings import prepare_knowledge_base
prepare_knowledge_base(show_progress=True)
PY
   ```
4. The process calls the Scibox embedding endpoint (`bge-m3`), writes `knowledge_base.index` and `knowledge_base_metadata.json` to `data/data/`, and logs progress.
5. Restart the backend to load the refreshed assets.

If the files are missing at runtime, the backend returns HTTP 500 with a diagnostic message pointing to the missing path.

---

## Backend Processing Flow
1. **Validation** – `backend/models/models.py` ensures incoming payload contains a non-empty `text` field.
2. **Resource loading** – `backend/point/dialog_user.py` lazily loads metadata and FAISS index on first request.
3. **Query sanitization** – `validate_and_process_query` (in `backend/support/support.py`) trims and validates the question, rejecting empty or overly long inputs.
4. **Classification** – `data.classifier.classify_query` calls `qwen2.5-72b-h100` to predict main/sub categories and a confidence score.
5. **Vector retrieval** – `data.search.search_and_rank` embeds the query (`bge-m3`), searches FAISS, boosts by category/priority/audience, and verifies answers with `verify_candidate_with_llm`.
6. **Fallback fuzzy matching** – If the vector path fails, the backend falls back to fuzzy comparisons against example questions / combined texts.
7. **Response formatting** – `_format_response` returns a JSON payload containing:
   - `recommended_answer`
   - `category_path` (if available)
   - `similarity_score`, `llm_confidence`
   - `classification` metadata

Errors are propagated as `HTTPException`:
- 400 – validation problems (e.g., empty query).
- 500 – missing resources or runtime failures (FAISS load, external API).

---

## Frontend Overview
- **Stack**: React 18, Vite 5, vanilla CSS.
- **Key component**: `App.jsx` manages stages (`intro` → `chat`), message log, optimistic user messages, loading spinner, and error banner.
- **API calls**: `fetch(API_URL, { method: "POST", body: JSON.stringify({ text }) })`. `API_URL` defaults to `http://localhost:8000/dialog` but respects `VITE_API_URL`.
- **Customization**: Update labels/text in `App.jsx`, styling in `styles.css`, and head metadata in `index.html`.

---

## Deployment Notes
- Inject secrets (e.g., `SCIBOX_API_KEY`) via environment variables or secret managers; never commit production keys.
- The backend Dockerfile mounts source directories read-only. Adjust Docker compose if write access is required.
- For production, consider running FastAPI under `gunicorn` + `uvicorn.workers.UvicornWorker`, add HTTPS termination, and integrate logging/metrics sinks.
- CORS is currently open to `http://localhost:5173` and `http://127.0.0.1:5173`. Update `backend/main.py` if hosting the frontend elsewhere.

---

## Troubleshooting
- **Missing metadata/index** – Re-run *Knowledge Base Preparation* and confirm the backend container sees `data/data/*.json` and `.index`.
- **FAISS errors** – Ensure `faiss-cpu` is installed for the active Python interpreter; reinstall if platform mismatch occurs.
- **Scibox API failures** – Inspect logs for HTTP status codes; verify connectivity and the API key.
- **Frontend cannot reach backend** – Check browser console, CORS configuration, and `VITE_API_URL`.
- **Encoding issues in source spreadsheet** – Confirm the Excel file uses UTF-8 compatible encoding; pandas expects proper headers for `embeddings.py`.

---

## Suggested Improvements
1. Add automated tests (unit + integration) for `support.py` and `data/` modules.
2. Introduce pre-commit hooks (Ruff, Black, ESLint) to enforce coding standards.
3. Enhance observability (structured logging, request tracing, metrics).
4. Localize the frontend copy and remove placeholder text before production launch.

With this documentation, new contributors should be able to set up the environment, understand the data flow, and extend the Smart Support Platform effectively.
