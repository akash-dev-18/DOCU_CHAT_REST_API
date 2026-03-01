# Doc Chat API

A FastAPI-based Retrieval-Augmented Generation (RAG) service for chatting with PDF documents.

It lets you:
- Upload a PDF and index it into Qdrant.
- Ask questions grounded in that document.
- Continue conversations with session-based chat history.
- Stream answers token-by-token over Server-Sent Events (SSE).

## Features

- PDF ingestion with chunking (`PyPDFLoader` + `RecursiveCharacterTextSplitter`)
- Vector storage in Qdrant (`langchain-qdrant`)
- Embeddings via OpenRouter-compatible OpenAI endpoint (`text-embedding-3-small`)
- Chat completion via OpenRouter (`upstage/solar-pro-3:free`)
- API key protection on all core endpoints
- Basic request rate limiting (`slowapi`)
- Swagger UI at `/docs`

## Tech Stack

- Python 3.12
- FastAPI + Uvicorn
- LangChain
- Qdrant
- OpenRouter-compatible OpenAI APIs

## Project Structure

```text
.
├── app/
│   ├── api.py            # FastAPI routes and request validation
│   ├── ingest.py         # PDF loading, chunking, embedding, indexing
│   ├── rag_pipeline.py   # Retrieval + prompt + chat/stream logic
│   ├── auth.py           # API key verification
│   └── rate_limiter.py   # Rate limiting
├── uploaded_pdfs/        # Temporary uploaded files
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## How It Works

1. `POST /ingest` uploads and validates a PDF.
2. PDF is split into chunks and embedded.
3. Chunks are written to Qdrant collection `pdf_chat`.
4. `POST /chat` retrieves top-k chunks (`k=4`) for each question.
5. LLM answers using only retrieved context and prior session messages.
6. `POST /chat/stream` returns streaming SSE tokens.

## Prerequisites

- Python 3.12+
- Running Qdrant instance (local or cloud)
- OpenRouter API key (used for both embeddings and chat model)

## Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_key
API_KEY=your_service_api_key
QDRANT_URL=https://your-qdrant-host
QDRANT_API_KEY=your_qdrant_api_key
```

## Run Locally

### Option A: `uv` (recommended)

```bash
uv sync
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- API docs: `http://localhost:8000/docs`

## Run with Docker

```bash
docker build -t doc-chat-api .
docker run --env-file .env -p 8000:8000 doc-chat-api
```

## Test the Deployed API

The API is deployed and ready to test at:

**[https://docu-chat-rest-api.onrender.com/docs](https://docu-chat-rest-api.onrender.com/docs)**

### Quick Start

Use the API key: `DOCU-CHAT-API-KEY` for authorization.

**Via Swagger UI:**
1. Open [https://docu-chat-rest-api.onrender.com/docs](https://docu-chat-rest-api.onrender.com/docs)
2. Click the lock icon on any endpoint
3. Enter API key: `DOCU-CHAT-API-KEY`
4. Try out the endpoints

**Via cURL:**

```bash
# Ingest a PDF
curl -X POST "https://docu-chat-rest-api.onrender.com/ingest" \
  -H "PDF-CHAT-API-KEY: DOCU-CHAT-API-KEY" \
  -F "file=@/path/to/document.pdf"

# Ask a question
curl -X POST "https://docu-chat-rest-api.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -H "PDF-CHAT-API-KEY: DOCU-CHAT-API-KEY" \
  -d '{"session_id":"test-user","question":"What is the main topic?"}'

# Stream a response
curl -N -X POST "https://docu-chat-rest-api.onrender.com/chat/stream" \
  -H "Content-Type: application/json" \
  -H "PDF-CHAT-API-KEY: DOCU-CHAT-API-KEY" \
  -d '{"session_id":"test-user","question":"Summarize the document"}'
```

## Authentication

Protected endpoints require header:

```http
PDF-CHAT-API-KEY: <your API_KEY value>
```

## API Endpoints

### 1) Health Redirect

- `GET /`
- Redirects to `/docs`

### 2) Ingest PDF

- `POST /ingest`
- `multipart/form-data` with field `file`
- Max file size: **50MB**
- Allowed extension: **.pdf**
- Rate limit: **5/minute**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "PDF-CHAT-API-KEY: your_service_api_key" \
  -F "file=@/absolute/path/to/document.pdf"
```

Example response:

```json
{
  "chunks_length": 87,
  "filename": "document.pdf",
  "saved_as": "a1b2c3d4e5f6.pdf",
  "status": "success"
}
```

### 3) Chat (non-streaming)

- `POST /chat`
- Body:

```json
{
  "session_id": "user-123",
  "question": "Summarize section 2"
}
```

- Rate limit: **30/minute**

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "PDF-CHAT-API-KEY: your_service_api_key" \
  -d '{"session_id":"user-123","question":"Summarize section 2"}'
```

### 4) Chat (streaming SSE)

- `POST /chat/stream`
- Content type: `text/event-stream`
- Rate limit: **20/minute**

```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "PDF-CHAT-API-KEY: your_service_api_key" \
  -d '{"session_id":"user-123","question":"Give me key takeaways"}'
```

Stream ends with:

```text
data: [DONE]
```

### 5) Clear Session

- `DELETE /session/{session_id}`
- Rate limit: **20/minute**

```bash
curl -X DELETE "http://localhost:8000/session/user-123" \
  -H "PDF-CHAT-API-KEY: your_service_api_key"
```

## Current Behavior and Important Notes

- Ingestion currently recreates the `pdf_chat` collection each time. Uploading a new PDF replaces previous indexed content.
- Chat history is kept in memory (`session_histories` dict), so restarting the server clears sessions.
- The system prompt instructs the model to answer only from retrieved context and say it does not know when context is insufficient.
- Streaming endpoint emits SSE `data:` chunks and then `[DONE]`.

## Error Handling

Common API errors:
- `400` invalid file type, empty file, empty question, or empty session ID
- `401` missing API key
- `403` invalid API key
- `429` rate limit exceeded
- `500` ingestion or chat internal failure

## Rate Limiting Note

Rate limiter keying currently reads header `x-api-key`, while auth validates `PDF-CHAT-API-KEY`.  
To get per-key rate limiting behavior as intended, you can send both headers:

```http
PDF-CHAT-API-KEY: <key>
x-api-key: <key>
```

## Example End-to-End Flow

1. Start server.
2. Upload PDF via `/ingest`.
3. Ask questions via `/chat` or `/chat/stream` using the same `session_id`.
4. Clear conversation with `DELETE /session/{session_id}` when needed.

## License

Add your preferred license (MIT, Apache-2.0, etc.) in this section.
