from dotenv import load_dotenv
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from ingest import ingest_pdf
from rag_pipeline import chat, clear_session
from auth import verify_api_key


load_dotenv()

app = FastAPI()

UPLOAD_DIR = "uploaded_pdfs"
MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class IngestResponse(BaseModel):
    chunks_length: int
    filename: str
    saved_as: str
    status: str


# ── Helpers ──────────────────────────────────────────────
def validate_file(file: UploadFile):
    # Check filename exists
    if not file.filename or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type '{ext}'. Only PDF allowed."
        )


def save_upload(file: UploadFile) -> tuple[str, str]:
    if not file.filename:
        raise HTTPException(400, "Filename missing")

    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    contents = file.file.read()

    if not contents:
        raise HTTPException(400, "Uploaded file is empty")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"large file {MAX_FILE_SIZE // (1024 * 1024)}MB allowed"
        )

    with open(save_path, "wb") as f:
        f.write(contents)

    return unique_name, save_path


def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ── Routes ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "working fine"}


@app.post(
    "/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)]
)
async def ingest(file: UploadFile = File(...)):
    validate_file(file)

    unique_name, save_path = save_upload(file)

    try:
        result = ingest_pdf(save_path)
    except Exception as e:
        cleanup_file(save_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        cleanup_file(save_path)

    return IngestResponse(
        filename=file.filename,
        saved_as=unique_name,
        chunks_length=result["chunks_created"],
        status=result["status"],
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    try:
        answer = chat(
            session_id=request.session_id,
            question=request.question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return ChatResponse(answer=answer, session_id=request.session_id)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str, dependencies=[Depends(verify_api_key)]):
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
