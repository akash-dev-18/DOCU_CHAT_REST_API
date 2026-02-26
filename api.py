from dotenv import load_dotenv
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse

from pydantic import BaseModel
from ingest import ingest_pdf
from rag_pipeline import chat, clear_session, chat_stream
from auth import verify_api_key
from rate_limiter import limiter
from slowapi.errors import RateLimitExceeded

load_dotenv()

app = FastAPI()
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429, content={"detail": "Rate limit exceeded. Try again later."}
    )


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


def validate_file(file: UploadFile):
    if not file.filename or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

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
            400, f"File too large. Max {MAX_FILE_SIZE // (1024 * 1024)}MB allowed"
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


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)]
)
@limiter.limit("5/minute")
async def ingest(request: Request, file: UploadFile = File(...)):
    validate_file(file)

    unique_name, save_path = save_upload(file)

    try:
        result = await ingest_pdf(save_path)
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
@limiter.limit("30/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not body.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    try:

        answer = await chat(
            session_id=body.session_id,
            question=body.question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return ChatResponse(answer=answer, session_id=body.session_id)


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def chat_stream_endpoint(request: Request, body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not body.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    async def token_generator():
        try:
            async for token in chat_stream(
                session_id=body.session_id,
                question=body.question,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def delete_session(request: Request, session_id: str):
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
