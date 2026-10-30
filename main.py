"""FastAPI backend: exposes the agent as a REST API so any frontend (Streamlit,
curl, Postman) can talk to it. This is what turns the notebook pipeline into
an actual service, matching the "FastAPI" line on the resume.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent import build_agent_executor, run_agent
from src.ingestion import TranscriptUnavailableError
from src.schemas import VideoAnswer
from src.session import VideoSession, get_or_create_session

app = FastAPI(title="YouTube Video Assistant (RAG + Agent)")


class AskRequest(BaseModel):
    video: str  # YouTube URL or bare video id
    question: str
    use_mmr: bool = False  # True = MMR retrieval (more diverse chunks) instead of plain similarity


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _session_or_http_error(video: str) -> VideoSession:
    """Build/fetch the cached session, translating known failures into HTTP errors."""
    try:
        return get_or_create_session(video)
    except TranscriptUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Agentic endpoint: the LLM decides whether to retrieve, summarize, or
    search the web to answer `question`."""
    session = _session_or_http_error(request.video)
    agent = build_agent_executor(session, use_mmr=request.use_mmr)
    answer = run_agent(agent, request.question)
    return AskResponse(answer=answer)


@app.post("/ask_structured", response_model=VideoAnswer)
def ask_structured(request: AskRequest):
    """Plain RAG (no agent) whose reply is validated against the VideoAnswer
    schema: answer, key_topics, is_grounded and confidence."""
    session = _session_or_http_error(request.video)
    return session.ask_structured(request.question, mmr=request.use_mmr)


@app.post("/summarize")
def summarize(video: str):
    session = _session_or_http_error(video)
    return {"summary": session.summarize()}
