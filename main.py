"""FastAPI backend: exposes the agent as a REST API so any frontend (Streamlit,
curl, Postman) can talk to it. This is what turns the notebook pipeline into
an actual service, matching the "FastAPI" line on the resume.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent import build_agent_executor, run_agent
from src.ingestion import TranscriptUnavailableError
from src.session import get_or_create_session

app = FastAPI(title="YouTube Video Assistant (RAG + Agent)")


class AskRequest(BaseModel):
    video: str  # YouTube URL or bare video id
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Agentic endpoint: the LLM decides whether to retrieve, summarize, or
    search the web to answer `question`."""
    try:
        session = get_or_create_session(request.video)
    except TranscriptUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent = build_agent_executor(session)
    answer = run_agent(agent, request.question)
    return AskResponse(answer=answer)


@app.post("/summarize")
def summarize(video: str):
    try:
        session = get_or_create_session(video)
    except TranscriptUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"summary": session.summarize()}
