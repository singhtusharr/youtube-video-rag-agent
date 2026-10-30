# Agentic YouTube Video Assistant (RAG)

Ask questions about any YouTube video using only its transcript — with a
tool-calling agent that can also summarize the full video or fall back to a
live web search when the transcript doesn't have the answer.

## Architecture

```
video URL/id
    │
    ▼
[ingestion.py] fetch transcript (YouTubeTranscriptApi) → split into chunks
    │                                                     (RecursiveCharacterTextSplitter)
    ▼
[vectorstore.py] embed chunks (HuggingFace all-MiniLM-L6-v2) → FAISS index
    │
    ▼
[retrieval.py] similarity OR MMR retriever (top-k relevant chunks)
    │
    ▼
[chains.py] LCEL chain: RunnableParallel({context, question}) → prompt → LLM → parser
    │              (plain string OR Pydantic-validated structured output)
    ▼
[tools.py + agent.py] AgentExecutor picks: retrieve_from_video | summarize_video | web_search
    │
    ▼
[main.py] FastAPI /ask, /summarize   ──▶   [streamlit_app.py] chat UI
```

## Why each piece is there

- **RecursiveCharacterTextSplitter**: splits on paragraph/sentence/word
  boundaries before falling back to a hard cut, so chunks stay coherent.
  200-char overlap avoids losing context at chunk boundaries.
- **FAISS**: in-memory approximate-nearest-neighbor index over the chunk
  embeddings — fast similarity search without standing up an external vector DB.
- **Similarity vs MMR retrieval**: similarity returns the k closest chunks,
  which can be near-duplicates of each other. MMR (Maximal Marginal Relevance)
  pulls a larger candidate pool and greedily balances relevance against
  diversity, so you get broader coverage of the video instead of 4 chunks
  all saying the same thing.
- **LCEL (`RunnableParallel` / `RunnablePassthrough`)**: declarative pipeline
  composition — the context-retrieval branch and the question-passthrough
  branch run in parallel and merge into the prompt's input dict.
- **Structured output (`VideoAnswer` Pydantic schema)**: forces the LLM to
  return `{answer, key_topics, is_grounded, confidence}` instead of free text,
  via OpenAI's native structured-output support. Lets the app programmatically
  check `is_grounded` instead of trusting prose — this is the hallucination
  mitigation.
- **Agent (`AgentExecutor` + `create_tool_calling_agent`)**: the LLM decides
  which tool to call per query instead of a hardcoded retrieve-then-answer
  path. Three tools: grounded retrieval, full-video summarization, and
  DuckDuckGo web search for anything outside the transcript.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your key to `.env`:
```
OPENAI_API_KEY=sk-...
```

## Run

```bash
# terminal 1
uvicorn main:app --reload

# terminal 2
streamlit run streamlit_app.py
```

Or hit the API directly:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"video": "https://www.youtube.com/watch?v=VIDEO_ID", "question": "What is this video about?"}'
```

## Study notebook

`rag_using_langchain.ipynb` walks through every step above cell-by-cell with
explanations — use it to review the concepts, not as the production code
(that lives in `src/`, `main.py`, `streamlit_app.py`).
