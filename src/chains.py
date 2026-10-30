"""Step 4 & 5 (Augmentation + Generation): assemble LCEL chains.

LCEL ("LangChain Expression Language") lets you compose Runnables with `|`
into a single pipeline that supports streaming/batching/async for free.

- RunnableParallel: runs multiple branches concurrently and merges their
  outputs into one dict. Here it runs the retriever (-> formatted context)
  and passes the question through, in parallel, producing
  {"context": ..., "question": ...} in one step.
- RunnablePassthrough: an identity function as a Runnable — used so the raw
  question flows into the parallel dict unchanged alongside the context branch.
"""
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY
from src.retrieval import format_docs
from src.schemas import VideoAnswer

RAG_PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions about a YouTube video.
Answer ONLY from the provided transcript context. If the context is insufficient,
say so explicitly rather than guessing.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"],
)

SUMMARY_PROMPT = PromptTemplate(
    template="""Summarize the following YouTube video transcript in a concise,
well-organized paragraph covering the main points discussed.

Transcript:
{context}
""",
    input_variables=["context"],
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)


def build_rag_chain(retriever: VectorStoreRetriever):
    """Plain-text RAG chain: retriever -> prompt -> llm -> string."""
    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    return parallel | RAG_PROMPT | get_llm() | StrOutputParser()


def build_structured_rag_chain(retriever: VectorStoreRetriever):
    """Same pipeline, but the LLM output is forced into the VideoAnswer schema
    instead of a free-text string."""
    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    structured_llm = get_llm().with_structured_output(VideoAnswer)
    return parallel | RAG_PROMPT | structured_llm


def build_summary_chain():
    """Full-video summarization: stuffs the whole transcript in (no retrieval —
    we want the whole video, not just the top-k relevant chunks)."""
    return SUMMARY_PROMPT | get_llm() | StrOutputParser()
