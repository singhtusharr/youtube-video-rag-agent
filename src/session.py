"""Bundles everything needed to query one specific video: its chunks, vector
store, and ready-to-use chains. The API/Streamlit layers create one of these
per video (and cache it so re-asking a question doesn't re-embed the transcript).
"""
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.chains import build_rag_chain, build_structured_rag_chain, build_summary_chain
from src.ingestion import load_video_chunks
from src.retrieval import format_docs, get_mmr_retriever, get_similarity_retriever
from src.vectorstore import build_vector_store


@dataclass
class VideoSession:
    video_id: str
    chunks: list[Document]
    vector_store: FAISS

    @classmethod
    def from_video(cls, video_url_or_id: str) -> "VideoSession":
        chunks = load_video_chunks(video_url_or_id)
        vector_store = build_vector_store(chunks)
        from src.ingestion import extract_video_id

        return cls(video_id=extract_video_id(video_url_or_id), chunks=chunks, vector_store=vector_store)

    def full_transcript(self) -> str:
        return format_docs(self.chunks)

    def ask(self, question: str, mmr: bool = False) -> str:
        retriever = get_mmr_retriever(self.vector_store) if mmr else get_similarity_retriever(self.vector_store)
        return build_rag_chain(retriever).invoke(question)

    def ask_structured(self, question: str, mmr: bool = False):
        retriever = get_mmr_retriever(self.vector_store) if mmr else get_similarity_retriever(self.vector_store)
        return build_structured_rag_chain(retriever).invoke(question)

    def summarize(self) -> str:
        return build_summary_chain().invoke({"context": self.full_transcript()})


# In-memory cache so the same video isn't re-embedded on every request.
# A dict is enough for a single-process demo app; swap for Redis if this
# ever needs to run across multiple workers.
_session_cache: dict[str, VideoSession] = {}


def get_or_create_session(video_url_or_id: str) -> VideoSession:
    from src.ingestion import extract_video_id

    video_id = extract_video_id(video_url_or_id)
    if video_id not in _session_cache:
        _session_cache[video_id] = VideoSession.from_video(video_url_or_id)
    return _session_cache[video_id]
