"""Step 1: pull a transcript off YouTube and split it into retrievable chunks.

Why RecursiveCharacterTextSplitter: it tries to split on paragraph/sentence/word
boundaries (in that order) before falling back to a hard character cut, so chunks
stay semantically coherent instead of getting sliced mid-sentence.
"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


class TranscriptUnavailableError(Exception):
    """Raised when a video has no transcript we can fetch (disabled/not found)."""


def extract_video_id(video_url_or_id: str) -> str:
    """Accept a raw video id OR a full YouTube URL and return just the id.

    Handles youtu.be/<id>, youtube.com/watch?v=<id>, and already-bare ids —
    this is what makes the app "seamless": the user pastes whatever they have.
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # watch?v=... or /embed/...
        r"^([0-9A-Za-z_-]{11})$",  # bare 11-char id
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract a video id from: {video_url_or_id!r}")


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """Fetch the raw transcript text for a video id, concatenated into one string."""
    languages = languages or ["en"]
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=languages)
        return " ".join(chunk["text"] for chunk in fetched.to_raw_data())
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        raise TranscriptUnavailableError(
            f"No transcript available for video '{video_id}': {exc}"
        ) from exc


def split_transcript(transcript: str) -> list[Document]:
    """Chunk the transcript into overlapping Documents ready for embedding.

    Overlap (200 chars) means context isn't lost at chunk boundaries — if the
    answer straddles two chunks, both still contain enough surrounding text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([transcript])


def load_video_chunks(video_url_or_id: str, languages: list[str] | None = None) -> list[Document]:
    """End-to-end: URL/id -> transcript -> chunks. This is the function the
    rest of the app (API, Streamlit, agent tools) calls."""
    video_id = extract_video_id(video_url_or_id)
    transcript = fetch_transcript(video_id, languages=languages)
    return split_transcript(transcript)
