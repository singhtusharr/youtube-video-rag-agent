"""Tools the agent can choose to call. Each one is a plain Python function
wrapped with @tool — LangChain reads the function name, docstring, and type
hints to build the JSON schema it hands the LLM, so the LLM knows what each
tool does and what arguments it takes without us writing that schema by hand.

Three tools, matching the resume bullet exactly:
1. grounded retrieval  -> answer from the video's own transcript
2. full-video summarization -> doesn't need retrieval, wants the whole video
3. external search -> for questions the video transcript can't answer at all
"""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from src.session import VideoSession

web_search = DuckDuckGoSearchRun()
web_search.name = "web_search"
web_search.description = (
    "Search the public web for information NOT found in the video's transcript "
    "(e.g. background on a person/topic mentioned, current events, facts outside "
    "the video's scope). Input should be a search query string."
)


def build_video_tools(session: VideoSession, use_mmr: bool = False) -> list:
    """Build the retrieval + summarization tools bound to one specific video's
    session (closures capture `session` so the agent doesn't need to re-pass
    video_id on every tool call). `use_mmr` switches the retrieval tool from
    plain similarity search to MMR (more diverse chunks)."""

    @tool
    def retrieve_from_video(question: str) -> str:
        """Answer a question using ONLY the video's transcript content via
        vector search. Use this for questions about what was actually
        said/discussed in the video."""
        return session.ask(question, mmr=use_mmr)

    @tool
    def summarize_video(_: str = "") -> str:
        """Produce a summary of the entire video. Use this when the user asks
        for an overview/summary/tl;dr rather than a specific detail."""
        return session.summarize()

    return [retrieve_from_video, summarize_video, web_search]
