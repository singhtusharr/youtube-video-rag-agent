"""The tool-calling agent: given a user query, the LLM itself decides which
tool(s) to call and in what order, rather than us hardcoding "always retrieve
then answer." This is the difference between a fixed RAG chain and an agent:
a chain has one path; an agent dynamically routes.

`create_agent` (LangChain 1.x, built on LangGraph under the hood) replaces the
older `create_tool_calling_agent` + `AgentExecutor` pair from LangChain 0.x —
same idea (LLM emits a native tool call -> tool runs -> result fed back to the
LLM -> repeat until a final answer), but as a compiled graph rather than a
manually wired executor. It's invoked with `{"messages": [...]}` and returns
the full message list, so the answer is the last message's `.content`.
"""
from langchain.agents import create_agent

from src.chains import get_llm
from src.session import VideoSession
from src.tools import build_video_tools

AGENT_SYSTEM_PROMPT = """You are an assistant that answers questions about a specific \
YouTube video. You have three tools available:

- retrieve_from_video: for questions about specific details discussed in the video
- summarize_video: for "summarize"/"overview"/"tl;dr" style requests
- web_search: only for things the video transcript cannot possibly contain \
(e.g. outside context, current events, background info about a person/topic)

Always prefer retrieve_from_video or summarize_video first. Only fall back to \
web_search if the video tools don't have the answer. Be concise."""


def build_agent_executor(session: VideoSession, use_mmr: bool = False):
    tools = build_video_tools(session, use_mmr=use_mmr)
    return create_agent(get_llm(), tools=tools, system_prompt=AGENT_SYSTEM_PROMPT)


def run_agent(agent, question: str) -> str:
    """Invoke the agent graph with a single user message and return the final
    answer text."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
