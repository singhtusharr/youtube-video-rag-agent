"""Structured output schema for the RAG answer chain.

Plain `StrOutputParser` just gives you back a string blob — the LLM can hedge,
ramble, or bury a "but I'm not actually sure" halfway through, and nothing in
your code can act on that. Binding the LLM to a Pydantic schema with
`.with_structured_output()` forces it to return fields that match this shape
(OpenAI does this via function-calling under the hood), so you can
programmatically check `is_grounded` before trusting an answer instead of
just trusting an unstructured string of prose. That's the "mitigate
hallucinations" part of the resume bullet.
"""
from typing import Literal

from pydantic import BaseModel, Field


class VideoAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question, grounded in the provided transcript context.")
    key_topics: list[str] = Field(description="2-5 short keywords/topics from the context relevant to the answer.")
    is_grounded: bool = Field(description="True only if the context actually contained enough information to answer confidently.")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence that `answer` is correct and fully supported by the context.")
