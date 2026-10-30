"""Step 3: turn a vector store into a retriever.

Two retrieval strategies, both real (this fixes the old notebook's bug, where
`seach_kwargs` was misspelled and passed a set `{"k", 4}` instead of a dict
`{"k": 4}` — LangChain silently dropped the bad kwarg and fell back to FAISS's
own default k=4, so it looked like it worked but never actually applied your
settings):

- similarity: return the k chunks whose embeddings are closest to the query
  embedding. Straightforward, but if the top-k chunks are all near-duplicates
  of each other, you lose coverage of the rest of the video.
- MMR (Maximal Marginal Relevance): fetches a larger candidate pool (`fetch_k`)
  then greedily picks k of them balancing relevance to the query against
  diversity from chunks already picked. `lambda_mult` tunes that balance
  (1.0 = pure relevance/same as similarity, 0.0 = pure diversity).
"""
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def get_similarity_retriever(vector_store: FAISS, k: int = 4) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_mmr_retriever(
    vector_store: FAISS, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
) -> VectorStoreRetriever:
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def format_docs(docs) -> str:
    """Join retrieved chunks into one context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)
