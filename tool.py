# tool.py
from langchain.tools import tool
from langfuse import get_client
from deps import get_vectorstore

lf = get_client()

@tool
def document_search(query: str, k: int = 3) -> str:
    """Use this tool when you need to answer any medical or clinical question.

    Args:
        query: The user's medical question as a plain string.
        k: Number of document chunks to retrieve. Default is 3.

    Returns:
        A string of the most relevant passages from the medical documents.
    """
    vectorstore = get_vectorstore()

    # similarity_search_with_score returns (doc, score) tuples
    # score is cosine similarity — higher = more relevant
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Log the retrieval as a span in Langfuse
    # as_type="span" marks this as a retrieval step, not an LLM call
    with lf.start_as_current_observation(as_type="span", name="retrieval") as obs:
        obs.update(
            input={"query": query, "k": k},
            output=[
                {"content": doc.page_content, "score": float(score)}
                for doc, score in results
            ],
        )

    return "\n\n".join(doc.page_content for doc, score in results)