# tool.py
from langchain.tools import tool
from deps import get_vectorstore

try:
    from langfuse import get_client
    lf = get_client()
except Exception:
    lf = None

def make_document_search(testing=False):
    @tool
    def document_search(query: str, k: int = 3) -> str:
        """Use this tool when you need to answer any medical or clinical question.

        Args:
            query: The user's medical question as a plain string.
            k: Number of document chunks to retrieve. Default is 3.

        Returns:
            A string of the most relevant passages from the medical documents.
        """
        try:
            vectorstore = get_vectorstore()
            results = vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            return f"Retrieval failed: {str(e)}"

        if not testing and lf is not None:
            try:
                with lf.start_as_current_observation(as_type="span", name="retrieval") as obs:
                    obs.update(
                        input={"query": query, "k": k},
                        output=[{"content": doc.page_content, "score": float(score)} for doc, score in results],
                    )
            except Exception as e:
                print(f"Langfuse logging failed: {e}")

        return "\n\n".join(doc.page_content for doc, _ in results)
    return document_search

# default instance used by agent.py — set testing=True in sandbox to skip Langfuse logging
document_search = make_document_search()