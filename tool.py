# tool.py
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from deps import get_vectorstore

@tool
def document_search(query: str) -> str:
    """Use this tool when you need to answer any medical or clinical question.

    Args:
        query: The user's medical question as a plain string.

    Returns:
        A string of the most relevant passages from the medical documents.
    """
    k = 8
    print(f"[tool] document_search called: query={query!r}")
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as e:
        return f"Retrieval failed: {str(e)}"

    return "\n\n".join(doc.page_content for doc, _ in results)


@tool
def web_search(query: str) -> str:
    """Use this tool to search the web for information not found in the medical documents.

    Args:
        query: The search query as a plain string.

    Returns:
        A string of the most relevant web search results.
    """
    from tavily import TavilyClient
    import os
    print(f"[tool] web_search called: query={query!r}")
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, max_results=2)
        return "\n\n".join(
            f"{r['title']}\n{r['url']}\n{r['content']}"
            for r in results.get("results", [])
        )
    except Exception as e:
        return f"Web search failed: {str(e)}"