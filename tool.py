from langchain.tools import tool
from deps import get_vectorstore

@tool
def document_search(query: str, k: int = 3) -> str:
    """Use this tool when you need to answer any medical or clinical question.
    
    Args:
        query: The user's medical question as a plain string.
        k: Number of document chunks to retrieve. Default is 3.
    
    Returns:
        A string of the most relevant passages from the medical documents.
    """
    #create a vectorstore instance (cached for efficiency)
    vectorstore = get_vectorstore()
    #conduct similarity search to find relevant document chunks k
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in results)