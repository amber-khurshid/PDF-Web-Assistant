import logging
from langchain_core.tools import tool
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tool
def pdf_retrieval_tool(query: str, vector_store: Any = None) -> str:
    """
    Retrieves information from a PDF document using vector search.
    """
    if vector_store is None:
        logger.error("Vector store not provided for PDF retrieval")
        return "Error: No PDF documents uploaded yet."
    
    try:
        logger.info(f"PDF Tool: Processing query: {query}")
        results = vector_store.similarity_search_with_score(query, k=3)
        
        if not results:
            logger.info("No similarity search results")
            return "NO_RELEVANT_CONTENT"
            
        best_doc, best_score = results[0]
        similarity = 1 / (1 + best_score)
        logger.info(f"PDF Tool: Best similarity score: {similarity:.4f}, Score: {best_score}")
        
        relevant_content = [doc.page_content.strip() for doc, score in results if (1 / (1 + score)) > 0.4 and doc.page_content.strip()]
        
        if relevant_content:
            logger.info(f"PDF Tool: Found {len(relevant_content)} relevant results")
            return f"PDF Content: {'\n\n'.join(relevant_content[:2])}"
        
        logger.info("PDF Tool: No relevant result found")
        return "NO_RELEVANT_CONTENT"
    except Exception as e:
        logger.error(f"PDF Tool Error: {str(e)}", exc_info=True)
        return f"Error: Failed to process PDF retrieval - {str(e)}"