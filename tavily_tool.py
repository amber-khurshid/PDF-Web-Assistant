import os
import logging
from dotenv import load_dotenv
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

from langchain_core.tools import tool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    logger.error("Missing TAVILY_API_KEY environment variable")
    raise ValueError("TAVILY_API_KEY not found")

# Initialize Tavily tool
tavily_tool = TavilySearch(
    api_key=tavily_api_key,
    max_results=5,
    search_depth="advanced"
)

@tool
def tavily_search_tool(query: str) -> str:
    """
    Searches the internet using the Tavily API when information is not found in the PDF.
    """
    try:
        logger.info(f"Tavily Tool: Processing query: {query}")
        results = tavily_tool.invoke(query)
        
        if not isinstance(results, list) or not results:
            logger.info("No results returned from Tavily")
            return "No web results found"
            
        content_parts = [f"Source {i} ({result.get('url', 'No URL')}): {result.get('content', 'No content')[:400]}..." 
                         for i, result in enumerate(results[:3], 1) if result.get('content', '').strip()]
        
        if content_parts:
            return f"Web Search Results: {'\n\n'.join(content_parts)}"
        return "No relevant web content found"
            
    except Exception as e:
        logger.error(f"Tavily Tool Error: {str(e)}", exc_info=True)
        return f"Error: Unable to fetch web results - {str(e)}"