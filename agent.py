import os
import sys
import logging
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import FAISS
from tavily_tool import tavily_search_tool
from pdf_retrieval import pdf_retrieval_tool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Loaded environment variables")
together_api_key = os.getenv("TOGETHER_API_KEY")
if not together_api_key:
    logger.error("Missing TOGETHER_API_KEY environment variable")
    sys.exit(1)

SYSTEM_PROMPT = """
You are an assistant that answers user queries accurately and comprehensively based solely on the provided context. 
First, ALWAYS try to find the answer in the provided PDF document using the pdf_retrieval_tool. 
If the PDF doesn't contain relevant information, you must use the tavily_search_tool to search the web.
Provide clear, accurate, and relevant responses strictly based on the retrieved information. If the information is insufficient, state clearly that no relevant information was found and avoid making assumptions or generating unverified content.
"""

# Define LLM with bound tools
llm = ChatTogether(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.2,
    api_key=together_api_key
).bind_tools([pdf_retrieval_tool, tavily_search_tool])

# Custom State
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    thread_id: str
    pdf_searched: bool
    web_searched: bool
    final_answer: str
    pdf_found_content: bool

def pdf_search_node(state: State, vector_store: FAISS) -> dict:
    logger.info(f"---PDF SEARCH NODE (Thread ID: {state['thread_id']})---")
    query = state["query"]
    if vector_store is None:
        logger.error("No vector store available for PDF search")
        return {
            "pdf_searched": True,
            "web_searched": False,
            "final_answer": "Error: No PDF documents uploaded yet.",
            "pdf_found_content": False,
            "messages": [AIMessage(content="Error: No PDF documents uploaded yet.")]
        }
    
    pdf_result = pdf_retrieval_tool.invoke({"query": query, "vector_store": vector_store})
    
    if pdf_result != "NO_RELEVANT_CONTENT" and not pdf_result.startswith("Error:"):
        prompt = f"""
Based on the following information from the PDF document, provide a clear and comprehensive answer to the user's query: "{query}"

PDF Information:
{pdf_result}

Please provide a well-structured response based on the PDF content. Make sure to be specific and accurate.
"""
        try:
            response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
            final_answer = response.content
            logger.info("PDF-based response generated successfully")
        except Exception as e:
            logger.error(f"Error generating PDF response: {e}", exc_info=True)
            final_answer = f"Based on the PDF: {pdf_result}"
        
        return {
            "pdf_searched": True,
            "web_searched": False,
            "final_answer": final_answer,
            "pdf_found_content": True,
            "messages": [AIMessage(content=final_answer)]
        }
    else:
        logger.info("PDF search unsuccessful - proceeding to web search")
        return {
            "pdf_searched": True,
            "web_searched": False,
            "final_answer": "",
            "pdf_found_content": False,
            "messages": [AIMessage(content="Information not found in PDF, searching web...")]
        }

def web_search_node(state: State) -> dict:
    logger.info(f"---WEB SEARCH NODE (Thread ID: {state['thread_id']})---")
    query = state["query"]
    web_result = tavily_search_tool.invoke(query)
    
    prompt = f"""
The user asked: "{query}"

The information was not found in the PDF document, so I searched the web and found the following:

{web_result}

Provide a clear and comprehensive answer based solely on these web search results. If the web results are not relevant or insufficient, state clearly: "The web search results do not contain enough information to answer the query. """
    
    try:
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        final_answer = response.content
        logger.info("Web-based response generated successfully")
    except Exception as e:
        logger.error(f"Error generating web response: {e}", exc_info=True)
        final_answer = f"Based on web search: {web_result}"
    
    return {
        "web_searched": True,
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }

def route_after_pdf(state: State) -> str:
    logger.info(f"Routing decision - PDF found content: {state.get('pdf_found_content', False)}")
    return END if state.get("pdf_found_content", False) else "web_search"

# def create_workflow(vector_store: FAISS):
#     workflow = StateGraph(State)
#     workflow.add_node("pdf_search", lambda state: pdf_search_node(state, vector_store))
#     workflow.add_node("web_search", web_search_node)
#     workflow.add_conditional_edges("pdf_search", route_after_pdf, {"web_search": "web_search", END: END})
#     workflow.add_edge("web_search", END)
#     return workflow.compile()

def create_workflow(vector_store: FAISS):
    workflow = StateGraph(State)
    workflow.add_node("pdf_search", lambda state: pdf_search_node(state, vector_store))
    workflow.add_node("web_search", web_search_node)
    
    # Set the entry point to the pdf_search node
    workflow.add_edge("__start__", "pdf_search")
    
    # Define conditional edges after pdf_search
    workflow.add_conditional_edges("pdf_search", route_after_pdf, {"web_search": "web_search", END: END})
    workflow.add_edge("web_search", END)
    return workflow.compile()

class RAGPipeline:
    def __init__(self, document_processor, chat_storage):
        logger.info("Initializing RAG Pipeline")
        self.document_processor = document_processor
        self.chat_storage = chat_storage
        self.graph = None
        logger.info("RAG Pipeline initialized")
    
    def update_workflow(self):
        """Initialize or update the LangGraph workflow after vector store is created."""
        if self.document_processor.vector_store is not None:
            self.graph = create_workflow(self.document_processor.vector_store)
            logger.info("LangGraph workflow updated")
        else:
            logger.warning("Cannot update workflow: No vector store available")
    
    def query(self, user_query: str, thread_id: str = "default_thread") -> str:
        logger.info(f"RAG Pipeline: Querying with: {user_query}")
        
        try:
            # Retrieve conversation history
            history = self.chat_storage.load_history(thread_id, limit=10)
            history_context = ""
            if history:
                history_context = "\n".join(
                    f"{msg[0].capitalize()}: {msg[1]}" for msg in history
                    if msg[0] in ["user", "assistant"] and msg[1].strip()
                )
                logger.info(f"Retrieved history for thread_id={thread_id}: {len(history)} messages")

            if not self.document_processor.vector_store:
                logger.error("No vector store available for query")
                self.chat_storage.save_message(thread_id, user_query, "user")
                error_message = "Error: No PDF documents uploaded yet. Please upload a PDF first."
                self.chat_storage.save_message(thread_id, error_message, "assistant")
                return error_message
            
            if not self.graph:
                self.update_workflow()
                if not self.graph:
                    error_message = "Error: Failed to initialize workflow due to missing vector store."
                    self.chat_storage.save_message(thread_id, user_query, "user")
                    self.chat_storage.save_message(thread_id, error_message, "assistant")
                    return error_message
            
            history_relevant = any(
                keyword in user_query.lower() for keyword in ["earlier", "previous", "before", "last", "what did you say"]
            )

            if history_relevant and history_context:
                prompt_prefix = f"""
Conversation History:
{history_context}

Current Query: {user_query}

Please answer the query, taking into account the conversation history if relevant. 
If the query refers to prior messages, use the history to provide a context-aware response. 
Otherwise, proceed with the standard retrieval process using the PDF or web search tools.
"""
            else:
                prompt_prefix = f"Current Query: {user_query}"

            initial_state = {
                "messages": [HumanMessage(content=prompt_prefix)],
                "query": user_query,
                "thread_id": thread_id,
                "pdf_searched": False,
                "web_searched": False,
                "final_answer": "",
                "pdf_found_content": False
            }
            
            result = self.graph.invoke(initial_state)
            answer = result["final_answer"]
            
            if not answer or answer.strip() == "":
                answer = "I apologize, but I couldn't find relevant information to answer your question."
            
            self.chat_storage.save_message(thread_id, user_query, "user")
            self.chat_storage.save_message(thread_id, answer, "assistant")
            return answer
            
        except Exception as e:
            logger.error(f"RAG Pipeline Error: {str(e)}", exc_info=True)
            error_message = f"Sorry, I encountered an error while processing your query: {str(e)}"
            self.chat_storage.save_message(thread_id, user_query, "user")
            self.chat_storage.save_message(thread_id, error_message, "assistant")

            
            return error_message
