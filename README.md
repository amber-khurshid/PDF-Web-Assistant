**PDF Web Assistant
**
A Retrieval-Augmented Generation (RAG) powered assistant designed to provide accurate and comprehensive answers by processing uploaded PDF documents and supplementing with web searches when necessary. Built with a modern tech stack including LangGraph, LangChain, FAISS, MongoDB, and Gradio for an intuitive user interface.

**Features**





**PDF Document Processing:** Upload and process PDF files to extract relevant information using vector search with FAISS and HuggingFace embeddings.



**Web Search Fallback:** Automatically performs web searches via the Tavily API when answers are not found in uploaded documents.



**Persistent Chat History:** Stores conversation history in MongoDB, allowing users to resume sessions and review past interactions.



**Gradio Interface:** User-friendly web interface for uploading documents, asking questions, and managing chat sessions.



**Robust Error Handling:** Comprehensive logging and error management for reliable performance.

**Tech Stack**


**Python:** Core programming language.



**LangGraph:** For orchestrating complex workflows and managing the RAG pipeline with dynamic agent-based interactions.



**LangChain:** For building the RAG pipeline and integrating LLMs and tools.



**FAISS:** Efficient vector store for document retrieval.



**HuggingFace Embeddings:** For generating text embeddings (all-MiniLM-L6-v2).



**MongoDB:** Persistent storage for chat history and document metadata.



**Gradio:** Web-based interface for user interaction.



**Tavily API:** For real-time web search capabilities.



**Mistral-7B-Instruct-v0.3:** LLM for generating responses (via Together API).



**Environment Management:** Uses python-dotenv for secure configuration.
