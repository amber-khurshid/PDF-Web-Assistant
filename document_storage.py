import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from history import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentStorage:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def save_document_metadata(self, session_id: str, filename: str, file_size: int, 
                              content_hash: str, chunk_count: int) -> bool:
        try:
            collection = self.db_manager.get_collection("documents")
            document = {
                "session_id": session_id,
                "filename": filename,
                "file_size": file_size,
                "content_hash": content_hash,
                "chunk_count": chunk_count,
                "upload_timestamp": datetime.utcnow(),
                "status": "processed"
            }
            result = collection.insert_one(document)
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")
            return False
    
    def get_session_documents(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            collection = self.db_manager.get_collection("documents")
            return list(collection.find({"session_id": session_id}))
        except Exception as e:
            logger.error(f"Error getting session documents: {e}")
            return []

class DocumentProcessor:
    def __init__(self):
        self.document_storage = DocumentStorage()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
    
    def process_pdf_file(self, file_path: str, filename: str, session_id: str) -> Tuple[bool, str, int]:
        try:
            if not file_path.lower().endswith('.pdf'):
                return False, "Only PDF files are supported", 0
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                return False, "Failed to load PDF document", 0
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                return False, "Failed to create document chunks", 0
            
            with open(file_path, 'rb') as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            
            file_size = os.path.getsize(file_path)
            success = self.document_storage.save_document_metadata(
                session_id, filename, file_size, content_hash, len(chunks)
            )
            
            if success:
                return True, f"Processed {filename} with {len(chunks)} chunks", len(chunks)
            return False, "Failed to save document metadata", 0
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
            return False, f"Error processing PDF: {str(e)}", 0
    
    def get_session_document_info(self, session_id: str) -> Dict[str, Any]:
        try:
            documents = self.document_storage.get_session_documents(session_id)
            total_size = sum(doc.get("file_size", 0) for doc in documents) / (1024 * 1024)
            total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)
            
            return {
                "document_count": len(documents),
                "total_size_mb": round(total_size, 2),
                "total_chunks": total_chunks,
                "documents": documents
            }
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {"document_count": 0, "total_size_mb": 0, "total_chunks": 0, "documents": []}