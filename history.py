import os
import logging
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    logger.error("Missing MONGO_URI environment variable")
    raise ValueError("MONGO_URI not found")

class DatabaseManager:
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._initialize_connection()
            self.initialized = True
    
    def _initialize_connection(self):
        try:
            self._client = MongoClient(
                mongo_uri,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
                serverSelectionTimeoutMS=10000
            )
            self._client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @property
    def client(self):
        if self._client is None:
            self._initialize_connection()
        return self._client
    
    def get_collection(self, collection_name: str):
        try:
            db = self.client["chat_db"]
            collection = db[collection_name]
            
            if collection_name == "chat_history":
                collection.create_index([("session_id", 1), ("timestamp", 1)])
                collection.create_index("session_id")
            elif collection_name == "documents":
                collection.create_index("session_id")
                collection.create_index("filename")
            elif collection_name == "sessions":
                collection.create_index("session_id", unique=True)  # Unique index for session_id
                collection.create_index("created_at")
            
            return collection
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise
    
    def close_connection(self):
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("MongoDB connection closed")

class ChatHistoryStorage:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def save_session(self, session_id: str, metadata: Optional[Dict] = None) -> bool:
        """Save a new session ID with optional metadata to the sessions collection."""
        if not session_id:
            logger.warning("Invalid session_id for save_session")
            return False
        
        try:
            collection = self.db_manager.get_collection("sessions")
            document = {
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "metadata": metadata or {}
            }
            result = collection.insert_one(document)
            logger.info(f"Session {session_id} saved successfully")
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Save session error for session_id={session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by its ID."""
        try:
            collection = self.db_manager.get_collection("sessions")
            session = collection.find_one({"session_id": session_id})
            return session
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve all session IDs with their metadata, sorted by creation time."""
        try:
            collection = self.db_manager.get_collection("sessions")
            sessions = collection.find().sort("created_at", -1).limit(limit)
            return [{
                "session_id": session["session_id"],
                "created_at": session["created_at"],
                "metadata": session["metadata"]
            } for session in sessions]
        except Exception as e:
            logger.error(f"Error retrieving all sessions: {e}")
            return []
    
    def save_message(self, session_id: str, message: str, role: str, metadata: Optional[Dict] = None) -> bool:
        """Save a message to the chat_history collection, ensuring session_id exists in sessions collection."""
        if not all([session_id, message, role]):
            logger.warning("Invalid parameters for save_message")
            return False
        
        if role not in ["user", "assistant", "system"]:
            logger.warning(f"Invalid role: {role}")
            return False
        
        # Ensure session_id exists in sessions collection
        if not self.get_session(session_id):
            self.save_session(session_id)
        
        try:
            collection = self.db_manager.get_collection("chat_history")
            document = {
                "session_id": session_id,
                "content": message,
                "role": role,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }
            result = collection.insert_one(document)
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Save message error for session_id={session_id}: {e}")
            return False
    
    def load_history(self, session_id: str, limit: int = 50) -> List[Tuple[str, str, datetime]]:
        """Load chat history for a given session_id."""
        try:
            collection = self.db_manager.get_collection("chat_history")
            messages = collection.find({"session_id": session_id}).sort("timestamp", 1).limit(limit)
            return [(msg.get("role", "unknown"), msg.get("content", ""), msg.get("timestamp", datetime.utcnow()))
                    for msg in messages]
        except Exception as e:
            logger.error(f"Load history error for session_id={session_id}: {e}")
            return []
    
    def get_user_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve user sessions with aggregated data from chat_history."""
        try:
            collection = self.db_manager.get_collection("chat_history")
            pipeline = [
                {"$group": {
                    "_id": "$session_id",
                    "last_message": {"$last": "$timestamp"},
                    "message_count": {"$sum": 1},
                    "first_message": {"$first": "$content"}
                }},
                {"$sort": {"last_message": -1}},
                {"$limit": limit}
            ]
            return [{
                "session_id": session["_id"],
                "last_activity": session["last_message"],
                "message_count": session["message_count"],
                "preview": session["first_message"][:50] + "..." if len(session["first_message"]) > 50 else session["first_message"]
            } for session in collection.aggregate(pipeline)]
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []






