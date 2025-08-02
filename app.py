import gradio as gr
import logging
import os
import uuid
from typing import List, Dict, Any, Tuple
from agent import RAGPipeline
from document_storage import DocumentProcessor
from history import ChatHistoryStorage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetSolChatInterface:
    def __init__(self):
        self.pipeline = RAGPipeline(DocumentProcessor(), ChatHistoryStorage())
        self.document_processor = self.pipeline.document_processor
        self.chat_storage = self.pipeline.chat_storage
        self.current_session_id = str(uuid.uuid4())
    
    def upload_pdf(self, files: List[Any]) -> Tuple[str, str]:
        try:
            if not files:
                return "No files uploaded.", ""
            
            upload_results = []
            total_chunks = 0
            
            for file in files:
                if file is None:
                    upload_results.append("❌ No file provided")
                    continue
                
                file_path = file.name if hasattr(file, 'name') else str(file)
                if not os.path.exists(file_path):
                    upload_results.append(f"❌ {file_path}: File not found")
                    continue
                
                filename = os.path.basename(file_path)
                
                success, message, chunk_count = self.document_processor.process_pdf_file(
                    file_path, filename, self.current_session_id
                )
                
                if success:
                    upload_results.append(f"✅ {filename}: {message}")
                    total_chunks += chunk_count
                else:
                    upload_results.append(f"❌ {filename}: {message}")
            
            self.pipeline.update_workflow()
            
            doc_info = self.document_processor.get_session_document_info(self.current_session_id)
            session_info = self._format_session_info(doc_info)
            
            result_message = "\n".join(upload_results)
            if total_chunks > 0:
                result_message += f"\n\n📊 Total chunks processed: {total_chunks}"
            
            return result_message, session_info
            
        except Exception as e:
            logger.error(f"File upload error: {e}", exc_info=True)
            return f"❌ Upload failed: {str(e)}", ""
    
    def new_session(self) -> Tuple[List, str, str]:
        try:
            self.current_session_id = str(uuid.uuid4())
            logger.info(f"Started new session: {self.current_session_id}")
            
            return [], "", f"🆕 New session started\n\n📋 Session ID: {self.current_session_id[:8]}...\n📁 Documents: 0\n📄 Chunks: 0"
            
        except Exception as e:
            logger.error(f"New session error: {e}")
            return [], "", "Error starting new session"
    
    def chat_response(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        try:
            if not message.strip():
                return history, ""
            
            response = self.pipeline.query(message, self.current_session_id)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            doc_info = self.document_processor.get_session_document_info(self.current_session_id)
            session_info = self._format_session_info(doc_info)
            
            return history, session_info
            
        except Exception as e:
            logger.error(f"Chat response error: {e}", exc_info=True)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "Error processing your request"})
            return history, ""
    
    def load_session(self, session_id: str) -> Tuple[List[Dict[str, str]], str]:
        try:
            if not session_id.strip():
                return [], "Please provide a valid session ID"
            
            history_tuples = self.chat_storage.load_history(session_id)
            gradio_history = []
            
            for role, content, _ in history_tuples:
                if role in ["user", "assistant"]:
                    gradio_history.append({"role": role, "content": content})
            
            self.current_session_id = session_id
            doc_info = self.document_processor.get_session_document_info(session_id)
            session_info = self._format_session_info(doc_info)
            
            return gradio_history, session_info
            
        except Exception as e:
            logger.error(f"Load session error: {e}")
            return [], f"Error loading session: {str(e)}"
    
    def get_session_list(self) -> str:
        try:
            sessions = self.chat_storage.get_user_sessions(limit=10)
            if not sessions:
                return "No previous sessions found."
            
            session_list = ["📋 Recent Sessions:", ""]
            for i, session in enumerate(sessions, 1):
                session_id = session["session_id"][:8] + "..."
                last_activity = session["last_activity"].strftime("%Y-%m-%d %H:%M")
                message_count = session["message_count"]
                preview = session["preview"]
                
                session_list.append(f"{i}. **{session_id}** ({message_count} messages)")
                session_list.append(f"   Last active: {last_activity}")
                session_list.append(f"   Preview: *{preview}*")
                session_list.append("")
            
            return "\n".join(session_list)
            
        except Exception as e:
            logger.error(f"Get session list error: {e}")
            return "Error retrieving session list"
    
    def _format_session_info(self, doc_info: Dict[str, Any]) -> str:
        try:
            session_short_id = self.current_session_id[:8] + "..."
            doc_count = doc_info.get("document_count", 0)
            chunk_count = doc_info.get("total_chunks", 0)
            total_size = doc_info.get("total_size_mb", 0)
            
            info_parts = [
                f"📋 Session: {session_short_id}",
                f"📁 Documents: {doc_count}",
                f"📄 Chunks: {chunk_count}"
            ]
            
            if total_size > 0:
                info_parts.append(f"💾 Size: {total_size}MB")
            
            documents = doc_info.get("documents", [])
            if documents:
                info_parts.append("\n📚 Uploaded Files:")
                for doc in documents:
                    filename = doc.get("filename", "Unknown")
                    chunks = doc.get("chunk_count", 0)
                    info_parts.append(f"  • {filename} ({chunks} chunks)")
            
            return "\n".join(info_parts)
            
        except Exception as e:
            logger.error(f"Format session info error: {e}")
            return "Session info unavailable"
    
    def create_interface(self):
        with gr.Blocks(
            title="NetSol Financial Assistant",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container { max-width: 1200px !important; }
            .chat-container { height: 500px; }
            .session-info { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; }
            """
        ) as interface:
           
            gr.Markdown(
                """
                # 💼 PDF Web Assistant
                **Upload PDF documents and ask questions about their content.**
                - 📄 Upload PDFs for document-based answers
                - 🌐 Automatic web search when documents don't contain the answer
                - 💬 Conversation history preserved across sessions
                 """
)
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="💬 Chat", height=400, type='messages')
                    with gr.Row():
                        msg_textbox = gr.Textbox(label="Your Question", placeholder="Ask about NetSol financials...", lines=2, scale=4)
                        send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat 🗑️")
                        new_session_btn = gr.Button("New Session 🆕")
                
                with gr.Column(scale=1):
                    session_info = gr.Markdown(f"📋 Session: {self.current_session_id[:8]}...\n📁 Documents: 0\n📄 Chunks: 0", elem_classes=["session-info"])
                    gr.Markdown("### 📁 Upload Documents")
                    file_upload = gr.File(label="Upload PDF Files", file_count="multiple", file_types=[".pdf"])
                    upload_status = gr.Textbox(label="Upload Status", lines=4, interactive=False)
                    gr.Markdown("### 🔄 Session Management")
                    with gr.Row():
                        session_id_input = gr.Textbox(label="Session ID", placeholder="Enter session ID to load...", scale=2)
                        load_btn = gr.Button("Load 📂", scale=1)
                    sessions_btn = gr.Button("Show Recent Sessions 📋")
                    sessions_display = gr.Markdown("Click 'Show Recent Sessions' to see your conversation history.")
            
            msg_textbox.submit(self.chat_response, [msg_textbox, chatbot], [chatbot, session_info])
            msg_textbox.submit(lambda: "", None, [msg_textbox])
            send_btn.click(self.chat_response, [msg_textbox, chatbot], [chatbot, session_info])
            send_btn.click(lambda: "", None, [msg_textbox])
            file_upload.upload(self.upload_pdf, [file_upload], [upload_status, session_info])
            new_session_btn.click(self.new_session, outputs=[chatbot, upload_status, session_info])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_textbox])
            load_btn.click(self.load_session, [session_id_input], [chatbot, session_info])
            sessions_btn.click(self.get_session_list, outputs=[sessions_display])
        
        return interface

def main():
    try:
        chat_interface = NetSolChatInterface()
        interface = chat_interface.create_interface()
        logger.info("Gradio interface launched successfully")
        interface.launch()
        # logger.info("Gradio interface launched successfully")
    except Exception as e:
        logger.error(f"Application startup error: {e}", exc_info=True)
        print(f"Failed to start application: {e}")
    finally:
        chat_interface.pipeline.chat_storage.db_manager.close_connection()

if __name__ == "__main__":
    main()