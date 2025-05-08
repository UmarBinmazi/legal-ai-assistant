# streamlit_app/ui.py
"""
Streamlit UI for Legal Assistant MVP.

This module:
- Creates a user-friendly interface for interacting with the legal assistant
- Handles document uploads (PDF, text files)
- Makes API calls to the FastAPI backend
- Displays search results and LLM-generated answers
- Manages user session state and history
"""

import streamlit as st
import requests
import json
import os
import uuid
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = os.environ.get("LEGAL_ASSISTANT_API_URL", "http://127.0.0.1:8000")
CHATS_DIR = "streamlit_app/chats"

# Create chats directory if it doesn't exist
os.makedirs(CHATS_DIR, exist_ok=True)

# Helper functions
def save_chat(chat_id, chat_data):
    """Save chat data to disk"""
    with open(os.path.join(CHATS_DIR, f"{chat_id}.json"), "w") as f:
        json.dump(chat_data, f)

def load_chat(chat_id):
    """Load chat data from disk"""
    try:
        with open(os.path.join(CHATS_DIR, f"{chat_id}.json"), "r") as f:
            return json.load(f)
    except Exception:
        # Return empty chat if file doesn't exist
        return {
            "id": chat_id,
            "name": "New Chat",
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

def get_all_chats():
    """Get list of all chats"""
    chats = []
    for filename in os.listdir(CHATS_DIR):
        if filename.endswith(".json"):
            chat_id = filename.replace(".json", "")
            chat_data = load_chat(chat_id)
            chats.append({
                "id": chat_id,
                "name": chat_data.get("name", "Unnamed Chat"),
                "created_at": chat_data.get("created_at", "")
            })
    
    # Sort by created date (newest first)
    chats.sort(key=lambda x: x["created_at"], reverse=True)
    return chats

def query_api(endpoint, data):
    """Send a query to the API"""
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {"error": str(e)}

def upload_document(file, case_number=None, year=None):
    """Upload a document to the API"""
    try:
        # Prepare the form data
        form_data = {
            "include_in_index": True
        }
        
        # These parameters are now always None from the UI
        # but we keep the function signature for backward compatibility
        
        # Create the files data
        files = {
            "file": (file.name, file.getvalue(), file.type)
        }
        
        # Send the request
        response = requests.post(
            f"{API_URL}/upload-document",
            data=form_data,
            files=files
        )
        
        return response.json()
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return {"error": str(e)}

# Initialize session state
if "current_chat_id" not in st.session_state:
    # Check if any chats exist
    existing_chats = get_all_chats()
    if existing_chats:
        st.session_state.current_chat_id = existing_chats[0]["id"]
    else:
        # Create a new chat
        new_chat_id = str(uuid.uuid4())
        save_chat(new_chat_id, {
            "id": new_chat_id,
            "name": "New Chat",
            "created_at": datetime.now().isoformat(),
            "messages": []
        })
        st.session_state.current_chat_id = new_chat_id

# Initialize other session state variables
if "show_reset_dialog" not in st.session_state:
    st.session_state.show_reset_dialog = False

# Load chat data for the current chat
chat_data = load_chat(st.session_state.current_chat_id)

# Sidebar for chat management
with st.sidebar:
    st.title("Legal AI Assistant")
    st.markdown("Your intelligent legal research companion")
    
    # Show API URL in settings
    st.sidebar.caption(f"API: {API_URL}")
    
    # New chat button
    if st.button("New Chat", key="new_chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        save_chat(new_chat_id, {
            "id": new_chat_id,
            "name": "New Chat",
            "created_at": datetime.now().isoformat(),
            "messages": []
        })
        st.session_state.current_chat_id = new_chat_id
        st.experimental_rerun()
    
    # Chat list
    st.subheader("Your Chats")
    chats = get_all_chats()
    
    for chat in chats:
        col1, col2, col3 = st.columns([5, 1, 1])
        
        # Chat name (clickable)
        with col1:
            if st.button(chat["name"], key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.experimental_rerun()
        
        # Edit button
        with col2:
            if st.button("📝", key=f"edit_{chat['id']}"):
                st.session_state.edit_chat_id = chat["id"]
                st.session_state.edit_chat_name = chat["name"]
        
        # Delete button
        with col3:
            if st.button("🗑️", key=f"delete_{chat['id']}"):
                try:
                    os.remove(os.path.join(CHATS_DIR, f"{chat['id']}.json"))
                    if st.session_state.current_chat_id == chat["id"]:
                        # If we're deleting the current chat, switch to another one
                        remaining_chats = [c for c in chats if c["id"] != chat["id"]]
                        if remaining_chats:
                            st.session_state.current_chat_id = remaining_chats[0]["id"]
                        else:
                            # Create a new chat if none left
                            new_chat_id = str(uuid.uuid4())
                            save_chat(new_chat_id, {
                                "id": new_chat_id,
                                "name": "New Chat",
                                "created_at": datetime.now().isoformat(),
                                "messages": []
                            })
                            st.session_state.current_chat_id = new_chat_id
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error deleting chat: {e}")
    
    # Chat rename dialog
    if "edit_chat_id" in st.session_state:
        with st.form(key="rename_chat_form"):
            st.subheader("Rename Chat")
            new_name = st.text_input(
                "Chat Name", 
                value=st.session_state.edit_chat_name
            )
            
            submit, cancel = st.columns(2)
            with submit:
                if st.form_submit_button("Save", use_container_width=True):
                    chat_data = load_chat(st.session_state.edit_chat_id)
                    chat_data["name"] = new_name
                    save_chat(st.session_state.edit_chat_id, chat_data)
                    del st.session_state.edit_chat_id
                    del st.session_state.edit_chat_name
                    st.experimental_rerun()
            
            with cancel:
                if st.form_submit_button("Cancel", use_container_width=True):
                    del st.session_state.edit_chat_id
                    del st.session_state.edit_chat_name
                    st.experimental_rerun()

    st.divider()
    
    # File upload section (moved below chat management)
    st.subheader("Upload Document")
    with st.form(key="upload_form"):
        uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"])
        submit_button = st.form_submit_button("Upload Document", use_container_width=True)
        
        if submit_button and uploaded_file is not None:
            with st.spinner("Uploading document..."):
                # Call upload without case number and year
                result = upload_document(uploaded_file)
                
                if "error" in result:
                    st.error(f"Upload failed: {result['error']}")
                else:
                    st.success(f"Document uploaded successfully")
                    
                    # Add upload info to chat
                    chat_data["messages"].append({
                        "role": "user",
                        "content": f"Uploaded document: {uploaded_file.name}"
                    })
                    
                    chat_data["messages"].append({
                        "role": "assistant",
                        "content": f"Document uploaded successfully. Document ID: {result['document_id']}",
                        "document_id": result["document_id"]
                    })
                    
                    # Save chat
                    save_chat(st.session_state.current_chat_id, chat_data)
                    st.experimental_rerun()

    # API stats
    st.divider()
    try:
        stats = requests.get(f"{API_URL}/index-stats").json()
        st.metric("Documents Indexed", stats.get("document_count", 0))
        
        # Index management
        if st.button("Reset Index", key="reset_index"):
            st.session_state.show_reset_dialog = True
        
        if st.session_state.get("show_reset_dialog", False):
            with st.form(key="reset_index_form"):
                st.warning("⚠️ This will delete all indexed documents. This action cannot be undone.")
                include_samples = st.checkbox("Include sample data in new index", value=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Confirm Reset", use_container_width=True):
                        response = requests.post(
                            f"{API_URL}/reset-index", 
                            params={"include_samples": include_samples}
                        )
                        if response.status_code == 200:
                            st.success("Index reset successfully")
                            # Refresh stats
                            stats = requests.get(f"{API_URL}/index-stats").json()
                            del st.session_state.show_reset_dialog
                            st.experimental_rerun()
                        else:
                            st.error(f"Error resetting index: {response.text}")
                
                with col2:
                    if st.form_submit_button("Cancel", use_container_width=True):
                        del st.session_state.show_reset_dialog
                        st.experimental_rerun()
    except:
        st.warning("Could not connect to API")

# Main chat area
st.title(chat_data["name"])

# Display chat messages
for msg in chat_data["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            
            # Display sources if available and they're from real documents (not samples)
            if "sources" in msg:
                # Filter out sample sources
                real_sources = [
                    source for source in msg["sources"] 
                    if source.get("source") != "sample"
                ]
                
                # Only show sources expander if there are actual sources
                if real_sources:
                    with st.expander("Sources"):
                        for source in real_sources:
                            st.write(f"**{source.get('case_number', 'Unknown')}** ({source.get('year', 'N/A')})")
                            if "text" in source:
                                st.text(source["text"])

# Query input
query = st.chat_input("Ask your legal question")

if query:
    # Add user message to chat
    st.chat_message("user").write(query)
    chat_data["messages"].append({
        "role": "user", 
        "content": query
    })
    
    # Save chat
    save_chat(st.session_state.current_chat_id, chat_data)
    
    # Check if query is about a document
    last_doc_msg = None
    for msg in reversed(chat_data["messages"]):
        if "document_id" in msg:
            last_doc_msg = msg
            break
    
    # Get the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if last_doc_msg and "Document ID:" in last_doc_msg["content"]:
                # Query the document
                document_id = last_doc_msg["document_id"]
                response = query_api("query-document", {
                    "query": query,
                    "document_id": document_id
                })
                
                answer = response.get("response", "I couldn't process this document query.")
                st.write(answer)
                
                # Get document details
                document_info = response.get("document", {})
                
                # Show document reference
                if document_info:
                    with st.expander("Document Reference"):
                        st.write(f"**Document**: {document_info.get('filename', 'Unknown')}")
                        st.write(f"**ID**: {document_info.get('document_id', 'Unknown')}")
                        st.write(f"**Case Number**: {document_info.get('case_number', 'Auto-generated')}")
                
                # Save response to chat
                chat_data["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "document_id": document_id,
                    "document_info": document_info
                })
            else:
                # General query
                response = query_api("query", {"query": query})
                
                answer = response.get("response", "I couldn't process your query.")
                sources = response.get("relevant_cases", [])
                
                st.write(answer)
                
                # Only show sources if there are non-sample sources
                real_sources = [s for s in sources if s.get("source") != "sample"]
                if real_sources:
                    with st.expander("Sources"):
                        for source in real_sources:
                            st.write(f"**{source.get('case_number', 'Unknown')}** ({source.get('year', 'N/A')})")
                            if "text" in source:
                                st.text(source["text"])
                
                chat_data["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
    
    # Save chat with assistant response
    save_chat(st.session_state.current_chat_id, chat_data)

# Add footer
st.divider()
st.markdown("**Legal AI Assistant** - Powered by FastAPI and Streamlit") 