# api.py

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from app.embedding import InLegalBERTEmbedder
from app.retrieval import LegalCaseRetriever, load_faiss_index, search_similar_cases
from app.llm import ask_llm
from app.utils import process_document, extract_text_with_ocr
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil

# Initialize embedder
embedder = InLegalBERTEmbedder()

# Initialize FastAPI app with detailed documentation
app = FastAPI(
    title="Legal AI Assistant",
    description="A powerful legal assistant for lawyers. Supports direct queries and document analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

print(f"Using data directory: {DATA_DIR}")
print(f"Index path: {INDEX_PATH}")
print(f"Metadata path: {METADATA_PATH}")

# Load or initialize FAISS index
try:
    index, case_metadata = load_faiss_index(INDEX_PATH, METADATA_PATH)
    retriever = LegalCaseRetriever()
    retriever.index = index
    retriever.metadata = case_metadata
    print(f"Loaded existing index with {len(case_metadata)} documents")
except Exception as e:
    print(f"Creating new index: {e}")
    retriever = LegalCaseRetriever()
    
    # Check if sample data should be included
    include_samples = os.environ.get("LEGAL_ASSISTANT_NO_SAMPLES") != "1"
    
    if include_samples:
        # Sample data for new index
        print("Adding sample data to new index")
        texts = ["Article 21 right to life", "High court bail judgment", "Criminal appeal precedent"]
        embeddings = [embedder.embed_text(text) for text in texts]
        metadata = [
            {"case_number": "ART21-2018", "year": 2018, "text": texts[0], "source": "sample"},
            {"case_number": "BAIL-2020", "year": 2020, "text": texts[1], "source": "sample"},
            {"case_number": "CRIM-2022", "year": 2022, "text": texts[2], "source": "sample"},
        ]
        retriever.add_to_index(embeddings, metadata)
    else:
        print("Creating empty index (no sample data)")
    
    retriever.save_index(INDEX_PATH, METADATA_PATH)
    index = retriever.index
    case_metadata = retriever.metadata

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class DocumentQueryRequest(BaseModel):
    query: str
    document_id: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    success: bool
    text_length: int
    message: str

class IndexStats(BaseModel):
    document_count: int
    sample_entries: List[dict]

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Legal AI Assistant API is live. Use /docs for API documentation."}

# Direct query endpoint
@app.post("/query")
def query_legal_assistant(request: QueryRequest):
    """
    Process a direct legal query without reference to specific documents.
    Returns related case information and an answer.
    """
    try:
        # Embed query
        query_embedding = embedder.embed_text(request.query)
        
        # Retrieve similar cases with a relevance threshold
        # For L2 distance, lower is better, so we filter out high distance results
        top_docs = search_similar_cases(
            query_embedding, 
            index, 
            case_metadata, 
            k=5,  # Get more candidates
            threshold=150.0  # Only keep results below this distance
        )
        
        # Handle case where no relevant documents were found
        if not top_docs:
            return {
                "response": "I couldn't find any relevant legal information for your query. Could you please provide more details or rephrase your question?",
                "relevant_cases": []
            }
        
        # Get LLM response
        answer = ask_llm(request.query, top_docs)
        
        return {"response": answer, "relevant_cases": top_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document upload endpoint
@app.post("/upload-document", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    include_in_index: Optional[bool] = Form(True),
    case_number: Optional[str] = Form(None),
    year: Optional[int] = Form(None)
):
    """
    Upload a legal document (PDF, DOCX, etc.) for processing.
    Automatically extracts text with OCR fallback for scanned documents.
    Optionally adds the document to the vector index.
    """
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create document directory
        doc_dir = os.path.join(UPLOADS_DIR, document_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(doc_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document (extract text, handle OCR if needed)
        text, used_ocr = process_document(file_path)
        
        # Save extracted text
        text_path = os.path.join(doc_dir, "extracted_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Add to index if requested
        if include_in_index and text:
            # Create metadata
            metadata = {
                "document_id": document_id,
                "filename": file.filename,
                "case_number": case_number or f"DOC-{document_id[:8]}",
                "year": year or 2023,
                "text": text[:1000],  # Store a preview of the text
                "source": "upload",
                "path": file_path
            }
            
            # Embed and add to index
            embedding = embedder.embed_text(text)
            retriever.add_to_index([embedding], [metadata])
            retriever.save_index(INDEX_PATH, METADATA_PATH)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            success=True,
            text_length=len(text),
            message=f"Document processed successfully. {'OCR was used.' if used_ocr else 'Text extracted directly.'}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document query endpoint
@app.post("/query-document")
def query_document(request: DocumentQueryRequest):
    """
    Ask a question about a specific uploaded document.
    Uses the document content as context for the LLM.
    """
    try:
        # Find document in metadata
        doc_matches = [doc for doc in case_metadata if doc.get("document_id") == request.document_id]
        
        if not doc_matches:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document path
        doc_dir = os.path.join(UPLOADS_DIR, request.document_id)
        text_path = os.path.join(doc_dir, "extracted_text.txt")
        
        # Read full text
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            raise HTTPException(status_code=404, detail="Document text not found")
        
        # Create a special context for the LLM with the document content
        doc_context = [{
            "case_number": doc_matches[0].get("case_number", "Unknown"),
            "year": doc_matches[0].get("year", "Unknown"),
            "text": full_text[:5000],  # First 5000 chars for context
            "is_document_context": True,  # Flag to indicate this is from a specific document
            "document_id": request.document_id,
            "filename": doc_matches[0].get("filename", "Unknown")
        }]
        
        # Get LLM response
        answer = ask_llm(request.query, doc_context)
        
        return {
            "response": answer,
            "document": {
                "document_id": request.document_id,
                "filename": doc_matches[0].get("filename", "Unknown"),
                "case_number": doc_matches[0].get("case_number", "Unknown")
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index stats endpoint
@app.get("/index-stats", response_model=IndexStats)
def get_index_stats():
    """
    Get statistics about the current vector index.
    Returns the number of documents and sample entries.
    """
    sample_size = min(5, len(case_metadata))
    samples = case_metadata[:sample_size]
    
    return IndexStats(
        document_count=len(case_metadata),
        sample_entries=samples
    )

# Index management endpoints
@app.post("/reset-index")
def reset_index(include_samples: bool = True):
    """
    Reset the index. Removes all indexed documents.
    Optionally adds back sample data.
    Warning: This is irreversible.
    """
    global retriever, index, case_metadata
    
    retriever.reset()
    
    # Add back sample data if requested
    if include_samples:
        print("Adding sample data to reset index")
        texts = ["Article 21 right to life", "High court bail judgment", "Criminal appeal precedent"]
        embeddings = [embedder.embed_text(text) for text in texts]
        metadata = [
            {"case_number": "ART21-2018", "year": 2018, "text": texts[0], "source": "sample"},
            {"case_number": "BAIL-2020", "year": 2020, "text": texts[1], "source": "sample"},
            {"case_number": "CRIM-2022", "year": 2022, "text": texts[2], "source": "sample"},
        ]
        retriever.add_to_index(embeddings, metadata)
    else:
        print("Creating empty index (no sample data)")
    
    retriever.save_index(INDEX_PATH, METADATA_PATH)
    
    index = retriever.index
    case_metadata = retriever.metadata
    
    return {
        "message": f"Index reset successfully. Sample data included: {include_samples}", 
        "document_count": len(case_metadata)
    }
