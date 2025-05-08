# app/utils.py
"""
Utility functions for the Legal Assistant MVP.

This module contains helper functions for:
- Token handling and authentication
- Text preprocessing (cleaning, normalization)
- Document chunking for embeddings
- PDF parsing and extraction
- Logging and error handling
- Configuration management
"""

import os
import re
import logging
from typing import Tuple, List, Dict, Optional
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image
import io
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Document processing functions
def process_document(file_path: str) -> Tuple[str, bool]:
    """
    Process a document file and extract its text content.
    Falls back to OCR if regular text extraction fails or returns minimal text.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Tuple of (extracted_text, used_ocr_flag)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # PDF processing
    if file_ext == '.pdf':
        return process_pdf(file_path)
    
    # Word document processing
    elif file_ext in ['.docx', '.doc']:
        try:
            return process_docx(file_path), False
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            return "", False
    
    # Text file processing
    elif file_ext in ['.txt', '.md', '.rtf']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), False
        except UnicodeDecodeError:
            # Try with a different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read(), False
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return "", False
    
    # Unsupported format
    else:
        logger.warning(f"Unsupported file format: {file_ext}")
        return "", False

def process_pdf(pdf_path: str) -> Tuple[str, bool]:
    """
    Process a PDF file, extracting text and falling back to OCR if needed.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, used_ocr_flag)
    """
    text = ""
    used_ocr = False
    
    try:
        # Try regular text extraction first
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text + "\n\n"
        
        # Check if we got meaningful text (more than just page numbers, headers, etc.)
        if len(text.strip()) < 100 or is_mostly_formatting(text):
            logger.info("PDF contains minimal text. Falling back to OCR.")
            text, used_ocr = ocr_pdf(pdf_path), True
    
    except Exception as e:
        logger.error(f"Error in regular PDF text extraction: {e}")
        # Fall back to OCR
        text, used_ocr = ocr_pdf(pdf_path), True
    
    return text, used_ocr

def process_docx(docx_path: str) -> str:
    """Extract text from a Word document."""
    doc = Document(docx_path)
    return "\n\n".join([para.text for para in doc.paragraphs])

def extract_text_with_ocr(image_path: str) -> str:
    """Extract text from an image using OCR."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        return text
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def ocr_pdf(pdf_path: str) -> str:
    """Process a PDF using OCR."""
    text = ""
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get the page as an image
            pix = page.get_pixmap(alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR
            page_text = pytesseract.image_to_string(img, lang='eng')
            text += page_text + "\n\n"
    
    except Exception as e:
        logger.error(f"OCR PDF error: {e}")
    
    return text

def is_mostly_formatting(text: str) -> bool:
    """Check if the extracted text is mostly formatting elements."""
    # Remove common PDF artifacts and formatting
    cleaned = re.sub(r'\d+\s*of\s*\d+', '', text)
    cleaned = re.sub(r'page\s*\d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)
    
    # If we've lost most of the text, it was probably just formatting
    return len(cleaned.strip()) < 0.5 * len(text.strip())

# Text preprocessing functions
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters that don't add meaning
    text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'`-]', '', text)
    
    return text

# Document chunking for embeddings
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: The text to chunk
        chunk_size: The target size of each chunk
        overlap: How much overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get a chunk of size chunk_size or whatever is left
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text, try to find a good breaking point
        if end < len(text):
            # Look for a period, question mark, or exclamation point followed by a space
            match = re.search(r'[.!?]\s+', text[end-100:end])
            if match:
                end = end - 100 + match.end()
            else:
                # If no sentence break, look for a newline
                match = re.search(r'\n', text[end-50:end])
                if match:
                    end = end - 50 + match.end()
                else:
                    # If no newline, look for a space
                    match = re.search(r'\s', text[end-20:end])
                    if match:
                        end = end - 20 + match.end()
        
        # Add the chunk to our list
        chunks.append(text[start:end])
        
        # Move the start position for the next chunk, considering overlap
        start = end - overlap
    
    return chunks

# Token handling functions (placeholder for future implementation)
def generate_api_token() -> str:
    """Generate a new API token."""
    # Implementation details would depend on your authentication system
    return "placeholder_token"

def validate_token(token: str) -> bool:
    """Validate an API token."""
    # Implementation details would depend on your authentication system
    return True 