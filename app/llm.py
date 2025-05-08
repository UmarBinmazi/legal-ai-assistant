# llm.py

import os
import requests
import re
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# Load from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("DEFAULT_MODEL")

# Detect query type
def detect_query_type(query: str) -> str:
    """
    Detect the type of query to determine the appropriate response format.
    
    Returns:
        str: One of "greeting", "conversation", "legal_query", "document_query"
    """
    # Check for greetings
    greeting_patterns = [
        r'^(hi|hello|hey|greetings|howdy)[\s\!\.\?]*$',
        r'^good\s(morning|afternoon|evening|day)[\s\!\.\?]*$',
        r'^(how are you|how\'s it going|how have you been|what\'s up)[\s\!\.\?]*$'
    ]
    
    for pattern in greeting_patterns:
        if re.search(pattern, query.lower()):
            return "greeting"
    
    # Check for legal specific query indicators
    legal_indicators = [
        r'case\s+(number|no|id)\s*\:?\s*([a-zA-Z0-9\-\.\/]+)',
        r'section\s+(\d+)',
        r'article\s+(\d+)',
        r'legal\s+(precedent|case|judgment|principle)',
        r'court\s+(ruling|decision|order|judgment)',
        r'supreme\s+court',
        r'high\s+court',
        r'vs\.?|versus',
        r'plaintiff|defendant|petitioner|respondent',
        r'bail|appeal|writ|petition',
        r'\d{4}\s+\(?[a-zA-Z]+\)?\s+\d+'  # Case citation format
    ]
    
    for indicator in legal_indicators:
        if re.search(indicator, query, re.IGNORECASE):
            return "legal_query"
            
    # Default to conversational for anything else
    return "conversation"

# Format the prompt using retrieved legal documents
def format_prompt(query: str, documents: List[Dict]) -> Tuple[str, bool]:
    # Detect query type
    query_type = detect_query_type(query)
    use_structured_format = False
    
    # Check if we're querying a specific document
    is_document_query = any(doc.get("is_document_context") for doc in documents)
    
    # Handle greetings and casual conversation
    if query_type == "greeting":
        prompt = f"""You are a friendly legal assistant. Respond naturally to this greeting:

Query: {query}

Keep your response brief, warm and offer to help with legal questions."""
        return prompt, use_structured_format
    
    if is_document_query:
        # Format for specific document query
        doc = documents[0]  # Should be only one document
        context = f"""Document: {doc.get('filename', 'Unknown')}
Case Number: {doc.get('case_number', 'N/A')}
Year: {doc.get('year', 'N/A')}

Content:
{doc.get('text', '')}"""

        prompt = f"""You are a legal assistant analyzing a specific document. Answer the question based on the content of this document.

Document Context:
{context}

Question: {query}

If the answer cannot be found in the document, clearly state that. Respond in a natural, conversational manner unless the query is specifically asking for legal analysis."""
        return prompt, query_type == "legal_query"
    
    # General conversation or legal query
    if not documents:
        if query_type == "legal_query":
            prompt = f"""You are a legal assistant. No relevant legal cases were found for this query.

Question: {query}

Provide a helpful response but explain that you don't have specific case information for this query. Offer suggestions for how the user might rephrase their question."""
            return prompt, False
        else:
            prompt = f"""You are a helpful and conversational legal assistant. The user has asked:

{query}

Respond naturally and conversationally. You don't need to format your answer in any particular way."""
            return prompt, False
    
    # We have documents and it's either conversational or legal
    # Filter out sample entries when displaying sources unless they're actually relevant
    filtered_docs = [
        doc for doc in documents 
        if doc.get('source') != 'sample' or doc.get('distance', float('inf')) < 80.0
    ]
    
    # If we filtered out all docs, but had samples, add a note
    had_samples = any(doc.get('source') == 'sample' for doc in documents)
    if not filtered_docs and had_samples:
        context = "No highly relevant cases were found in the database."
    else:
        # Format the context from the retrieved documents
        context = "\n\n".join(
            [f"Case Number: {doc.get('case_number', 'N/A')}\nYear: {doc.get('year', 'N/A')}\nRelevance: {doc.get('distance', 'N/A')}\n\nContent: {doc.get('text', '')}" 
            for doc in filtered_docs]
        )

    # For legal queries, use structure. For conversation, be more natural.
    if query_type == "legal_query":
        prompt = f"""You are a legal assistant. The following case documents might be relevant to the question.

Context:
{context}

Question: {query}

Respond with a helpful answer that references the relevant cases if they're useful. If the query is asking about specific cases or legal principles, structure your answer with relevant case numbers and citations."""
        use_structured_format = True
    else:
        prompt = f"""You are a conversational legal assistant. The following information might be helpful for answering the question, but don't explicitly mention it unless relevant.

Background:
{context}

Question: {query}

Respond naturally and conversationally. Avoid formal structure unless the question clearly calls for legal analysis."""
    
    return prompt, use_structured_format

# Call LLM API with formatted prompt
def ask_llm(query: str, documents: List[Dict]) -> str:
    prompt, use_structured_format = format_prompt(query, documents)
    
    system_message = "You are a helpful AI legal assistant."
    if use_structured_format:
        system_message += " Provide your response in a structured format with clear sections when appropriate."
    else:
        system_message += " Respond naturally and conversationally."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 700
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            # Fallback for API errors
            return f"I'm having trouble connecting to my knowledge base. Please try again later. Error: {response.status_code}"
    except Exception as e:
        return f"I encountered an error processing your request: {str(e)}"
