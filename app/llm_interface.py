# llm_interface.py

import os
import requests
from dotenv import load_dotenv
from typing import List, Dict

# Load from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("DEFAULT_MODEL")

# Format the prompt using retrieved legal documents
def format_prompt(query: str, documents: List[Dict]) -> str:
    context = "\n\n".join(
        [f"Case Number: {doc.get('case_number', 'N/A')}\nYear: {doc.get('year', 'N/A')}\nSummary: {doc['text']}" for doc in documents]
    )
    prompt = f"""You are a legal assistant. Use the following case documents to answer the question.

Context:
{context}

Question: {query}

Respond in the following format:
- Similar Case Number:
- Year:
- Bench:
- Judgment Summary:
- Relevance:
"""
    return prompt

# Call Groq API with formatted prompt
def ask_llm(query: str, documents: List[Dict]) -> str:
    prompt = format_prompt(query, documents)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI legal assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 700
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Groq API Error: {response.status_code} - {response.text}")
