from embedding import InLegalBERTEmbedder
from retrieval import LegalCaseRetriever

# Create embedder instance
embedder = InLegalBERTEmbedder()
retriever = LegalCaseRetriever()

# Index documents
texts = ["Case about Article 21", "High court bail case", "Criminal appeal dismissed"]
embeddings = [embedder.embed_text(text) for text in texts]
metadata = [
    {"case_number": "A21-001", "year": 2018},
    {"case_number": "HC-BAIL-002", "year": 2020},
    {"case_number": "CRIM-003", "year": 2022},
]
retriever.add_to_index(embeddings, metadata)

# Save index and metadata
retriever.save_index()

# Later...
retriever.load_index()
query = "What are some bail-related judgments?"
query_vec = embedder.embed_text(query)
results = retriever.search(query_vec)

for dist, meta in results:
    print(f"{dist:.2f} | {meta}")
