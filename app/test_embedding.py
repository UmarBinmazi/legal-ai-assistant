# test_embedding.py (just for testing)
from embedding import InLegalBERTEmbedder

embedder = InLegalBERTEmbedder()
vector = embedder.embed_text("The Supreme Court ruled in favor of the petitioner.")
print(vector.shape)  # Expect something like (768,)
