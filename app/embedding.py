# app/embedding.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class InLegalBERTEmbedder:
    def __init__(self, model_name='law-ai/InLegalBERT', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """Returns a dense vector for a single string using mean pooling."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state  # shape: (1, 512, hidden_size)
        attention_mask = inputs['attention_mask']

        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        embedding = sum_embeddings / sum_mask

        return embedding.squeeze().cpu().numpy()
