import torch
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

class SentenceEncoder:
    def __init__(self,):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode(self, sentences):
        with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            embeddings = self.model.encode(sentences)
        return embeddings
    