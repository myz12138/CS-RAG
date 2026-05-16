import os, json, time, torch, numpy as np
from typing import List

def _remote_embed(texts: List[str], endpoint: str, api_key: str, model: str):
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    resp = requests.post(
        endpoint.rstrip('/') + "/v1/embeddings",
        headers=headers,
        json={"model": model, "input": texts}
    )
    resp.raise_for_status()
    data = resp.json()
    embs = [d["embedding"] for d in data["data"]]
    return torch.tensor(embs, dtype=torch.float32)

class EmbeddingClient:
    def __init__(self, local_model: str, device: str="cpu"):
        self.endpoint = os.getenv("EMB_ENDPOINT",'')
        self.api_key  = os.getenv("EMB_API_KEY",'')
        self.model    = os.getenv("EMB_MODEL",)
        self.use_remote = self.endpoint and self.api_key and self.model
        if not self.use_remote:
            from sentence_transformers import SentenceTransformer
            self.local = SentenceTransformer(local_model, device=device)
            self.dim = self.local.get_sentence_embedding_dimension()
        else:
            self.local = None
            self.dim = int(os.getenv("EMB_DIM", "384"))

    def encode(self, texts: List[str]):
        if self.use_remote:
            return _remote_embed(texts, self.endpoint, self.api_key, self.model)
        else:
            v = self.local.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return torch.from_numpy(v).float()

    def encode_one(self, text: str):
        return self.encode([text])[0]
