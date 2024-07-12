import faiss
import numpy as np
from fastapi import HTTPException, status

from config import CFG


class DataBase:
    def __init__(
        self,
        top_k=5,  # nearest neighbors
        threshold=0.9,
    ) -> None:
        self.top_k = top_k
        self.index = faiss.IndexFlatIP(CFG.emb_dim)
        self.threshold = threshold

    def add(self, embedding: np.array) -> None:
        # faiss.normalize_L2(np.expand_dims(embedding, axis=0))
        embedding = embedding / np.linalg.norm(embedding)
        self.index.add(embedding)

    def check(self, embedding: np.array):
        # faiss.normalize_L2(np.expand_dims(embedding, axis=0))
        embedding = embedding / np.linalg.norm(embedding)
        distances, _ = self.index.search(embedding, self.top_k)
        # print(np.round(distances[0], 2))
        for d in distances[0]:
            if self.threshold < d:
                return True
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    def remove(self):
        pass
