from pathlib import Path
import numpy as np
import json

class EmbeddingRetriever:
    def __init__(self,
                 run_dir: Path,
                 run_name: str):
        
        with open(run_dir / run_name / "vocabulary_map.json", encoding="utf-8") as f:
            self.vocabulary_map = json.load(f)

        self.id_to_word = {v: k for k, v in self.vocabulary_map.items()}

        self.embeddings = np.load(run_dir / run_name / "input_weights.npy")
            

    def get_embedding(self, word: str) -> np.ndarray:
        idx = self.vocabulary_map.get(word)
        return self.embeddings[idx]

    def get_neighbours(self, word: str, method: str) -> list[str]:
        methods = {
            "euclidean_distance": self._euclidean_distance,
            "cosine_similarity": self._cosine_similarity
        }

        assert(method in methods.keys())

        query_id = self.vocabulary_map.get(word)
        central_embedding = self.get_embedding(word)

        scores = methods.get(method)(central_embedding, self.embeddings)

        if method == "cosine_similarity":
            score_idx = np.argsort(scores)[::-1]   # descending
        else:
            score_idx = np.argsort(scores)          # ascending
        score_idx = np.delete(score_idx, np.where(score_idx == query_id))

        return [(self.id_to_word.get(idx), scores[idx]) for idx in score_idx]

    def _cosine_similarity(self, q_vec: np.ndarray, E: np.ndarray) -> np.ndarray:
        q_norm = np.linalg.norm(q_vec) + 1e-8
        E_norms = np.linalg.norm(E, axis=1) + 1e-8
        sims = (E @ q_vec) / (E_norms * q_norm)
        return sims
    
    def _euclidean_distance(self, q_vec: np.ndarray, E: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(E - q_vec, axis=1)
        return distances
        