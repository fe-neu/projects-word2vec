from pathlib import Path
import numpy as np
import json


class EmbeddingRetriever:
    """
    Utility class for loading trained word embeddings and retrieving
    similar words based on distance or similarity metrics.

    The retriever loads a vocabulary mapping and embedding matrix from
    a completed training run and provides methods to query nearest
    neighbors using different similarity measures.
    """

    def __init__(
        self,
        run_dir: Path,
        run_name: str
    ):
        """
        Initialize the embedding retriever from a saved training run.

        This loads the vocabulary-to-index mapping and the input embedding
        matrix produced during training.

        :param run_dir: Base directory containing all runs.
        :type run_dir: Path
        :param run_name: Name of the specific run to load.
        :type run_name: str
        """
        with open(
            run_dir / run_name / "vocabulary_map.json",
            encoding="utf-8"
        ) as f:
            self.vocabulary_map = json.load(f)

        self.id_to_word = {v: k for k, v in self.vocabulary_map.items()}

        self.embeddings = np.load(
            run_dir / run_name / "input_weights.npy"
        )

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding vector for a given word.

        :param word: Word whose embedding should be returned.
        :type word: str
        :return: Embedding vector corresponding to the given word.
        :rtype: np.ndarray
        """
        assert word in self.vocabulary_map.keys()
        idx = self.vocabulary_map.get(word)
        return self.embeddings[idx]

    def get_neighbours(
        self,
        word: str,
        method: str
    ) -> list[tuple[str, np.float64]]:
        """
        Retrieve neighboring words for a given query word.

        The query word itself is excluded from the returned results.
        Depending on the selected method, neighbors are ordered by
        ascending distance or descending similarity.

        :param word: Query word from the vocabulary.
        :type word: str
        :param method: Similarity metric to use. Supported values are
                       ``"euclidean_distance"`` and ``"cosine_similarity"``.
        :type method: str
        :return: List of tuples containing neighboring words and their
                 corresponding distance or similarity scores.
        :rtype: list[tuple[str, np.float64]]
        """
        methods = {
            "euclidean_distance": self._euclidean_distance,
            "cosine_similarity": self._cosine_similarity
        }

        assert method in methods.keys()

        query_id = self.vocabulary_map.get(word)
        query_embedding = self.get_embedding(word)

        scores = methods.get(method)(
            query_embedding,
            self.embeddings
        )

        if method == "cosine_similarity":
            score_idx = np.argsort(scores)[::-1]
        else:
            score_idx = np.argsort(scores)

        score_idx = np.delete(
            score_idx,
            np.where(score_idx == query_id)
        )

        return [
            (self.id_to_word.get(idx), scores[idx])
            for idx in score_idx
        ]

    def get_similar_words(
        self,
        query_embedding: str,
        method: str
    ) -> list[tuple[str, np.float64]]:
        """
        Retrieve similar words for an arbitrary query embedding.

        Unlike :meth:`get_neighbours`, this method does not assume that
        the query vector corresponds to a word in the vocabulary.

        :param query_embedding: Query embedding vector.
        :type query_embedding: np.ndarray
        :param method: Similarity metric to use. Supported values are
                       ``"euclidean_distance"`` and ``"cosine_similarity"``.
        :type method: str
        :return: List of tuples containing words and their corresponding
                 distance or similarity scores.
        :rtype: list[tuple[str, np.float64]]
        """
        methods = {
            "euclidean_distance": self._euclidean_distance,
            "cosine_similarity": self._cosine_similarity
        }

        assert method in methods.keys()

        scores = methods.get(method)(
            query_embedding,
            self.embeddings
        )

        if method == "cosine_similarity":
            score_idx = np.argsort(scores)[::-1]
        else:
            score_idx = np.argsort(scores)

        return [
            (self.id_to_word.get(idx), scores[idx])
            for idx in score_idx
        ]

    def _cosine_similarity(
        self,
        q_vec: np.ndarray,
        E: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and all embeddings.

        :param q_vec: Query embedding vector.
        :type q_vec: np.ndarray
        :param E: Matrix of embedding vectors.
        :type E: np.ndarray
        :return: Array of cosine similarity scores.
        :rtype: np.ndarray
        """
        q_norm = np.linalg.norm(q_vec) + 1e-8
        E_norms = np.linalg.norm(E, axis=1) + 1e-8
        sims = (E @ q_vec) / (E_norms * q_norm)
        return sims

    def _euclidean_distance(
        self,
        q_vec: np.ndarray,
        E: np.ndarray
    ) -> np.ndarray:
        """
        Compute Euclidean distance between a query vector and all embeddings.

        :param q_vec: Query embedding vector.
        :type q_vec: np.ndarray
        :param E: Matrix of embedding vectors.
        :type E: np.ndarray
        :return: Array of Euclidean distances.
        :rtype: np.ndarray
        """
        distances = np.linalg.norm(E - q_vec, axis=1)
        return distances
