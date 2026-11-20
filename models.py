from functools import cached_property

import numpy as np

from als import predict_als, train_als
from data import LIKE_THRESHOLD


MODEL_LABELS = {
    "popular": "Popularity",
    "itemcf": "ItemCF",
    "shrinkage": "ItemCF + Shrinkage",
    "als": "Implicit ALS",
}


class Recommender:
    def __init__(self, history: np.ndarray) -> None:
        self.likes = (history >= LIKE_THRESHOLD).astype(np.float32)
        self.popularity = self.likes.sum(axis=0)
        self.catalog = np.any(history != 0, axis=0)
        # Stable sort gives movie ID order when popularity is tied.
        self.popular_order = np.argsort(-self.popularity, kind="stable")

    @cached_property
    def common_likes(self) -> np.ndarray:
        return self.likes.T @ self.likes

    @cached_property
    def cosine_similarity(self) -> np.ndarray:
        norms = np.sqrt(self.popularity)
        denominators = np.outer(norms, norms)
        similarity = np.divide(
            self.common_likes,
            denominators,
            out=np.zeros_like(self.common_likes),
            where=denominators > 0,
        )
        np.fill_diagonal(similarity, 0)
        return similarity

    def similarity(self, model: str, shrinkage: float = 10.0) -> np.ndarray | None:
        if model == "popular":
            return None
        if model == "itemcf":
            return self.cosine_similarity
        weight = self.common_likes / (self.common_likes + shrinkage)
        return self.cosine_similarity * weight

    @cached_property
    def movie_factors(self) -> np.ndarray:
        return train_als(self.likes)

    def score_profiles(self, profiles: np.ndarray, model: str, shrinkage: float = 10.0) -> np.ndarray:
        if model == "popular":
            return np.broadcast_to(self.popularity, profiles.shape)
        liked = profiles >= LIKE_THRESHOLD
        if model == "als":
            return predict_als(liked, self.movie_factors)
        return liked.astype(np.float32) @ self.similarity(model, shrinkage)

    def rank(self, profiles: np.ndarray, scores: np.ndarray, top_k: int) -> np.ndarray:
        """Return zero-based movie indices, using -1 for missing recommendations."""
        eligible = self.catalog & (profiles == 0)
        order = self.popular_order
        sort_values = np.where(eligible[:, order], -scores[:, order], np.inf)

        # Keep popularity order as the tie-breaker, including zero-score fallback.
        ranked = order[np.argsort(sort_values, axis=1, kind="stable")[:, :top_k]]
        return np.where(np.take_along_axis(eligible, ranked, axis=1), ranked, -1)
