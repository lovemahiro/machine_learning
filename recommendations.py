from dataclasses import dataclass

import numpy as np

from data import LIKE_THRESHOLD, MovieLens
from models import Recommender


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    detail: str


class RecommendationSession:
    def __init__(self, dataset: MovieLens) -> None:
        self.dataset = dataset
        self.history = dataset.rating_matrix(dataset.ratings)
        self.model = Recommender(self.history)

    def matching_movies(self, query: str) -> list[tuple[int, str]]:
        query = query.strip().casefold()
        return [
            (movie_id, title) for movie_id, title in self.dataset.titles.items()
            if query == str(movie_id) or query in title.casefold()
        ]

    def recommend(
        self, *, method: str, top_k: int, shrinkage: float = 10.0,
        user: int | None = None, likes: tuple[int, ...] = (),
    ) -> list[Recommendation]:
        if user is not None:
            if not 1 <= user <= len(self.history) or not self.history[user - 1].any():
                raise ValueError(f"找不到用户 {user}。请输入 1 至 {len(self.history)} 之间的用户编号。")
            profile = self.history[user - 1]
        else:
            unknown = sorted(set(likes) - self.dataset.titles.keys())
            if unknown:
                raise ValueError(f"找不到电影编号 {', '.join(map(str, unknown))}。请按片名查询编号。")
            profile = np.zeros(self.history.shape[1], dtype=self.history.dtype)
            profile[np.array(likes, dtype=int) - 1] = LIKE_THRESHOLD

        profiles = profile[None, :]
        predictions = self.model.score_profiles(profiles, method, shrinkage)
        ranked = self.model.rank(profiles, predictions, top_k)[0]
        similarity = self.model.similarity(method, shrinkage) if method in ("itemcf", "shrinkage") else None
        liked_indices = np.flatnonzero(profile >= LIKE_THRESHOLD)
        recommendations = []
        for index in ranked[ranked >= 0]:
            movie_id = int(index) + 1
            if method == "popular":
                detail = f"共有 {int(self.model.popularity[index])} 条高分评分"
            elif method == "als" and len(liked_indices):
                detail = f"ALS 偏好分数 {predictions[0, index]:.4f}"
            elif similarity is not None:
                weights = similarity[liked_indices, index]
                strongest = np.argsort(-weights, kind="stable")[:2]
                reasons = [self.dataset.titles[int(liked_indices[pos]) + 1] for pos in strongest if weights[pos] > 0]
                detail = f"相似电影 {'、'.join(reasons)}" if reasons else "按热门度补齐"
            else:
                detail = "按热门度补齐"
            recommendations.append(Recommendation(movie_id, self.dataset.titles[movie_id], detail))
        return recommendations
