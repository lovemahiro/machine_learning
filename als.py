import numpy as np


FACTORS = 24
ITERATIONS = 12
ALPHA = 20.0
REGULARIZATION = 0.5
SEED = 42


def fit_factors(positive_indices: list[np.ndarray], fixed: np.ndarray) -> np.ndarray:
    """Solve the weighted least-squares subproblem with the other side fixed."""
    dimensions = fixed.shape[1]
    shared = fixed.T @ fixed + REGULARIZATION * np.eye(dimensions)
    fitted = np.zeros((len(positive_indices), dimensions))
    for row, indices in enumerate(positive_indices):
        if not len(indices):
            continue
        positives = fixed[indices]
        # Every pair has base confidence 1, and positive pairs get ALPHA more.
        normal_matrix = shared + ALPHA * (positives.T @ positives)
        target = (1 + ALPHA) * positives.sum(axis=0)
        fitted[row] = np.linalg.solve(normal_matrix, target)
    return fitted


def train_als(likes: np.ndarray) -> np.ndarray:
    users_liked = [np.flatnonzero(row) for row in likes]
    movie_fans = [np.flatnonzero(column) for column in likes.T]
    rng = np.random.default_rng(SEED)
    movie_factors = rng.normal(0, 0.01, size=(likes.shape[1], FACTORS))
    for _ in range(ITERATIONS):
        user_factors = fit_factors(users_liked, movie_factors)
        movie_factors = fit_factors(movie_fans, user_factors)
    return movie_factors


def predict_als(likes: np.ndarray, movie_factors: np.ndarray) -> np.ndarray:
    users_liked = [np.flatnonzero(row) for row in likes]
    user_factors = fit_factors(users_liked, movie_factors)
    return user_factors @ movie_factors.T
