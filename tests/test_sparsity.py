import unittest

import numpy as np

from experiment import evaluate, evaluation_window
from models import Recommender
from sparsity import SparsityData, visible_profiles


class SparseHistoryTest(unittest.TestCase):
    def test_visible_profiles_expose_only_the_requested_prefix(self):
        history = np.array([[5, 2, 4, 0]], dtype=np.uint8)
        model = Recommender(np.array([[4, 4, 4, 4]], dtype=np.uint8))
        window = evaluation_window(history, np.array([[0, 0, 0, 5]]), model.catalog)
        data = SparsityData(model, window, np.array([[2, 0]]), 5, [])

        np.testing.assert_array_equal(visible_profiles(data, 0), [[0, 0, 0, 0]])
        np.testing.assert_array_equal(visible_profiles(data, 1), [[0, 0, 4, 0]])
        np.testing.assert_array_equal(visible_profiles(data, 2), [[4, 0, 4, 0]])
        np.testing.assert_array_equal(window.profiles, history)

    def test_hidden_but_previously_rated_movie_stays_excluded(self):
        model = Recommender(np.array([[5, 5, 0], [0, 5, 0], [5, 0, 5]]))
        window = evaluation_window(np.array([[5, 5, 0]]), np.array([[0, 0, 5]]), model.catalog)
        result = evaluate(model, window, "itemcf", 1, visible_profiles=np.array([[4, 0, 0]]))

        np.testing.assert_array_equal(result.ranked, [[2]])
        np.testing.assert_array_equal(result.recall, [1.0])
        np.testing.assert_array_equal(result.ndcg, [1.0])


if __name__ == "__main__":
    unittest.main()
