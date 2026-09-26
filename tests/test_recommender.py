import numpy as np
import pandas as pd
import pytest
from book_recommender import BookRecommender


def fitted():
    books = pd.DataFrame({"isbn": ["a", "b", "c", "d"], "title": ["A", "B", "C", "D"]})
    ratings = pd.DataFrame([(u, b, r) for b, rs in zip("abcd", [[5,0], [5,0], [0,5], [3,4]])
                            for u, r in enumerate(rs)], columns=["user", "isbn", "rating"])
    return BookRecommender().fit(books, ratings, 1, 1)


def test_actual_cosine_distances_and_self_exclusion():
    model = fitted()
    title, neighbours = model.get_recommends("B")
    assert title == "B"
    assert [x[0] for x in neighbours] == ["A", "D", "C"]
    assert np.allclose([x[1] for x in neighbours], [0, .4, 1], atol=1e-6)
    assert len(model.get_recommends("A", 1)[1]) == 1


def test_invalid_query_and_count():
    model = fitted()
    with pytest.raises(ValueError): model.get_recommends("unknown")
    with pytest.raises(ValueError): model.get_recommends("A", 0)
    with pytest.raises(ValueError): BookRecommender().get_recommends("A")


def test_empty_filtered_catalogue():
    books = pd.DataFrame({"isbn": ["a"], "title": ["A"]})
    ratings = pd.DataFrame({"user": [1], "isbn": ["a"], "rating": [5]})
    with pytest.raises(ValueError): BookRecommender().fit(books, ratings)
