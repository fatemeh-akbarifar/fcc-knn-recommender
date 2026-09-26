"""Item-based collaborative filtering with cosine-distance nearest neighbors."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from data_utils import download, extract_zip

DEFAULT_BOOK = "Where the Heart Is (Oprah's Book Club (Paperback))"


class BookRecommender:
    def fit(self, books, ratings, min_user_ratings=200, min_book_ratings=100):
        if min_user_ratings < 1 or min_book_ratings < 1:
            raise ValueError("Rating thresholds must be positive")
        user_counts = ratings.user.value_counts()
        book_counts = ratings.isbn.value_counts()
        filtered = ratings[
            ratings.user.isin(user_counts[user_counts >= min_user_ratings].index)
            & ratings.isbn.isin(book_counts[book_counts >= min_book_ratings].index)
        ]
        merged = filtered.merge(books, on="isbn").drop_duplicates()
        matrix = merged.pivot_table(index="title", columns="user", values="rating").fillna(0)
        # A zero vector has no usable cosine similarity.
        matrix = matrix.loc[(matrix != 0).any(axis=1)].sort_index()
        if len(matrix) < 2:
            raise ValueError("At least two rated books must remain after filtering")
        self.titles = matrix.index
        self.matrix = csr_matrix(matrix.to_numpy(dtype=np.float32))
        self.model = NearestNeighbors(metric="cosine", algorithm="brute").fit(self.matrix)
        return self

    def get_recommends(self, book, count=5):
        if not hasattr(self, "model"):
            raise ValueError("Fit the recommender before requesting recommendations")
        if book not in self.titles:
            raise ValueError(f"Book {book!r} is not in the filtered catalogue")
        if count < 1:
            raise ValueError("count must be positive")
        index = self.titles.get_loc(book)
        # Query all eligible titles to make distance ties deterministic.
        distances, indices = self.model.kneighbors(self.matrix[index], n_neighbors=len(self.titles))
        neighbours = [(str(self.titles[i]), float(d)) for i, d in zip(indices[0], distances[0]) if i != index]
        neighbours.sort(key=lambda item: (item[1], item[0]))
        return [book, [[title, distance] for title, distance in neighbours[:count]]]


def load_data(directory):
    directory = Path(directory)
    if not all((directory / f).exists() for f in ["BX-Books.csv", "BX-Book-Ratings.csv"]):
        archive = download("https://cdn.freecodecamp.org/project-data/books/book-crossings.zip", directory / "book-crossings.zip")
        extract_zip(archive, directory)
    books = pd.read_csv(directory / "BX-Books.csv", sep=";", encoding="ISO-8859-1", usecols=[0, 1, 2], dtype=str)
    books.columns = ["isbn", "title", "author"]
    ratings = pd.read_csv(directory / "BX-Book-Ratings.csv", sep=";", encoding="ISO-8859-1", dtype={"ISBN": str})
    ratings.columns = ["user", "isbn", "rating"]
    return books, ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--book", default=DEFAULT_BOOK)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/recommendations.json"))
    args = parser.parse_args()
    recommender = BookRecommender().fit(*load_data(args.data_dir))
    result = {"recommendations": recommender.get_recommends(args.book, args.count),
              "books": len(recommender.titles), "users": recommender.matrix.shape[1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
