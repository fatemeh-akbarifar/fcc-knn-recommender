# Book recommendations with k-nearest neighbors

[![Tests](https://github.com/fatemeh-akbarifar/fcc-knn-recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/fatemeh-akbarifar/fcc-knn-recommender/actions/workflows/tests.yml)

An item-based collaborative-filtering project that recommends books from reader-rating patterns. It demonstrates data filtering, a sparse book–user representation, cosine similarity, and reproducible nearest-neighbor retrieval.

## Original work

The original notebook preserves the book/user filtering, rating matrix, and cosine-neighbor implementation. The maintained edition returns computed recommendations throughout; the evidence notes explain how its validation differs from the original saved example.

[Original notebook and evidence](docs/original-work.md). The runnable edition below includes maintenance fixes; new validation numbers are kept separate from historical achievements.

## Run locally

```bash
git clone https://github.com/fatemeh-akbarifar/fcc-knn-recommender.git
cd fcc-knn-recommender
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python book_recommender.py --book "Where the Heart Is (Oprah's Book Club (Paperback))"
```

Use Python **3.9–3.11**; the automated workflow targets Python 3.11. On Windows, activate with `.venv\Scripts\activate`. A first run requires internet access for dependencies and missing datasets. Subsequent runs reuse local data. CPU execution is supported; neural-network training is slower without an accelerator.

Use a Python 3.9–3.11 kernel for the notebook; hosted runtimes with newer Python versions are outside the pinned environment. The verified execution path is the local command line.

Notebook: [book_recommendation.ipynb](book_recommendation.ipynb). [Open in Colab](https://colab.research.google.com/github/fatemeh-akbarifar/fcc-knn-recommender/blob/main/book_recommendation.ipynb).

## Method

```mermaid
flowchart LR
    A[Book and rating CSVs] --> B[Filter active users and rated books]
    B --> C[Book-user matrix]
    C --> D[CSR sparse representation]
    D --> E[Cosine-distance nearest neighbors]
    E --> F[Five similar books]
```

1. Load the freeCodeCamp Book-Crossing CSV files.
2. Retain users with at least 200 ratings and ISBNs with at least 100 ratings, counting both thresholds on the original ratings table.
3. Join titles, average duplicate title/user entries, fill missing ratings with zero, and convert to CSR format. Remove all-zero book vectors.
4. Fit brute-force nearest neighbors with cosine distance, `1 − cosine_similarity`.
5. Exclude the query by its actual index and return up to five closest titles, **closest first**, with full-precision distances. Title order breaks distance ties.

No query has a hard-coded answer. A distance is a similarity measure, not a probability or a recommendation-quality score.

## Outputs

`artifacts/recommendations.json` contains the query, recommendations, distances, and matrix dimensions.

## Maintenance validation (September 2026)

The maintained implementation was checked on the real Book-Crossing data: **673 books**, **888 users**, and five model-derived recommendations per default query.

See [the reproducibility report](docs/validation.md) for measured results, commands, environment, and the limits of validation.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The included GitHub Actions workflow runs focused tests, including saved-model round trips where applicable. It does not retrain the full dataset on every push.

## Limitations

This is a similarity-retrieval demonstration, not a measured ranking benchmark. No precision@k, recall@k, or user study is claimed. Zero-valued ratings and missing ratings share the zero representation; merging by title can conflate editions. The catalogue is filtered, so an unknown or infrequently rated book raises a clear error. Querying all candidates provides deterministic ties for this small catalogue but is not intended for web-scale retrieval.

## Project background and attribution

Developed by **Fatemeh Akbarifar** as part of freeCodeCamp's Machine Learning with Python projects. This repository packages and modernizes the original implementation with reusable Python entry points, dependency pins, tests, and reproducible evaluation. [Engineering notes](docs/engineering.md) distinguish original work from the reproducibility improvements.

- [freeCodeCamp project starter](https://github.com/freeCodeCamp/boilerplate-book-recommendation-engine)
- [Dataset used by the project](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip)

The challenge design and supplied datasets are external resources; their original terms apply. No new dataset ownership or certification claim is made here.
