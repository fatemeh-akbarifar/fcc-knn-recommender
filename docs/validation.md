# Reproducibility report

Verified on 26 September 2026 with Python 3.9.13 on macOS ARM64 (CPU).

## Dataset run

```bash
python book_recommender.py
```

The validation run used the same public data cached under `/private/tmp/fcc-data`; output paths were supplied explicitly where needed. Defaults use repository-local `data/` and `artifacts/`.

Real dataset: 673 eligible books and 888 users. Five model-derived recommendations were returned, with no answer substitution.

[Machine-readable results](results.json). These are newly measured results, not historical notebook output.

## Behavioral checks

`python -m pytest -q` passed 5 tests. Tests cover archive traversal rejection and cosine distances, self-exclusion, unknown titles, and filtering edge cases.

## Environment

Core versions: NumPy 1.23.5, pandas 2.2.3, scikit-learn 1.6.1 and SciPy 1.13.1. Dependency pins are in `requirements.txt`. No GPU was used. Results can vary across platforms; the neural-network seed is 42.

The GitHub Actions workflow is configured separately; local success does not itself establish a successful hosted workflow run.

## Clean-environment verification

The pinned requirements installed successfully into a new virtual environment with no inherited site packages. `pip check` found no broken requirements, and this project's test suite passed in that environment. Python 3.11 hosted CI is tracked separately.

## Hosted CI

[Python 3.11 GitHub Actions run](https://github.com/fatemeh-akbarifar/fcc-knn-recommender/actions/runs/36242602253) completed successfully for the published implementation.

## First-run downloads

The CDN rejected Python's default HTTP user agent (403). The downloader now supplies an explicit client header; a fresh real HTTPS download matched the recorded checksum. A focused test verifies that header, complete file writing, and cache reuse.
