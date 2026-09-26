# Engineering notes

## Original project

Book recommendations with k-nearest neighbors by Fatemeh Akbarifar, developed using a freeCodeCamp starter. The existing Git history and original Colab source establish the project provenance.

## Reproducibility work (September 2026)

- Replace notebook-only shell/magic commands in Python entry points with explicit download helpers and command-line interfaces.
- Pin dependencies and add focused tests plus a GitHub Actions workflow.
- Make imports free of downloads and training side effects.
- Save measured outputs separately from source code.
- Provide a notebook entry point that runs the same Python implementation.

## Method-specific changes

1. Load the freeCodeCamp Book-Crossing CSV files.
2. Retain users with at least 200 ratings and ISBNs with at least 100 ratings, counting both thresholds on the original ratings table.
3. Join titles, average duplicate title/user entries, fill missing ratings with zero, and convert to CSR format. Remove all-zero book vectors.
4. Fit brute-force nearest neighbors with cosine distance, `1 − cosine_similarity`.
5. Exclude the query by its actual index and return up to five closest titles, **closest first**, with full-precision distances. Title order breaks distance ties.

No query has a hard-coded answer. A distance is a similarity measure, not a probability or a recommendation-quality score.

## Reading the evidence

The validation report describes newly executed runs. It does not retroactively claim that historical notebook outputs used the corrected evaluation pipeline. Unit tests verify behavior; they are not model-quality benchmarks.

## Data provenance

[Dataset checksums](data-manifest.json) identify the exact downloaded inputs used for validation. These are hashes of public dataset files, not private Drive content.
