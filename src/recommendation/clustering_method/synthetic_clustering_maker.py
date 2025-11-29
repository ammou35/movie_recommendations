"""
Create synthetic clustering outputs from a TMDB-like movie dataset.

The script samples 200 movies from the provided CSV, normalizes selected
numerical features, assigns synthetic cluster labels, and stores both the
assignments and cluster centroids as parquet files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

REQUIRED_COLUMNS: Sequence[str] = (
    "id",
    "title",
    "vote_average",
    "vote_count",
    "release_date",
    "revenue",
    "runtime",
    "adult",
    "budget",
)

NUMERIC_FEATURES: Sequence[str] = (
    "vote_average",
    "vote_count",
    "release_date",
    "revenue",
    "runtime",
    "adult",
    "budget",
)

CLUSTER_COUNT = 5
SAMPLE_SIZE = 200
RANDOM_SEED = 0

INPUT_CSV_PATH = REPO_ROOT / "data" / "raw" / "TMDB_movie_dataset_v11_25-11-2025_9_32.csv"
ASSIGNMENTS_PATH = SCRIPT_DIR / "assignments.parquet"
CENTROIDS_PATH = SCRIPT_DIR / "centroids.parquet"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    last_error: Exception | None = None
    for encoding in (None, "latin1"):
        for engine in ("c", "python"):
            kwargs = {}
            if encoding is not None:
                kwargs["encoding"] = encoding
            if engine == "python":
                kwargs["engine"] = "python"
            try:
                df = pd.read_csv(path, **kwargs)
                _ensure_required_columns(df)
                return df
            except (UnicodeDecodeError, pd.errors.ParserError) as err:
                last_error = err

    try:
        df = pd.read_parquet(path)
        _ensure_required_columns(df)
        return df
    except Exception as parquet_error:
        raise RuntimeError(f"Unable to read dataset at {path}") from (
            parquet_error if last_error is None else parquet_error
        )


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def sample_subset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Dataset contains only {len(df)} rows, but {SAMPLE_SIZE} are required."
        )
    sample = df.sample(n=SAMPLE_SIZE, random_state=seed).reset_index(drop=True)
    return sample.copy()


def encode_adult(series: pd.Series) -> pd.Series:
    def _to_bool(value: object) -> int:
        if pd.isna(value):
            return 0
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return 1 if value != 0 else 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y"}:
                return 1
            if lowered in {"0", "false", "f", "no", "n"}:
                return 0
        raise ValueError(f"Cannot interpret 'adult' value: {value!r}")

    return series.map(_to_bool).astype(np.int8)


def convert_release_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    base_date = parsed.dropna().min()
    if pd.isna(base_date):
        base_date = pd.Timestamp(0)
    days_since_min = (
        (parsed.fillna(base_date) - base_date).dt.days.fillna(0).astype(np.int64)
    )
    return days_since_min


def normalize_features(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        values = pd.to_numeric(normalized[column], errors="coerce").fillna(0)

        min_val = values.min()
        max_val = values.max()
        if min_val == max_val:
            normalized[column] = np.zeros(len(values), dtype=np.float64)
            continue

        normalized[column] = ((values - min_val) / (max_val - min_val)).astype(
            np.float64
        )
    return normalized


def assign_clusters(size: int) -> pd.Series:
    labels = np.arange(size, dtype=np.int64) % CLUSTER_COUNT
    return pd.Series(labels, name="cluster_label")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    df = load_dataset(INPUT_CSV_PATH)
    subset = sample_subset(df, RANDOM_SEED)

    subset["adult"] = encode_adult(subset["adult"])
    subset["release_date"] = convert_release_date(subset["release_date"])

    subset = normalize_features(subset, NUMERIC_FEATURES)
    subset["cluster_label"] = assign_clusters(len(subset))

    assignments = subset[
        ["id", "title", "cluster_label", *NUMERIC_FEATURES]
    ].copy()
    save_parquet(assignments, ASSIGNMENTS_PATH)

    centroids = (
        assignments.groupby("cluster_label")[list(NUMERIC_FEATURES)]
        .mean()
        .reset_index()
    )
    save_parquet(centroids, CENTROIDS_PATH)


if __name__ == "__main__":
    main()
