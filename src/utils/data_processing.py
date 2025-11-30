import numpy as np
import pandas as pd
from pathlib import Path

# Charger les données relatifs aux films à partir du fichier CSV
def load_movies(csv_path):
    df = pd.read_csv(csv_path, na_values=["", " "], quotechar='"')

    # Transformation de la date de sortie en année de sortie
    df["date_x"] = df["date_x"].astype(str).str.strip()
    df["date_x"] = pd.to_datetime(df["date_x"], format="%m/%d/%Y", errors="coerce")
    idx = df.columns.get_loc("date_x")
    df.insert(idx, "year", df["date_x"].dt.year)

    # Colonnes à supprimer car jamais traitées
    df = df.drop(columns=["date_x", "status", "orig_title"], errors="ignore")

    # Remplacer les valeurs 0 par NaN pour les colonnes avec des valeurs aberrantes
    for feature in ["score", "revenue"]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df.loc[df[feature] == 0, feature] = np.nan
        mean_val = df[feature].mean(skipna=True)
        df[feature] = df[feature].fillna(mean_val)

    # Sauvegarder le DataFrame nettoyé au format .parquet
    base_parquet_path = Path("data/processed/clean_data.parquet")
    base_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_parquet_path, index=False)


def load_for_clustering():
    df = pd.read_parquet("data/processed/clean_data.parquet")
    feature_cols = ["year", "score", "budget_x", "revenue"]
    df = df[feature_cols]
    df.to_parquet("data/processed/clustering_data.parquet", index=False)


def load_for_embed():
    df = pd.read_parquet("data/processed/clean_data.parquet")
    feature_cols = ["names", "genre", "overview", "crew", "orig_lang", "country"]
    df = df[feature_cols]
    df.to_parquet("data/processed/embed_data.parquet", index=False)


def count_nans(df):
    nan_counts = df.isna().sum()
    print("\nNombre de NaN par feature :\n")
    print(nan_counts)
    return nan_counts


load_movies("data/raw/imdb_movies.csv")
load_for_clustering()
load_for_embed()
