import numpy as np
import pandas as pd

# Charger les données relatifs aux films à partir du fichier CSV
def load_movies(csv_path):
    df = pd.read_csv(csv_path, na_values=["", " "])

    # Transformation des colonnes nécessaires
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["adult"] = df["adult"].astype(str).str.lower() == "true"
    df["adult"] = df["adult"].astype(int)

    # Colonnes à supprimer
    df = df.drop(columns=["status", "release_date", "backdrop_path", "homepage", "imdb_id", "popularity", "poster_path"], errors="ignore")

    # Remplacer les valeurs 0 par NaN pour les colonnes avec des valeurs aberrantes
    for feature in ["vote_average", "vote_count", "runtime", "budget", "revenue"]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df.loc[df[feature] == 0, feature] = np.nan

    # Filtrer les films où il manque trop d'informations
    required = ["vote_average", "vote_count", "revenue", "runtime", "budget", "release_year"]
    df = df.dropna(subset=required, how="all")
    df = df.dropna(subset=["title"])

    # Remplacer les NaN des caractéristiques pertinentes par des moyennes
    for feature in required:
        mean_val = df[feature].mean(skipna=True)
        df[feature] = df[feature].fillna(mean_val)
    
    # Remplacer les NaN de la colonne "adult" par True (just in case !)
    df["adult"] = df[feature].fillna(True).astype(bool)
            
    # Filtrer les films avec le même id en gardant celui avec le moins de valeurs manquantes
    missing_count = df.isna().sum(axis=1)
    df = df.loc[missing_count.sort_values().index].drop_duplicates(subset="id", keep="first")

    X = df.to_numpy()
    return X


def count_nans(df):
    nan_counts = df.isna().sum()
    print("\nNombre de NaN par feature :\n")
    print(nan_counts)
    return nan_counts


X = load_movies("data/raw/TMDB_movie_dataset_v11_25-11-2025_9_32.csv")
