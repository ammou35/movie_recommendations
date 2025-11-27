import pandas as pd

def analyse_tmdb(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    required_cols = ["id", "imdb_id", "title", "budget", "revenue"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV: '{col}'")

    nb_rows = len(df)

    # Doublons
    dup_ids_mask = df["id"].duplicated(keep=False)

    # On ignore les NaN pour compter les imdb_id / titles dupliqués
    dup_imdb_mask = df["imdb_id"].duplicated(keep=False) & df["imdb_id"].notna()
    dup_title_mask = df["title"].duplicated(keep=False) & df["title"].notna()

    nb_ids_dupliques = df.loc[dup_ids_mask, "id"].nunique()
    nb_imdb_dupliques = df.loc[dup_imdb_mask, "imdb_id"].nunique()
    nb_titles_dupliques = df.loc[dup_title_mask, "title"].nunique()

    # Budgets / revenus à 0
    budget_zero_mask = df["budget"] == 0
    revenue_zero_mask = df["revenue"] == 0
    nb_budget_zero = budget_zero_mask.sum()
    nb_revenue_zero = revenue_zero_mask.sum()

    # Films avec au moins un NaN
    row_has_nan_mask = df.isna().any(axis=1)
    nb_films_avec_nan = row_has_nan_mask.sum()

    # Films « propres »:
    # - id non dupliqué
    # - imdb_id non dupliqué et non NaN
    # - title non dupliqué et non NaN
    # - budget != 0
    # - revenue != 0
    # - aucune valeur NaN dans la ligne
    bad_mask = (
        dup_ids_mask
        | dup_imdb_mask
        | dup_title_mask
        | budget_zero_mask
        | revenue_zero_mask
        | row_has_nan_mask
    )
    good_mask = ~bad_mask
    nb_films_propres = good_mask.sum()

    print(f"Nombre total de lignes (films): {nb_rows}\n")

    print(f"Nombre d'ID distincts dupliqués         : {nb_ids_dupliques}")
    print(f"Nombre d'IMDB_ID distincts dupliqués    : {nb_imdb_dupliques}")
    print(f"Nombre de TITRES distincts dupliqués    : {nb_titles_dupliques}\n")

    print(f"Nombre de films avec budget = 0         : {nb_budget_zero}")
    print(f"Nombre de films avec revenue = 0        : {nb_revenue_zero}\n")

    print(f"Nombre de films avec au moins un NaN    : {nb_films_avec_nan}\n")

    print("Nombre de films utilisables sans traitement (toutes données raisonnables):")
    print(f"  → {nb_films_propres}")

if __name__ == "__main__":
    analyse_tmdb("raw/TMDB_movie_dataset_v11_25-11-2025_9_32.csv")
