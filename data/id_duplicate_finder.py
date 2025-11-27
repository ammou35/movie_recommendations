import pandas as pd
import math

def find_id_duplicates(csv_path):
    df = pd.read_csv(csv_path)

    if "id" not in df.columns:
        raise ValueError("La colonne 'id' n'existe pas dans le fichier.")

    # Trouver les lignes où 'id' apparait plus d'une fois
    duplicates = df[df["id"].duplicated(keep=False)]

    if duplicates.empty:
        print("Aucun doublon dans la colonne 'id'.")
        return

    print("Doublons détectés dans la colonne 'id' :")

    for value in duplicates["id"].unique():
        # Gestion propre des NaN
        if isinstance(value, float) and math.isnan(value):
            count = df["id"].isna().sum()
            label = "NaN"
        else:
            count = (df["id"] == value).sum()
            label = value

        print(f"- {label} ({count} occurrences)")

if __name__ == "__main__":
    find_id_duplicates("raw/TMDB_movie_dataset_v11_25-11-2025_9_32.csv")
