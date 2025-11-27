import pandas as pd

def find_imdb_duplicates(csv_path):
    df = pd.read_csv(csv_path)

    if "imdb_id" not in df.columns:
        raise ValueError("La colonne 'imdb_id' n'existe pas dans le fichier.")

    # Doublons strictement dans la colonne 'imdb_id'
    duplicates = df[df["imdb_id"].duplicated(keep=False)]

    if duplicates.empty:
        print("Aucun doublon dans la colonne 'imdb_id'.")
        return

    print("Doublons détectés dans la colonne 'imdb_id':")
    for imdb in duplicates["imdb_id"].unique():
        count = (df["imdb_id"] == imdb).sum()
        print(f"- {imdb}  ({count} occurrences)")

if __name__ == "__main__":
    find_imdb_duplicates("raw/TMDB_movie_dataset_v11_25-11-2025_9_32.csv")
