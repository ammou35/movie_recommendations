import pandas as pd

def find_title_duplicates(csv_path):
    df = pd.read_csv(csv_path)

    if "title" not in df.columns:
        raise ValueError("La colonne 'title' n'existe pas dans le fichier.")

    # Doublons strictement dans la colonne 'title'
    duplicates = df[df["title"].duplicated(keep=False)]

    if duplicates.empty:
        print("Aucun doublon dans la colonne 'title'.")
        return

    print("Titres dupliqués (colonne 'title' uniquement):")
    for title in duplicates["title"].unique():
        count = (df["title"] == title).sum()
        print(f"- {title}  ({count} occurrences)")

if __name__ == "__main__":
    find_title_duplicates("raw/TMDB_movie_dataset_v11_25-11-2025_9_32.csv")
