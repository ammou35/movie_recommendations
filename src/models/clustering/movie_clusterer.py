from typing import Optional

from pandas import DataFrame
from sklearn.cluster import KMeans


class MovieClusterer:
    def __init__(self, X: DataFrame, number_of_clusters: int, random_state: Optional[int] = None):
        self.model = KMeans(n_clusters=number_of_clusters, random_state=random_state)
        self.X = X
        self._feature_names = self.X.columns
        self._cluster_assignation: Optional[DataFrame] = None
        self._cluster_centroids: Optional[DataFrame] = None

    def fit(self):
        labels = self.model.fit_predict(self.X)
        self._cluster_assignation = self._attach_cluster_labels(self.X, labels)
        self._cluster_centroids = self._build_centroids(self.model.cluster_centers_)
        return self

    @property
    def cluster_centroids(self) -> DataFrame:
        if self._cluster_centroids is None:
            raise ValueError("Call fit() before accessing cluster centroids.")
        return self._cluster_centroids

    @property
    def cluster_assignation(self) -> DataFrame:
        if self._cluster_assignation is None:
            raise ValueError("Call fit() before accessing cluster assignations.")
        return self._cluster_assignation

    def _attach_cluster_labels(self, data: DataFrame, labels):
        labeled_data = data.copy()
        labeled_data["cluster_id"] = labels
        return labeled_data

    def _build_centroids(self, centers):
        centroids = DataFrame(centers, columns=self._feature_names)
        centroids.insert(0, "cluster_id", range(len(centers)))
        return centroids
