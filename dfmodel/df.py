import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

class DigitalFamilyBinary:

    def __init__(self, bootstrap_iterations: int = 100, bootstrap_fraction: float = 0.9,
                 bootstrap_replace: bool = False, n_neighbors: int = 10, neighbor_metric: str = 'euclidean',):
        self.bootstrap_results_df = None
        self.bootstrap_columns = None
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_fraction = bootstrap_fraction
        self.bootstrap_replace = bootstrap_replace
        self.n_neighbors = n_neighbors
        self.neighbor_metric = neighbor_metric
        self.estimators_ = []
        self.data_ = []

    def fit(self, X, features: list[str]):

        for bootstrap in range(self.bootstrap_iterations):

            X_sample = X.sample(
                frac=self.bootstrap_fraction,
                replace=self.bootstrap_replace,
                random_state=bootstrap,
            )

            neighbors = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=self.neighbor_metric,
            )

            neighbors.fit(X_sample[features]) # removed feature columns

            self.estimators_.append(neighbors)
            self.data_.append(X_sample)

    def predict(self, X, features: list[str], target_column: str):

        bootstrap_results = {}

        for bootstrap, estimator in enumerate(self.estimators_):

            X_test_subset = X.copy()

            distances, knn_results = estimator.kneighbors(X_test_subset[features], return_distance=True)

            neighborhood_sizes = []
            mean_distance = []
            target_probabilities = []

            for i in range(knn_results.shape[0]):

                knn_idx = knn_results[i, :]

                if knn_idx.size > 0:

                    neighborhood = self.data_[bootstrap].iloc[knn_idx, :].copy()

                    neighborhood_sizes.append(neighborhood.shape[0])

                    mean_distance.append(
                        distances[i].mean()
                    )

                    target_binary_vector = np.where(neighborhood[target_column] == 1, 1, 0)

                    target_binary_probability = target_binary_vector.sum() / neighborhood.shape[0]

                    target_probabilities.append(target_binary_probability)

            bootstrap_results[f"target_bootstrap_{bootstrap}"] = target_probabilities

        self.bootstrap_columns = list(bootstrap_results.keys())

        self.bootstrap_results_df = pd.DataFrame(bootstrap_results)

        return self.bootstrap_results_df[self.bootstrap_columns].mean(axis=1)


