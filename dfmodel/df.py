from collections import Counter

import pandas as pd
import numpy as np
import networkx as nx

from sklearn.neighbors import NearestNeighbors

class DigitalFamilyBinary:

    def __init__(self, bootstrap_iterations: int = 100, bootstrap_fraction: float = 0.9,
                 bootstrap_replace: bool = False, n_neighbors: int = 10, neighbor_metric: str = 'euclidean',):
        self.X_ = None
        self.full_estimator = None
        self.network_ = None
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

        edge_counts = Counter()

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

            distances, knn_results = neighbors.kneighbors(X_sample[features], return_distance=True)

            neighborhood_sizes = []
            mean_distance = []
            target_probabilities = []

            for i in range(knn_results.shape[0]):
                knn_idx = knn_results[i, :]

            self.estimators_.append(neighbors)
            self.data_.append(X_sample)

        self.full_estimator = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.neighbor_metric,
        )

        self.full_estimator.fit(X[features])

        connectivity_matrix = self.full_estimator.kneighbors_graph(n_neighbors=self.n_neighbors, mode='distance')

        self.network_ = nx.Graph()
        self.X_ = X.copy()

        for i in range(connectivity_matrix.shape[0]):

            node_idx = int(X.index[i])

            row = connectivity_matrix.getrow(i)

            for j in range(row.indices.size):

                neighbor = int(row.indices[j])
                weight = 1 / row.data[j]
                self.network_.add_edge(node_idx, neighbor, weight=weight)



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

        # TODO: remove this, I don't think that these columns need to be set like this
        self.bootstrap_columns = list(bootstrap_results.keys())

        self.bootstrap_results_df = pd.DataFrame(bootstrap_results)

        return self.bootstrap_results_df[self.bootstrap_columns].mean(axis=1)

    
    def model_sample(self, sample_data, features: list[str]):

        new_index = len(self.X_)

        sample_df = pd.DataFrame([sample_data])

        distances, neighbors = self.full_estimator.kneighbors(sample_df[features], return_distance=True)

        patient_specific_graph = self.network_.copy()

        for distance, neighbor in zip(distances[0, :], neighbors[0, :]):

            patient_specific_graph.add_edge(
                new_index, int(neighbor), weight= 1 / distance
            )

        return patient_specific_graph

    def neighbors(self, X, features: list[str]):

        distances, knn_results = self.full_estimator.kneighbors(X[features], return_distance=True)

        return distances, knn_results
