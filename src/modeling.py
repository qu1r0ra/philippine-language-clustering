from typing import Literal

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score

DimReductionMethod = Literal["svd", "pca"]
ClusteringMethod = Literal["kmeans", "agglomerative"]
RANDOM_STATE = 26  # it's my birthday!


def reduce_dimensionality(
    matrix: pd.DataFrame, method: DimReductionMethod = "svd", n_components: int = 50
) -> tuple[pd.DataFrame, object]:
    """
    Reduce dimensionality of a feature matrix.

    Returns:
        reduced_df (pd.DataFrame): Reduced feature space.
        reducer_model (object): The fitted dimensionality reduction model.
    """
    # Prevent crashing during training
    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two samples are required for dimensionality reduction."
        )

    # Ensure n_components doesn't exceed the number of samples or features
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)

    if method == "svd":
        reducer_model = TruncatedSVD(
            n_components=n_components, random_state=RANDOM_STATE
        )
        X = csr_matrix(matrix.to_numpy())
    elif method == "pca":
        reducer_model = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X = matrix
    else:
        raise ValueError(f"Unsupported method: {method}")

    reduced_matrix = reducer_model.fit_transform(X)

    # Report retained variance
    if hasattr(reducer_model, "explained_variance_ratio_"):
        explained = reducer_model.explained_variance_ratio_.sum() * 100
        print(f"{method.upper()} retained variance: {explained:.2f}%")

    reduced_df = pd.DataFrame(
        reduced_matrix,
        index=matrix.index,
        columns=[f"{method}_{i+1}" for i in range(n_components)],
    )

    return reduced_df, reducer_model


def cluster_languages(
    matrix: pd.DataFrame,
    method: ClusteringMethod = "agglomerative",
    n_clusters: int = 5,
    n_init: int = 10,
) -> tuple[pd.Series, object]:
    """
    Cluster languages using KMeans or Agglomerative Clustering.

    Returns:
        labels (pd.Series): Cluster labels for each language.
        clustering_model (object): The fitted clustering model.
    """
    if method == "kmeans":
        clustering_model = KMeans(
            n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=n_init
        )
    elif method == "agglomerative":
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    labels = clustering_model.fit_predict(matrix)
    clusters = pd.Series(labels, index=matrix.index, name="cluster")

    return clusters, clustering_model


def model_pipeline(
    matrix: pd.DataFrame,
    reducer_method: DimReductionMethod = "svd",
    n_components: int = 50,
    clusterer_method: ClusteringMethod = "agglomerative",
    n_clusters: int = 5,
) -> tuple[pd.DataFrame, pd.Series, object, object, float | None]:
    """
    Full modeling pipeline: dimensionality reduction + clustering.

    Returns:
        reduced (pd.DataFrame): Reduced feature matrix.
        clusters (pd.Series): Cluster labels.
        reducer_model (object): Fitted reduction model.
        clustering_model (object): Fitted clustering model.
        score (float | None): Silhouette score if applicable.
    """
    print(
        f"\nReducing dimensionality using {reducer_method.upper()} "
        f"({n_components} components)..."
    )
    reduced_df, reducer_model = reduce_dimensionality(
        matrix, method=reducer_method, n_components=n_components
    )

    print(
        f"\nClustering languages using {clusterer_method.upper()} "
        f"({n_clusters} clusters)..."
    )
    clusters, clustering_model = cluster_languages(
        reduced_df, method=clusterer_method, n_clusters=n_clusters
    )

    score = None
    if len(set(clusters)) > 1:
        score = float(silhouette_score(reduced_df, clusters))
        print(f"\nSilhouette score: {score:.4f}")
    else:
        print("\nSilhouette score not computed (only one cluster).")

    return reduced_df, clusters, reducer_model, clustering_model, score
