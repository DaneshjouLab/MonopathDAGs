# pylint: disable=broad-exception-caught,too-few-public-methods

# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
This module implements similarity and clustering analysis of trajectory embeddings with dimensionality optimization.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define paths
RESULTS_DIR = Path("output/results")
PLOTS_DIR = Path("output/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS = []
GRAPH_IDS = []

def main(method="pca", dim_range=range(2, 11), cluster_range=range(2, 11)):
    # Step 1: Extract trajectory embeddings
    for file in RESULTS_DIR.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            emb = data.get("trajectory_embedding")
            if emb:
                EMBEDDINGS.append(emb)
                GRAPH_IDS.append(file.stem)

    if not EMBEDDINGS:
        raise ValueError("No valid embeddings found in results directory.")

    embedding_matrix = np.array(EMBEDDINGS)

    best_silhouette = -1
    best_dim = None
    best_k = None
    best_embedding = None
    best_labels = None
    silhouette_scores = {}

    if method == "none":
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=4)
            kmeans_labels = kmeans.fit_predict(embedding_matrix)
            sil_score = silhouette_score(embedding_matrix, kmeans_labels)
            ch_score = calinski_harabasz_score(embedding_matrix, kmeans_labels)
            db_score = davies_bouldin_score(embedding_matrix, kmeans_labels)
            # print(f"[dim=none, k={k}] Silhouette={sil_score:.4f}, CH={ch_score:.1f}, DB={db_score:.3f}")
            silhouette_scores[("none", k)] = sil_score

            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_dim = "none"
                best_k = k
                best_embedding = embedding_matrix
                best_labels = kmeans_labels
    else:
        for dim in dim_range:
            if method == "umap":
                reducer = umap.UMAP(n_components=dim, random_state=4, n_neighbors=5, min_dist=0.1)
            elif method == "pca":
                reducer = PCA(n_components=dim, random_state=4)
            else:
                raise ValueError("Invalid reduction method. Choose 'pca', 'umap', or 'none'.")

            reduced_embedding = reducer.fit_transform(embedding_matrix)

            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=4)
                kmeans_labels = kmeans.fit_predict(reduced_embedding)
                sil_score = silhouette_score(reduced_embedding, kmeans_labels)
                ch_score = calinski_harabasz_score(reduced_embedding, kmeans_labels)
                db_score = davies_bouldin_score(reduced_embedding, kmeans_labels)
                print(f"[dim={dim}, k={k}] Silhouette={sil_score:.4f}, CH={ch_score:.1f}, DB={db_score:.3f}")
                silhouette_scores[(dim, k)] = sil_score

                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_dim = dim
                    best_k = k
                    best_embedding = reduced_embedding
                    best_labels = kmeans_labels


    # Compute metrics for best config
    ch_score = calinski_harabasz_score(best_embedding, best_labels)
    db_score = davies_bouldin_score(best_embedding, best_labels)
    print(f"Best silhouette score: {best_silhouette:.4f} at dimension: {best_dim}, clusters: {best_k} using {method.upper()}")
    print(f"Calinski-Harabasz score: {ch_score:.1f}")
    print(f"Davies-Bouldin score: {db_score:.3f}")

    # Plot best scores bar chart
    plt.figure(figsize=(6, 4))
    metrics = [best_silhouette, ch_score, db_score]
    names = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
    colors = ["steelblue", "seagreen", "indianred"]
    plt.bar(names, metrics, color=colors)
    plt.title(f"Best Clustering Metrics (k={best_k}, dim={best_dim})")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"clustering_metrics_best_{method}.png")
    plt.close()

    # Output clustering results
    cluster_df = pd.DataFrame({
        "Graph ID": GRAPH_IDS,
        "KMeans Label": best_labels,
    })
    cluster_df.to_csv(PLOTS_DIR / "cluster_assignments.csv", index=False)

    # Plot silhouette score heatmap (skip for 'none')
    if method != "none":
        sil_matrix = pd.DataFrame(
            {(dim, k): [score] for (dim, k), score in silhouette_scores.items()}
        ).T.reset_index()
        sil_matrix.columns = ["Dimension", "Clusters", "Silhouette"]

        pivot_table = sil_matrix.pivot(index="Dimension", columns="Clusters", values="Silhouette")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Silhouette Score by Dimension and Number of Clusters")
        plt.xlabel("Clusters")
        plt.ylabel("Dimension")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"silhouette_heatmap_{method}.png")
        plt.close()

    # Visualize projection with cluster labels (only if reduced)
    if method != "none" and best_embedding.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=best_embedding[:, 0], y=best_embedding[:, 1], hue=best_labels, palette="tab10", legend="full")
        plt.title(f"{method.upper()} Projection (KMeans, k={best_k})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{method}_kmeans_clusters.png")
        plt.close()

    # Heatmap of cosine similarity
    similarity_matrix = cosine_similarity(embedding_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=GRAPH_IDS, yticklabels=GRAPH_IDS,
                cmap="coolwarm", square=True)
    plt.title("Trajectory Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "similarity_heatmap.png")
    plt.close()

    # Save cluster assignments
    with open(RESULTS_DIR / "trajectory_clusters_kmeans.json", "w") as f:
        json.dump(dict(zip(GRAPH_IDS, best_labels.tolist())), f, indent=2)


if __name__ == "__main__":
    main(method="umap")  # Options: "pca", "umap", or "none"