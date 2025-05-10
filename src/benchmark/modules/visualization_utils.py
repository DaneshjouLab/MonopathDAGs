# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

""" This module provides functions to visualize and summarize the results of the benchmarking
pipeline. It includes functions to plot BERTScore F1 scores, t-SNE embeddings, and topology
distributions, as well as to create a summary table of key metrics."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

PLOTS_DIR = "output/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_bertscore_f1(graph_ids, bertscore_f1):
    """Bar plot of BERTScore F1 for each graph."""
    _, ax = plt.subplots()
    sns.barplot(x=graph_ids, y=bertscore_f1, ax=ax)
    ax.set_title("BERTScore F1 per Graph")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bertscore_f1_barplot.png"))
    plt.close()


def plot_tsne_embeddings(embeddings, graph_ids):
    """2D t-SNE scatterplot for graph embeddings."""
    tsne_results = TSNE(n_components=2, perplexity=2, random_state=42).fit_transform(embeddings)
    _, ax = plt.subplots()
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=graph_ids, s=100, ax=ax)
    ax.set_title("t-SNE of Trajectory Embeddings")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(PLOTS_DIR, "trajectory_tsne.png"))
    plt.close()


def plot_topology_distributions(node_counts, edge_counts):
    """Histograms for node and edge counts across graphs."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(node_counts, bins=5, ax=ax1, kde=True)
    ax1.set_title("Node Count Distribution")
    ax1.set_xlabel("Number of Nodes")

    sns.histplot(edge_counts, bins=5, ax=ax2, kde=True)
    ax2.set_title("Edge Count Distribution")
    ax2.set_xlabel("Number of Edges")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "topology_distributions.png"))
    plt.close()


def summarize_metrics_table(graph_ids, bertscore_f1, node_counts, edge_counts):
    """Create and return a summary DataFrame of key metrics per graph."""
    df = pd.DataFrame({
        "Graph ID": graph_ids,
        "BERTScore F1": bertscore_f1,
        "Nodes": node_counts,
        "Edges": edge_counts
    })
    return df
