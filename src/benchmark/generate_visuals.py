# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""This module generates visualizations for the results of the benchmark pipeline."""

import os
import json
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Adds 'src' to sys.path

from benchmark.modules.visualization_utils import (
    plot_bertscore_f1,
    plot_tsne_embeddings,
    plot_topology_distributions,
    summarize_metrics_table
)

RESULTS_DIR = "output/results"

# Lists to collect metrics
graph_ids = []
bertscore_f1s = []
trajectory_embeddings = []
node_counts = []
edge_counts = []

# Iterate over each result file
for fname in os.listdir(RESULTS_DIR):
    if fname.endswith(".json"):
        graph_id = fname.replace(".json", "")
        with open(os.path.join(RESULTS_DIR, fname), "r", encoding="utf-8") as f:
            result = json.load(f)

        graph_ids.append(graph_id)

        # Get BERTScore F1
        f1 = result.get("bertscore", {}).get("f1", [np.nan])[0]
        bertscore_f1s.append(f1)

        # Get embedding
        emb = result.get("trajectory_embedding")
        if emb:
            trajectory_embeddings.append(emb)

        # Get topology stats
        topo = result.get("topology", {})
        node_counts.append(topo.get("node_count", 0))
        edge_counts.append(topo.get("edge_count", 0))

# Convert embedding list to numpy array if not empty
if trajectory_embeddings:
    trajectory_embeddings = np.array(trajectory_embeddings)

# Generate Visualizations

plot_bertscore_f1(graph_ids, bertscore_f1s)

if len(trajectory_embeddings) > 1:
    plot_tsne_embeddings(trajectory_embeddings, graph_ids)

plot_topology_distributions(node_counts, edge_counts)

summary_df = summarize_metrics_table(graph_ids, bertscore_f1s, node_counts, edge_counts)
print("\n=== Summary Table ===")
print(summary_df)
