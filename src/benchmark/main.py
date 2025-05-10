# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This script runs the full benchmarking pipeline on a single patient graph.
Intended for validating information extraction, structure, and representation quality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Adds 'src' to sys.path


# Local application imports
from benchmark.modules.io_utils import (
    load_graph_from_file,
    save_results,
    data_display,
)
from benchmark.modules.run_benchmark import run_pipeline
from benchmark.modules.logging_utils import setup_logger

logger = setup_logger(__name__)

ORIGINAL_TEXT = (
    "A patient"
)


if __name__ == "__main__":
    current_graph_path = "/Users/bikia/Documents/Code/DynamicData/webapp/static/graphs/graph_001.json"
    graph, ok = load_graph_from_file(current_graph_path)

    if ok:
        cfg = {
            "reconstruct_params": {"include_nodes": True, "include_edges": True},
            "bertscore": True,
            "topology": True,
            "trajectory_embedding": True,
        }
        results = run_pipeline(graph, ORIGINAL_TEXT, cfg)
        # data_display(results)
        save_results(results, "output/results/results_001.json")
