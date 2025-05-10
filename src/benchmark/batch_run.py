# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Run the full benchmarking pipeline on a batch of graph files in JSON format.
Each graph is benchmarked for information fidelity, topology structure,
and trajectory representation.
"""

from pathlib import Path
import os

# Local application imports
from benchmark.modules.io_utils import (
    load_graph_from_file,
    save_results,
)
from benchmark.modules.run_benchmark import run_pipeline
from benchmark.modules.logging_utils import setup_logger

logger = setup_logger(__name__)


# Replace this with your actual text used as reference
ORIGINAL_TEXT = (
    "A patient presents with chest pain. ECG was abnormal. "
    "Treated and sent to cath lab."
)

GRAPH_INPUT_DIR = "samples/json_output/am_journal_case_reports_2024"
RESULTS_OUTPUT_DIR = "output/graphs"

os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    input_dir = Path(GRAPH_INPUT_DIR)

    for fpath in input_dir.glob("*.json"):
        graph_id = fpath.stem
        logger.info(f"Running pipeline on: {fpath.name}")

        graph, ok = load_graph_from_file(fpath)
        if not ok:
            logger.error(f"Skipping {fpath.name} due to load error.")
            continue

        cfg = {
            "reconstruct_params": {"include_nodes": True, "include_edges": True},
            "bertscore": True,
            "topology": True,
            "trajectory_embedding": True,
        }

        results = run_pipeline(graph, ORIGINAL_TEXT, cfg)

        output_file = Path(RESULTS_OUTPUT_DIR) / f"{graph_id}.json"
        save_results(results, output_file)

    logger.info("Batch benchmarking completed.")
