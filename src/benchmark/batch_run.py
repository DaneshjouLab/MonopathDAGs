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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Adds 'src' to sys.path

# Local application imports
from benchmark.modules.io_utils import (
    load_graph_from_file,
    save_results,
    build_graph_to_text_mapping,
    extract_original_text_from_html
)
from benchmark.modules.run_benchmark import run_pipeline
from benchmark.modules.logging_utils import setup_logger

logger = setup_logger(__name__)


# Replace this with your actual text used as reference
ORIGINAL_TEXT = (
    "Starry starry night, "
    "Paint your palette blue and grey, "
    "Look out on a summer's day, "
    "With eyes that know the darkness in my soul. "
    "Shadows on the hills, "
    "Sketch the trees and daffodils, "
    "Catch the breeze and winter chills, "
    "In colors on the snowy linen land. "
    "Now I understand, "
    "What you tried to say to me, "
    "And how you suffered for your sanity, "
    "And how you tried to set them free. "
    "They would not listen, they did not know how, "
    "Perhaps they'll listen now. "
)

# Input directory: All graph files
GRAPH_INPUT_DIR = Path(__file__).resolve().parents[2] / "webapp/static/graphs"

# Output directory: Results
RESULTS_OUTPUT_DIR = Path("output/results")
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths
metadata_csv = "webapp/static/graphs/mapping/graph_metadata.csv"
html_root_dir = "webapp/static/pmhc_html"

# graph_to_html = build_graph_to_text_mapping(metadata_csv, html_root_dir)

if __name__ == "__main__":
    graph_files = list(GRAPH_INPUT_DIR.glob("*.json"))

    if not graph_files:
        logger.warning("No JSON files found in input directory.")
    else:
        for fpath in graph_files[:5]:  # Limit to first 5 files for demonstration
            graph_id = fpath.stem
            # html_path = graph_to_html.get(graph_id)

            # if not html_path:
            #     logger.warning(f"No HTML path found for {graph_id}")
            #     continue
            
            # reference_case_text = extract_original_text_from_html(html_path)
            reference_case_text = ORIGINAL_TEXT

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

            results = run_pipeline(graph, reference_case_text, cfg)
            output_path = RESULTS_OUTPUT_DIR / f"{graph_id}.json"
            save_results(results, output_path)

        logger.info("Batch benchmarking completed.")