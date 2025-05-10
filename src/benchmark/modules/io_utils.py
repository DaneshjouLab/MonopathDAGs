# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

# pylint: disable=broad-exception-caught

"""This modules containes I/O utilities for loading and saving data."""

import json
import os
from typing import Any
import networkx as nx
from networkx.readwrite import json_graph

# Local application imports
from .logging_utils import setup_logger

logger = setup_logger(__name__)
UTF_8 = "utf-8"

def load_graph_from_file(path: str) -> tuple[nx.DiGraph, bool]:
    """
    Load a graph from a JSON file using networkx's node-link format.
    Args:
        path (str): Path to the JSON file containing the graph data.
    Returns:
        tuple: A tuple containing the loaded graph and a boolean indicating success.
    """
    try:
        with open(path, encoding=UTF_8) as f:
            data = json.load(f)
            converted_data = convert_to_node_link_format(data) 
        return json_graph.node_link_graph(converted_data, directed=True), True
    except Exception as e:
        logger.error("Error loading graph: %s", e)
        return nx.DiGraph(), False

def save_results(results: dict, path: str) -> None:
    """
    Save the pipeline results to a JSON file.
    Args:
        results (dict): The results to save.
        path (str): The path to the output file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=UTF_8) as f:
        json.dump(results, f, indent=2)

def data_display(results: dict[str, Any]) -> None:
    """
    Pretty-print the pipeline results as structured tables for easy terminal viewing.
    Narrative is shown first, followed by status and individual metric tables.

    Args:
        results: The dict returned by run_pipeline.
    """
    # Header
    print("\n========== Pipeline Results ==========\n")

    # 1) Reconstructed Narrative first
    if "reconstructed_narrative" in results:
        print("Reconstructed Narrative:")
        print(results["reconstructed_narrative"])
        print()

    # 2) Status
    status = results.get("status")
    if status:
        print(f"Status: {status}\n")

    # 3) BERTScore table
    if "bertscore" in results:
        bs = results["bertscore"]
        print("BERTScore:")
        print(f"{'Metric':<12} {'Score':>10}")
        print("-" * 22)
        for metric, scores in bs.items():
            val = scores[0] if scores else float("nan")
            print(f"{metric:<12} {val:>10.4f}")
        print()

    # 4) Topology table
    if "topology" in results:
        topo = results["topology"]
        print("Topology Validation:")
        print("-" * 22)
        key_width = max(len(k) for k in topo)
        for key, val in topo.items():
            print(f"{key:<{key_width}} : {val}")
        print()

    # 5) Regex checks table
    if "regex" in results:
        rgx = results["regex"]
        print("Regex Checks:")
        print("-" * 22)
        key_width = max(len(k) for k in rgx)
        for key, val in rgx.items():
            print(f"{key:<{key_width}} : {val}")
        print()

    # 6) Any other top-level keys
    other_keys = {
        k: v for k, v in results.items()
        if k not in {"status", "bertscore", "topology", "regex", "reconstructed_narrative"}
    }
    if other_keys:
        print("Additional Data:")
        for key, val in other_keys.items():
            print(f"{key}: {val}")
        print()

def convert_to_node_link_format(original_json: dict) -> dict:
    """
    Convert a custom-formatted graph JSON with 'node_id' and 'edges'
    into NetworkX node-link format with 'id' and 'links'.

    Args:
        original_json (dict): Your original graph dictionary.

    Returns:
        dict: A transformed JSON compatible with networkx.node_link_graph().
    """
    converted = {
        "directed": True,
        "graph": {},
        "nodes": [],
        "links": []
    }

    for node in original_json.get("nodes", []):
        new_node = dict(node)  # copy
        new_node["id"] = new_node.pop("node_id", None)
        converted["nodes"].append(new_node)

    for edge in original_json.get("edges", []):
        new_edge = dict(edge)  # copy
        new_edge["source"] = new_edge.pop("from_node", None)
        new_edge["target"] = new_edge.pop("to_node", None)
        converted["links"].append(new_edge)

    return converted

def save_embedding_vector(vec: list[float], out_path: str, metadata: dict = None):
    """
    Save an embedding vector to a JSON file, appending it to the file if it already exists.
    Args:
        vec (list[float]): The embedding vector to save.
        out_path (str): The path to the output file.
        metadata (dict, optional): Additional metadata to include in the JSON object.
    """
    row = metadata or {}
    for i, v in enumerate(vec):
        row[f"dim_{i}"] = v
    with open(out_path, "a", encoding=UTF_8) as f:
        json.dump(row, f)
        f.write("\n")
