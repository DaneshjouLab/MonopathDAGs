# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

# pylint: disable=broad-exception-caught

"""This modules containes I/O utilities for loading and saving data."""

# Standard library imports
import csv
import json
import os
from typing import Any
import warnings
import re

# Third-party imports
import networkx as nx
from networkx.readwrite import json_graph
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning

# Local application imports
from .logging_utils import setup_logger

logger = setup_logger(__name__)

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

UTF_8 = "utf-8"
# Extended to match parts like:
#   - 2.1. Case Summary
#   - 3.2.1 Case Report
#   - Case presentation and technical note
#   - Case Description
CASE_SECTION_PATTERNS = [
    r"^\d+(\.\d+)*\.?\s*case (presentation|summary|report|description|history)\b",
    r"^case presentation and technical note\b",
    r"^case\b",
]


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
            normalized_data = normalize_graph_for_networkx(converted_data)
        return json_graph.node_link_graph(normalized_data, directed=True), True
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
        k: v
        for k, v in results.items()
        if k
        not in {"status", "bertscore", "topology", "regex", "reconstructed_narrative"}
    }
    if other_keys:
        print("Additional Data:")
        for key, val in other_keys.items():
            print(f"{key}: {val}")
        print()


def convert_to_node_link_format(original_json: dict) -> dict:
    """
    Convert a JSON object to the node-link format used by networkx.
    Args:
        original_json (dict): The original JSON object.
    Returns:
        dict: The converted JSON object in node-link format.
    """
    converted = {"directed": True, "graph": {}, "nodes": [], "links": []}

    for node in original_json.get("nodes", []):
        new_node = dict(node)  # copy
        new_node["id"] = (
            new_node.get("node_id")
            or new_node.get("id")
            or new_node.get("customData", {}).get("node_id")
        )
        converted["nodes"].append(new_node)

    for edge in original_json.get("edges", []):
        new_edge = dict(edge)  # copy
        new_edge["source"] = new_edge.pop("from", None)
        new_edge["target"] = new_edge.pop("to", None)
        converted["links"].append(new_edge)

    return converted


def normalize_graph_for_networkx(raw_graph: dict) -> dict:
    """
    Normalize the graph for use with networkx.
    Args:
        raw_graph (dict): The raw graph data.
    Returns:
        dict: The normalized graph data.
    """
    for node in raw_graph["nodes"]:
        # Promote content and timestamp from customData
        if "customData" in node:
            custom = node["customData"]
            if "content" in custom:
                node["content"] = custom["content"]
            if "timestamp" in custom:
                node["timestamp"] = custom["timestamp"]
            node.update(custom)
            del node["customData"]

    for edge in raw_graph["links"]:
        if "from" in edge:
            edge["source"] = edge.pop("from")
        if "to" in edge:
            edge["target"] = edge.pop("to")

    return raw_graph


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


def build_graph_to_text_mapping(
    metadata_csv_path: str, html_root_dir: str
) -> dict[str, str]:
    """
    Build a mapping from graph ID (e.g., 'graph_006') to full HTML file path.

    Args:
        metadata_csv_path (str): Path to the graph_metadata.csv file.
        html_root_dir (str): Base path for the HTML files.

    Returns:
        dict: Mapping from graph ID to corresponding HTML file path.
    """
    graph_to_html = {}

    with open(metadata_csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            graph_id = row[0].strip()
            html_rel_path = row[2].strip().replace("./pmc_htmls", str(html_root_dir))
            graph_to_html[graph_id] = html_rel_path

    return graph_to_html


def extract_case_presentation(html: str) -> str:
    """
    Extract the case presentation section from an HTML string.
    Args:
        html (str): The HTML content as a string.
    Returns:
        str: The extracted case presentation text.
    """
    soup = BeautifulSoup(html, "html.parser")
    sections = soup.find_all("section")

    for sec in sections:
        # Find section title tag: prefers class='pmc_sec_title', otherwise any h2/h3
        title_tag = sec.find(["h2", "h3"], class_="pmc_sec_title") or sec.find(
            ["h2", "h3"]
        )
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True).lower()
        for pattern in CASE_SECTION_PATTERNS:
            if re.search(pattern, title):
                return sec.get_text(separator="\n", strip=True)

    return ""


def extract_case_presentation_from_file(filepath: str) -> str:
    """
    Extract the case presentation section from an HTML file.
    Args:
        filepath (str): Path to the HTML file.
    Returns:
        str: The extracted case presentation text."""
    with open(filepath, "r", encoding="utf-8") as file:
        html = file.read()
    return extract_case_presentation(html)
