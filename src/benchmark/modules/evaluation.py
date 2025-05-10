# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

# pylint: disable=too-few-public-methods

"""This module contains evaluation classes for BERTScore and graph topology validation."""
from typing import Any
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from bert_score import score

# Local application imports
from .config import Graph
from .logging_utils import setup_logger

logger = setup_logger(__name__)


class BERTScoreEvaluator:
    """
    Computes BERTScore (precision, recall, F1) between original and reconstructed texts.

    Usage:
        evaluator = BERTScoreEvaluator(model_type="BERTSCORE_MODEL")
        results = evaluator.evaluate(
            refs=["ground truth text"],
            cands=["model generated text"]
        )
        # results -> {"precision": [...], "recall": [...], "f1": [...]}
    """

    def __init__(self, model_type: str, device: str = None):
        self.model_type = model_type
        self.device = device  # 'cuda', 'cpu', or None for auto-detection

        # ── Ensure the checkpoint is available locally ──
        # This will pull the tokenizer & model into ~/.cache/huggingface if missing.
        AutoTokenizer.from_pretrained(self.model_type)
        AutoModel.from_pretrained(self.model_type)

    def evaluate(self, refs: list[str], cands: list[str]) -> dict[str, list[float]]:
        """
        Args:
          refs: list of reference texts.
          cands: list of candidate texts.

        Returns:
          Dict with keys "precision", "recall", "f1" mapping to lists of scores.
        """
        if not refs or not cands:
            logger.warning("Empty references or candidates list")
            return {"precision": [], "recall": [], "f1": []}

        if len(refs) != len(cands):
            logger.warning(
                f"Mismatched list lengths: refs={len(refs)}, cands={len(cands)}"
            )

        try:
            precision, recall, f1_score = score(
                cands,
                refs,
                model_type=self.model_type,
                device=self.device,
                verbose=False,
            )

            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1_score.tolist(),
            }
        except Exception as e:
            logger.error(f"BERTScore evaluation failed: {str(e)}")
            raise


# GRAPH TOPOLOGY VALIDATION


class TopologyValidator:
    """
    Validates DAG properties: acyclicity, timestamp order, connectivity stats.

    Usage:
        validator = TopologyValidator(graph)
        stats = validator.run()
        # stats -> {"is_acyclic": bool, "timestamps_in_order": bool, ...}
    """

    def __init__(self, graph: Graph):
        self.graph_s = graph

    def run(self) -> dict[str, Any]:
        """
        Returns:
          A dict with structural validation results.
        """
        if not self.graph_s or self.graph_s.number_of_nodes() == 0:
            logger.warning("Empty graph in topology validator")
            return {
                "is_acyclic": True,  # Empty graphs are technically acyclic
                "timestamps_in_order": True,
                "weakly_connected_components": 0,
                "avg_in_degree": 0.0,
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
            }

        is_acyclic = nx.is_directed_acyclic_graph(self.graph_s)

        # Check if timestamps exist in all nodes
        timestamp_attr = "timestamp"
        all_have_timestamps = all(
            timestamp_attr in data for _, data in self.graph_s.nodes(data=True)
        )

        timestamps_ok = False
        if all_have_timestamps:
            try:
                topo_nodes = list(nx.topological_sort(self.graph_s))
                timestamps = [self.graph_s.nodes[n][timestamp_attr] for n in topo_nodes]
                timestamps_ok = timestamps == sorted(timestamps)
            except nx.NetworkXUnfeasible:
                # Graph has cycles, can't do topological sort
                logger.warning("Cannot check timestamp order: graph has cycles")
                timestamps_ok = False
        else:
            logger.warning("Not all nodes have timestamp attributes")

        components = nx.number_weakly_connected_components(self.graph_s)
        node_count = self.graph_s.number_of_nodes()
        edge_count = self.graph_s.number_of_edges()

        avg_in_deg = sum(dict(self.graph_s.in_degree()).values()) / max(1, node_count)
        density = nx.density(self.graph_s)

        return {
            "is_acyclic": is_acyclic,
            "timestamps_in_order": timestamps_ok,
            "all_nodes_have_timestamps": all_have_timestamps,
            "weakly_connected_components": components,
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_in_degree": avg_in_deg,
            "density": density,
        }
