import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel

import logging

import networkx as nx
from networkx.readwrite import json_graph
from bert_score import score
import dspy

# ─── Configuration ──────────────────────────────────────────────────────────────

Graph = nx.DiGraph

# DSPy LLM settings
LM_MODEL     = "ollama_chat/llama3.2"
LM_API_BASE  = "http://localhost:11434"
LM_API_KEY   = ""  # or your local key if required

# BERTScore model settings
#BERTSCORE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BERTSCORE_MODEL = "bert-base-uncased"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




# ─── 1. LLM-BASED NARRATIVE RECONSTRUCTION ────────────────────────────────────

class LLMReconstructor:
    """
    Uses a DSPy-compatible LLM to reconstruct a narrative from selected parts of a graph.
    
    Usage:
        reconstructor = LLMReconstructor(
            model_name="ollama_chat/llama3.2",
            api_base="http://localhost:11434",
            api_key=""
        )
        narrative = reconstructor.reconstruct(
            graph,
            include_nodes=True,
            include_edges=False,
            node_ids=["n1", "n2"],
            node_attrs=["content"],
        )
    """
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        prompt_tpl: str = (
            "Reconstruct the clinical case report from this data:\n\n{payload}\n\n"
            "Write a coherent narrative including patient demographics, timeline of diagnoses, treatments, and outcomes."
        ),
        max_retries: int = 3
    ):
        try:
            self.lm = dspy.LM(
                model_name,
                api_base=api_base,
                api_key=api_key
            )
            self.prompt_tpl = prompt_tpl
            self.max_retries = max_retries
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def _build_payload(
        self,
        graph: Graph,
        include_nodes: bool,
        include_edges: bool,
        node_ids: Optional[List[str]],
        node_attrs: Optional[List[str]],
        edge_attrs: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build a structured payload from the graph."""
        payload: Dict[str, Any] = {}

        if include_nodes:
            payload["nodes"] = []
            for nid, attrs in graph.nodes(data=True):
                if node_ids and nid not in node_ids:
                    continue
                entry = {"id": nid}
                for k in (node_attrs or list(attrs.keys())):
                    if k in attrs:  # Only include existing attributes
                        entry[k] = attrs[k]
                payload["nodes"].append(entry)

        if include_edges:
            payload["edges"] = []
            for src, tgt, attrs in graph.edges(data=True):
                # Skip edges if we're filtering nodes and either endpoint is filtered out
                if node_ids and (src not in node_ids or tgt not in node_ids):
                    continue
                entry = {"source": src, "target": tgt}
                for k in (edge_attrs or list(attrs.keys())):
                    if k in attrs:  # Only include existing attributes
                        entry[k] = attrs[k]
                payload["edges"].append(entry)

        return payload

    def reconstruct(
        self,
        graph: Graph,
        *,
        include_nodes: bool = True,
        include_edges: bool = True,
        node_ids: Optional[List[str]] = None,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None
    ) -> str:
        """
        Build a custom payload from the graph and call the LLM.

        Args:
          graph: networkx.DiGraph with node/edge attributes.
          include_nodes: include a list of nodes in the payload.
          include_edges: include a list of edges in the payload.
          node_ids: list of node IDs to include (default: all).
          node_attrs: list of node attribute names to include (default: all).
          edge_attrs: list of edge attribute names to include (default: all).

        Returns:
          A string containing the reconstructed clinical narrative.
        """
        if not graph:
            logger.warning("Empty graph provided")
            return "No data available to reconstruct narrative."

        # Verify graph is a DiGraph
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Expected DiGraph, got {type(graph)}")
            if isinstance(graph, nx.Graph):
                logger.info("Converting undirected graph to directed")
                graph = nx.DiGraph(graph)
            else:
                raise TypeError("Input must be a networkx Graph or DiGraph")

        payload = self._build_payload(
            graph, include_nodes, include_edges, node_ids, node_attrs, edge_attrs
        )
        
        if not payload.get("nodes") and not payload.get("edges"):
            logger.warning("No nodes or edges in payload")
            return "Insufficient data to reconstruct narrative."

        payload_json = json.dumps(payload, indent=2)
        prompt = self.prompt_tpl.format(payload=payload_json)
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                resp_list = self.lm(messages=[{"role": "user", "content": prompt}])
                # dspy returns a list type of responses; by default that list contains exactly one completion,
                # so resp_list[0] is the single response you want
                return resp_list[0].strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error("All LLM call attempts failed")
                    raise
        
        # This should never be reached due to the exception above, but added for completeness
        return "Failed to reconstruct narrative due to LLM service errors."


# ─── 2. BERTScore-BASED FIDELITY EVALUATION ───────────────────────────────────

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

    def evaluate(self, refs: List[str], cands: List[str]) -> Dict[str, List[float]]:
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
            logger.warning(f"Mismatched list lengths: refs={len(refs)}, cands={len(cands)}")
        
        try:
            P, R, F1 = score(
                cands,
                refs,
                model_type=self.model_type,
                device=self.device,
                verbose=False
            )
            
            return {
                "precision": P.tolist(),
                "recall":    R.tolist(),
                "f1":        F1.tolist(),
            }
        except Exception as e:
            logger.error(f"BERTScore evaluation failed: {str(e)}")
            raise


# ─── 3. GRAPH TOPOLOGY VALIDATION ─────────────────────────────────────────────

class TopologyValidator:
    """
    Validates DAG properties: acyclicity, timestamp order, connectivity stats.

    Usage:
        validator = TopologyValidator(graph)
        stats = validator.run()
        # stats -> {"is_acyclic": bool, "timestamps_in_order": bool, ...}
    """
    def __init__(self, graph: Graph):
        self.G = graph

    def run(self) -> Dict[str, Any]:
        """
        Returns:
          A dict with structural validation results.
        """
        if not self.G or self.G.number_of_nodes() == 0:
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
            
        is_acyclic = nx.is_directed_acyclic_graph(self.G)
        
        # Check if timestamps exist in all nodes
        timestamp_attr = "timestamp"
        all_have_timestamps = all(timestamp_attr in data for _, data in self.G.nodes(data=True))
        
        timestamps_ok = False
        if all_have_timestamps:
            try:
                topo_nodes = list(nx.topological_sort(self.G))
                timestamps = [self.G.nodes[n][timestamp_attr] for n in topo_nodes]
                timestamps_ok = timestamps == sorted(timestamps)
            except nx.NetworkXUnfeasible:
                # Graph has cycles, can't do topological sort
                logger.warning("Cannot check timestamp order: graph has cycles")
                timestamps_ok = False
        else:
            logger.warning("Not all nodes have timestamp attributes")
        
        components = nx.number_weakly_connected_components(self.G)
        node_count = self.G.number_of_nodes()
        edge_count = self.G.number_of_edges()
        
        avg_in_deg = sum(dict(self.G.in_degree()).values()) / max(1, node_count)
        density = nx.density(self.G)
        
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


# ─── 4. REGEX-BASED SANITY CHECK (OPTIONAL) ───────────────────────────────────

class RegexValidator:
    """
    Lightweight regex tests on the raw JSON string to catch formatting issues.

    Usage:
        json_str = json.dumps(json_graph.node_link_data(graph))
        validator = RegexValidator(json_str)
        results = validator.run()
        # results -> {"nodes_array": bool, ...}
    """
    PATTERNS = {
        "nodes_array":        r'"nodes"\s*:\s*\[',
        "edges_array":        r'"edges"\s*:\s*\[',
        "no_trailing_commas": r',\s*\]',
        "node_entry":         r'\{\s*"id".+?\}',
        "edge_entry":         r'\{\s*"source".+?"label".+?\}',
    }

    def __init__(self, json_str: str):
        self.s = json_str

    def run(self) -> Dict[str, bool]:
        """
        Returns:
          A dict mapping each regex check name to True/False.
        """
        if not self.s or not isinstance(self.s, str):
            logger.warning(f"Invalid input to RegexValidator: {type(self.s)}")
            return {name: False for name in self.PATTERNS}
            
        results: Dict[str, bool] = {}
        for name, pat in self.PATTERNS.items():
            try:
                found = bool(re.search(pat, self.s))
                results[name] = not found if name == "no_trailing_commas" else found
            except Exception as e:
                logger.error(f"Regex check '{name}' failed: {str(e)}")
                results[name] = False
        return results


# ─── 5. ORCHESTRATOR ──────────────────────────────────────────────────────────

def run_pipeline(
    graph: Graph,
    original_text: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes modular evaluations in order:
      1. LLM reconstruction + BERTScore
      2. Topology checks
      3. Optional regex checks

    Args:
      graph: your clinical-case DAG (networkx.DiGraph)
      original_text: ground truth case-report text
      config:
        - "reconstruct_params": dict of args for reconstruct()
        - "bertscore": bool
        - "topology": bool
        - "regex": bool

    Returns:
      A dict with results from each enabled module.

    Usage:
        cfg = {
            "reconstruct_params": {"include_nodes": True, "include_edges": True},
            "bertscore": True,
            "topology": True,
            "regex": False
        }
        report = run_pipeline(G, original_text, cfg)
    """
    print("\n=============Starting Evaluation=============\n")
    results: Dict[str, Any] = {
        "status": "success",
        "errors": []
    }

    # Validate inputs
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        error = f"Invalid graph type: {type(graph)}"
        logger.error(error)
        results["status"] = "error"
        results["errors"].append(error)
        return results
        
    if not isinstance(original_text, str):
        error = f"Invalid original_text type: {type(original_text)}"
        logger.error(error)
        results["status"] = "error"
        results["errors"].append(error)
        return results

    # 1. Reconstruct narrative
    narrative = None
    if "reconstruct_params" in config:
        try:
            logger.info("Starting narrative reconstruction")
            reconstructor = LLMReconstructor(
                model_name=LM_MODEL,
                api_base=LM_API_BASE,
                api_key=LM_API_KEY
            )
            narrative = reconstructor.reconstruct(graph, **config["reconstruct_params"])
            results["reconstructed_narrative"] = narrative
            logger.info("Narrative reconstruction completed")
        except Exception as e:
            error = f"Narrative reconstruction failed: {str(e)}"
            logger.error(error)
            results["status"] = "partial"
            results["errors"].append(error)

    # 2. BERTScore
    if config.get("bertscore"):
        try:
            if narrative is None:
                error = "`reconstruct_params` must be provided to run `bertscore`"
                logger.error(error)
                results["status"] = "partial"
                results["errors"].append(error)
            else:
                logger.info("Starting BERTScore evaluation")
                # pick model from per-run override or top-level constant
                model_name = config.get("bertscore_model", BERTSCORE_MODEL)
                evaluator = BERTScoreEvaluator(model_type=model_name)
                results["bertscore"] = evaluator.evaluate(
                    refs=[original_text],
                    cands=[narrative]
                )
                logger.info("BERTScore evaluation completed")
        except Exception as e:
            error = f"BERTScore evaluation failed: {str(e)}"
            logger.error(error)
            results["status"] = "partial"
            results["errors"].append(error)

    # 3. Topology
    if config.get("topology"):
        try:
            logger.info("Starting topology validation")
            results["topology"] = TopologyValidator(graph).run()
            logger.info("Topology validation completed")
        except Exception as e:
            error = f"Topology validation failed: {str(e)}"
            logger.error(error)
            results["status"] = "partial"
            results["errors"].append(error)

    # 4. Regex (optional)
    if config.get("regex"):
        try:
            logger.info("Starting regex validation")
            from networkx.readwrite import json_graph
            json_str = json.dumps(json_graph.node_link_data(graph))
            results["regex"] = RegexValidator(json_str).run()
            logger.info("Regex validation completed")
        except Exception as e:
            error = f"Regex validation failed: {str(e)}"
            logger.error(error)
            results["status"] = "partial"
            results["errors"].append(error)

    if results["errors"]:
        if results["status"] == "success":
            results["status"] = "partial"
    
    if not results["errors"]:
        del results["errors"]

    return results


# ─── UTILITY FUNCTIONS ───────────────────────────────────────────────────────

def load_graph_from_file(filepath: str) -> Tuple[Graph, bool]:
    """
    Load a graph from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        Tuple of (graph, success_flag)
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = json_graph.node_link_graph(data, directed=True)
        return graph, True
    except Exception as e:
        logger.error(f"Failed to load graph from {filepath}: {str(e)}")
        return nx.DiGraph(), False


def save_results(results: Dict[str, Any], output_path: str) -> bool:
    """
    Save pipeline results to a JSON file.
    
    Args:
        results: The pipeline results dictionary.
        output_path: Where to save the results.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {str(e)}")
        return False


def data_display(results: Dict[str, Any]) -> None:
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


# ─── USAGE EXAMPLE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
   
    # 1. Call the nodes and edges for the graph object
    G = nx.DiGraph()

    # graph.get_nodes()
    # graph.get_edges()

    ########################

    # Hard-coded example
    # Build a 5-node example graph
    example_nodes = [
        ("n1", {"timestamp": "2025-04-29T08:00:00Z", "content": "Patient presents with chest pain"}),
        ("n2", {"timestamp": "2025-04-29T08:10:00Z", "content": "ECG performed"}),
        ("n3", {"timestamp": "2025-04-29T08:15:00Z", "content": "ST elevation noted"}),
        ("n4", {"timestamp": "2025-04-29T08:20:00Z", "content": "Aspirin administered"}),
        ("n5", {"timestamp": "2025-04-29T08:45:00Z", "content": "Transported to cath lab"}),
    ]
    for nid, attrs in example_nodes:
        G.add_node(nid, **attrs)
    # Add 4 edges with labels
    example_edges = [
        ("n1", "n2", {"label": "led_to"}),
        ("n2", "n3", {"label": "revealed"}),
        ("n3", "n4", {"label": "treated_with"}),
        ("n4", "n5", {"label": "followed_by"}),
    ]
    for src, tgt, attrs in example_edges:
        G.add_edge(src, tgt, **attrs)


    ########################

    # 2. Original case report
    original = (
        "A patient presents with chest pain. An ECG is performed, which reveals ST elevation. "
        "The patient receives aspirin and is then transported to the cath lab."
    )

    # 3. Configure your pipeline to include the required evaluations
    cfg = {
        "reconstruct_params": {"include_nodes": True, "include_edges": True},
        "bertscore": True,
        "topology": True,
        "regex": True,
    }

    
    # 4. Run and print the report
    try:
        report = run_pipeline(G, original, cfg)
        #print(json.dumps(report, indent=2))
        data_display(report)
        
        # Optionally save the results
        save_results(report, "output/pipeline_results.json")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}")
    
        

    
