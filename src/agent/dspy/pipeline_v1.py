from __future__ import annotations
import csv
import random
import dspy
from dspy import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from src.data.data_processors.pdf_to_text import extract_text_from_pdf

# =====================================
# DOCSTRING CONTENT
# =====================================

docstring_dict = {
    "dag_primer":
        """
    You are an assistant that converts clinical case narratives into dynamic Directed Acyclic Graphs (DAGs).

    Each DAG consists of:
    - Nodes = snapshots of the patient's state.
    - Edges = transitions between those states.

    Terminology guidance:
    - Use OMOP-standard concepts when possible for consistency and interoperability.
    - If a concept isn't covered by OMOP, use clear, logical labeling.
    """,

    "node_instructions":
        """
    Each node represents the patient at a specific point in time or logical step.

    Explicit guidelines for nodes:
    - Define nodes based on distinct clinical events or logical steps.
    - Combine simultaneous lab and imaging results into the same node.
    - Use separate nodes when events are clearly sequential or clinically distinct.    

    Node fields:
    - node_id (required): Unique identifier (Use capital letter of alphabet "A" and then "B" and so on and so forth.)
    - node_step_index (required): Integer for sequence ordering
    - timestamp (optional, only include if clearly given): ISO 8601 datetime (e.g., "2025-03-01T10:00:00Z")
    - branch_label (required): boolean label for branches, mark "TRUE" if a side branch, "FALSE" if otherwise
    - branch_id (required): str label for which branch, main branch is 0, and use increasing numerical id as side branches emerge
    - confidence (optional): Float from 0–1 for certainty, particularly from LLM outputs
    - content (required): Free-text interpretation or summary

    """,

    "edge_instructions":
        """
    Each edge represents a change from one node to another.

    Explicit guidelines for edges:
    - Create edges only when there is a clear clinical progression or change between nodes.
    - Maintain narrative or logical order — edges should flow from earlier to later events.
    - Combine co‑occurring findings into the same node, not across multiple edges.

    Edge fields:
    - edge_id (required): Unique identifier (Use format "node_id"_to_"node_id", such that the first "node_id" is the upstream node and the second "node_id" is the downstream node bounding the edge)
    - edge_step_index (required): Integer for narrative ordering
    - event_type (required): "Intervention" | "Observation" | "SpontaneousChange" | "Reinterpretation"
    - branch_flag (required): Boolean if this starts a side branch
    - confidence (optional): Float from 0–1, especially useful from LLM annotations
    - timestamp (optional, only include if clearly given): ISO 8601 datetime (e.g., "2025-03-01T10:00:00Z")
    - content (required): Free-text interpretation or summary of what changed between the nodes
    - change_type: "add" | "remove" | "update" | "reinterpretation" | "composite" | "narrative_add" | "split" | "merge"
    - Additional fields depending on change_type (`from`, `to`, `value`, `reason`, etc.)
    - Include `ambiguity_flag` and `temporal_reference` for uncertain timing or ambiguous sequencing.
    """,

    "branch_instructions":
        """
    Branches arise when physiologic changes or complications aren't part of the main pathway but impact patient states. Specifically, we are thinking of ephemeral changes.

    Mark side branches clearly:
    - Edge leading to branch: branch_flag = true
    - Nodes in branch: use branch_label clearly distinguishing alternate tracks.
    #### ^^ nah make something else for changing labels, this is just boolean
    """
}

# =====================================
# SELECTED LLM
# =====================================

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

# =====================================
# DSPY SIGNATURES
# =====================================

class nodeConstruct(dspy.Signature):
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    node_output = dspy.OutputField(type=list[dict], desc='A list of dictionaries, where each dictionary represents a node')

class edgeConstruct(dspy.Signature):
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    node_input: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
    edge_output = dspy.OutputField(type=list[dict], desc='A list of dictionaries, where each dictionary represents an edge')

class determineBranch(dspy.Signature):
    content: str = dspy.InputField(desc="content section from either a node or an edge")
    branch_input: list[dict] = dspy.InputField(desc="A list of ordered nodes and edges")
    branch_bool: bool = dspy.OutputField()

# =====================================
# FORM AND APPLY DOCSTRINGS
# =====================================

nodeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['node_instructions']
edgeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['edge_instructions']
determineBranch.__doc__ = docstring_dict["dag_primer"] + docstring_dict['branch_instructions']

# =====================================
# MODULES
# =====================================

class NodeEdgeGenerate(dspy.Module):
    def __init__(self):
        super().__init__()

    def generate_node(self, report_text):
        self.node_module = dspy.Predict(nodeConstruct)
        return self.node_module(report_text=report_text)

    def generate_edge(self, report_text, node_output):
        self.edge_module = dspy.Predict(edgeConstruct)
        return self.edge_module(report_text=report_text, node_input=node_output)

# =====================================
# FEW-SHOT OPTIMIZATION
# =====================================

class DetermineBranch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(determineBranch)

    def forward(self, content, branch_input):
        return self.program(content=content, branch_input=branch_input)

def branching_accuracy(gold, pred):
    return int(gold["branch_bool"] == pred["branch_bool"])

#####
# Need to fix this
def load_branch_examples_from_csv(csv_path):
    examples = []
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bool_val = row['branch_bool'].strip().upper() == "TRUE"
            example = {
                "inputs": {
                    "content": row['content'],
                    "branch_input": [
                        {"from_node": "A", "to_node": "B", "event_type": "Observation"}
                    ]
                },
                "outputs": {
                    "branch_bool": bool_val
                }
            }
            examples.append(example)
    return examples


branch_examples = load_branch_examples_from_csv("src/data/branch_data.csv")
random.shuffle(branch_examples)
trainset = branch_examples[:85]
devset   = branch_examples[85:]

teleprompter = BootstrapFewShot(
    metric=branching_accuracy,
    max_bootstrapped_demos=8,
    max_labeled_demos=85,
    max_rounds=1
)

optimized_determine_branch = teleprompter.compile(
    DetermineBranch(),
    trainset=trainset
)

print("\nSelected few-shot demonstrations:")
print(teleprompter.demonstrations)

evaluate = Evaluate(devset=devset, metric=branching_accuracy, display_progress=True)
eval_result = evaluate(optimized_determine_branch)
print("\n📊 Evaluation result on dev set:", eval_result)

# =====================================
# RUN PIPELINE
# =====================================

#report_text = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")
report_text = "A 64-year-old male with a history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history presented to the emergency department with progressive shortness of breath, dry cough, and unintentional weight loss over the past two months. He denied chest pain or hemoptysis."

NodeEdgeGenerate = NodeEdgeGenerate()
node_result = NodeEdgeGenerate.generate_node(report_text)
edge_result = NodeEdgeGenerate.generate_edge(report_text, node_result)

print("Nodes:\n", node_result)
print("\nEdges:\n", edge_result)

if edge_result['edge_output']:
    content = edge_result['edge_output'][-1].get("content", "")
else:
    content = ""
    print("No content found")

branch_result = optimized_determine_branch(
    content=content,
    branch_input=edge_result['edge_output']
)

print("\nBranch decision:", branch_result.branch_bool)

if branch_result.branch_bool and edge_result['edge_output']:
    print("Branching triggered — updating edge with branch_flag = TRUE")
    edge_result['edge_output'][-1]['branch_flag'] = True
else:
    print("No branch triggered or no edges available to update.")
