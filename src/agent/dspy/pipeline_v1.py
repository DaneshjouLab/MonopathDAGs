import dspy
from src.data.data_processors.pdf_to_text import extract_text_from_pdf

docstring_dict={
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

    Explicit guidelines for node boundaries:
    - Define nodes based on distinct clinical events or logical steps.
    - Combine simultaneous lab and imaging results into the same node.
    - Use separate nodes when events are clearly sequential or clinically distinct.    

    Required node fields:
    - node_id (required): Unique identifier (Use capital letter of alphabet \"A"\ and then \"B"\ and so on and so forth.)
    - step_index (required): Integer for sequence ordering
    - timestamp (optional, only include if clearly given): ISO 8601 datetime (e.g., \"2025-03-01T10:00:00Z\")
    - branch_label (required): boolean label for branches, mark "TRUE" if a side branch, "FALSE" if otherwise
    - branch_id (required): str label for which branch, main branch is 0, and use increasing numerical id as side branches emerge
    - confidence (optional): Float from 0–1 for certainty, particularly from LLM outputs
    - commentary (optional): Free-text interpretation or summary

    """,

"edge_instructions":
    """
    Each edge represents a change from one node to another.

    Explicit handling for edge types:
    - Branch only for actionable physiological or clinical events.
    - Informational updates (no immediate clinical impact) should NOT branch.
    - Rejoin explicitly when patient state aligns structurally with prior nodes.

    Explicit guidelines for node boundaries:
    - edge_id (required): Unique identifier (Use format "node_id"_to_"node_id", such that the first "node_id" is the upstream node and the second "node_id" is the downstream node bounding the edge)
    - from_node, to_node (required): IDs referencing source/target nodes
    - step_index (required): Integer for narrative ordering
    - event_type: \"Intervention\" | \"Observation\" | \"SpontaneousChange\" | \"Reinterpretation\"
    - branch_flag (optional): Boolean if this starts a branch
    - confidence (optional): Float from 0–1, especially useful from LLM annotations
    - timestamp (optional)
    - commentary (optional)

    Changes Array — required:
    Each item includes:
    - field: What changed
    - change_type: \"add\" | \"remove\" | \"update\" | \"reinterpretation\" | \"composite\" | \"narrative_add\" | \"split\" | \"merge\"
    - Additional fields depending on change_type (`from`, `to`, `value`, `reason`, etc.)
    - Include `ambiguity_flag` and `temporal_reference` for uncertain timing or ambiguous sequencing.
    """,

"branch_instructions":
    """

    Branches arise when physiologic changes or complications aren't part of the main pathway but impact patient states. Specifically, when a state is ephemeral.

    Mark side branches clearly:
    - Edge leading to branch: branch_flag = true
    - Nodes in branch: use branch_label clearly distinguishing alternate tracks.
    - Rejoin explicitly when interventions successfully revert to previous stable states.
    - Modular structure for easy modification or removal.

    """
    # Will put in a training set of what branches and what doesn't
}

# Language model
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm = lm, adapter = dspy.JSONAdapter())

class nodeConstruct(dspy.Signature):
    """
    """
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
   
    node_output = dspy.OutputField(type='list[dict]', desc='A list of dictionaries, where each dictionary represents a node')

class edgeConstruct(dspy.Signature):
    """
    """
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    node_input: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
   
    edge_output = dspy.OutputField(type='list[dict]', desc='A list of dictionaries, where each dictionary represents an edge')

class branchConstruct(dspy.Signature):
    """
    """
    # Insert the edge as a new dict in between the relevant nodes
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    node_input: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
    edge_input: list[dict] = dspy.InputField(desc="A list of edges which connect nodes")
    
    branch_output: list[dict] = dspy.OutputField(type='list[dict]', desc='Conneted nodes and edges based on ')

class determineBranch(dspy.Signature):
    """
    """
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    branch_input: list[dict] = dspy.InputField(desc="A list of ordered nodes and edges")

    branch_bool: bool = dspy.OutputField()

    # Make this a boolean and map the branching, tag whether it will be a side branch; later reconstruct



########################################

########################################

# Multi-stage module
# Combine these together

class dagGenerate(dspy.Module):
   
    def __init__(self):
        return None

    def generate_node(self, report_text):
        self.node_module = dspy.Predict(nodeConstruct)
        return self.node_module(report_text=report_text)

    def generate_edge(self, report_text, node_output):
        self.edge_module = dspy.Predict(edgeConstruct)
        return self.edge_module(report_text=report_text, node_input=node_output)

    # No branching yet
    # Actually don't need to generate the actual graph because I think that's Aaron's thing

    """
    def generate_dag(self, report_text):
        node_result = self.generate_node(report_text)
        edge_result = self.generate_edge(report_text, node_result.node_output)
        return {
            "nodes": node_result,
            "edges": edge_result
        }
    """




########################################

########################################


# Form docstrings using the docstring_dict
nodeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['node_instructions']
edgeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['edge_instructions']


# Extract text from PDF
#report_text = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")
report_text = "A 64-year-old male with a history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history presented to the emergency department with progressive shortness of breath, dry cough, and unintentional weight loss over the past two months. He denied chest pain or hemoptysis."


# Instantiate and generate nodes and edges
dagGenerate = dagGenerate()
node_result = dagGenerate.generate_node(report_text)
edge_result = dagGenerate.generate_edge(report_text, node_result)


print("Nodes:\n", node_result)
print("\nEdges:\n", edge_result)

