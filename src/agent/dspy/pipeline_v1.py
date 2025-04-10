import dspy

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
    - step_index: Integer for sequence ordering
    - timestamp (optional, only include if clearly given): ISO 8601 datetime (e.g., \"2025-03-01T10:00:00Z\")
    - branch_label (optional): String or boolean label for branches/merges
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
    node_output: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
    edge_output = dspy.OutputField(type='list[dict]', desc='A list of dictionaries, where each dictionary represents an edge')

class branchConstruct(dspy.Signature):
    """
    """
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    node_output: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
    edge_output: list[dict] = dspy.InputField(desc="A list of edges which connect nodes")

class determineSideBranch(dspy.Signature):
    """
    """
    report_text: str = dspy.InputField(desc="body of text extracted from a case report")
    node_output: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
    edge_output: list[dict] = dspy.InputField(desc="A list of edges which connect nodes")



########################################

########################################

# Multi-stage module
# Combine these together

class dagGenerate(dspy.Module):
    """
    Use the signatures above to do a multi-stage pipeline to generate the graph
    """

    def __init__(self):
        """
        """
    
    def generate_node(str):
        """
        """

    def generate_edge(str):
        """
        """
    
    # Decide if this can be standalone, or will take in generate_node and generate_edge
    def generate_dag(str):
        """
        Use the signatures above and apply modules
        """



########################################

########################################


# Form docstrings using the docstring_dict
nodeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['node_instructions']
edgeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['edge_instructions']

# Will replace this later with extracted text
case_report = "A 58-year-old male with a 35-pack-year smoking history presented to the outpatient clinic with complaints of chronic cough, hemoptysis, and progressive dyspnea over the past 2 months. Initial physical examination revealed decreased breath sounds in the right lung field. Chest X-ray showed a right hilar mass, and subsequent contrast-enhanced CT of the chest identified a 5.5 cm mass in the right upper lobe with involvement of mediastinal lymph nodes and possible pleural effusion. CT-guided biopsy confirmed the diagnosis of poorly differentiated squamous cell carcinoma of the lung. Further staging with PET-CT revealed metabolic activity in the primary lesion, mediastinal nodes, and a suspicious lesion in the liver, suggesting stage IV disease. Brain MRI was negative for metastasis. Molecular profiling was performed and returned negative for EGFR, ALK, ROS1, and PD-L1 expression was low (<1%). Given the histology, stage, and biomarker profile, the patient was deemed a candidate for platinum-based chemotherapy. He was initiated on carboplatin and paclitaxel every three weeks. After two cycles, restaging scans demonstrated a partial response, with reduction in tumor size and decreased lymphadenopathy. The patient reported mild nausea and alopecia but tolerated the regimen well overall. After four cycles, the patient developed increasing fatigue, low-grade fever, and productive cough. Repeat imaging showed a new left lower lobe infiltrate and worsening pleural effusion. Thoracentesis revealed exudative effusion with malignant cells, confirming progressive disease. Second-line therapy was initiated with docetaxel and ramucirumab. The patient experienced transient stabilization of symptoms, but imaging at 12 weeks revealed hepatic progression and a new 1.8 cm brain metastasis in the right frontal lobe. Given the disease progression and declining performance status (ECOG 2), the patient was not a candidate for further systemic chemotherapy. He was referred for palliative whole-brain radiation therapy (WBRT) and symptomatic management. Supportive care was optimized, including low-dose opioids for dyspnea and corticosteroids for cerebral edema. Two months after WBRT, the patient presented with worsening confusion and hemiparesis. MRI revealed progression of brain metastases. After discussion with his family and care team, the decision was made to transition to hospice care. He died peacefully at home three weeks later. This case illustrates the challenges of treating advanced squamous cell lung cancer with limited molecular targets and the importance of supportive and palliative care in late-stage disease management."


# These will go inside the class dagConstruct
node_module = dspy.ChainOfThought(nodeConstruct)
node = node_module(report_text=case_report)

edge_module = dspy.ChainOfThought(edgeConstruct)
edge = edge_module(report_text=case_report, node_output=node)

print("")
print("NODES:   ", node.node_output)
print("")
print("EDGES:   ", edge.edge_output)
    

