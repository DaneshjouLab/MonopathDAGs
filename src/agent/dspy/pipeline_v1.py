from __future__ import annotations
import csv
import random
import dspy
import json
from pprint import pprint
from dspy import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from src.data.data_processors.pdf_to_text import extract_text_from_pdf



# TO DO:
# UMLS API Key?????? To help map
# Modify code so that if the variable is a null value, just delete it
# Build ordered data structure to store all the nodes and edges chronilogically, so that branchClassifier can go back and change the branch_flags
# Need to do better job with the clinical data, it is not putting anything in right now

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
    - Use UMLS-standard concepts when possible for consistency and interoperability.
    - If a concept isn't covered by UMLS, use clear, logical labeling.
    """,

    "node_instructions": 
        """
    Each node represents the patient at a specific point in time or logical step.

    Guidelines for nodes:
    - Define nodes based on distinct clinical events or logical transitions.
    - Combine simultaneous lab and imaging results into a single node.
    - Use separate nodes when events are clearly sequential or clinically distinct.

    Node fields:
    - node_id (required): Unique identifier (Use capital letters in sequence: "A", "B", "C", etc.)
    - node_step_index (required): Integer for sequence ordering
    - content (required): All relevant content describing the node
    - timestamp (optional): ISO 8601 datetime, include only if explicitly available (e.g., "2025-03-01T10:00:00Z")

    Specific fields:
    Each node includes an optional structured field `clinical_data`, formatted as a dictionary with categories mapping to lists of dictionaries. Values should only be included if matched to the UMLS Metathesaurus; otherwise, omit and do not print/store it.
    clinical_data = {
        "medications": [
            {
                "drug": "UMLS_CUI or string",
                "dosage": "string",
                "frequency": "string",
                "modality": "oral | IV | IM | subcutaneous | transdermal | inhaled | other",
                "start_date": "ISO8601",
                "end_date": "ISO8601 or null",
                "indication": "UMLS_CUI or string"
            }
        ],
        "vitals": [
            {
                "type": "UMLS_CUI or string",
                "value": "numeric or string",
                "unit": "string",
                "timestamp": "ISO8601"
            }
        ],
        "labs": [
            {
                "test": "UMLS_CUI or string",
                "value": "numeric or string",
                "unit": "string",
                "flag": "normal | abnormal | critical | borderline | unknown",
                "reference_range": "string",
                "timestamp": "ISO8601"
            }
        ],
        "imaging": [
            {
                "type": "UMLS_CUI or string",
                "body_part": "UMLS_CUI or string",
                "modality": "X-ray | CT | MRI | Ultrasound | PET | other",
                "finding": "string",
                "impression": "string",
                "date": "ISO8601"
            }
        ],
        "procedures": [
            {
                "name": "UMLS_CUI or string",
                "approach": "open | laparoscopic | endoscopic | percutaneous | other",
                "date": "ISO8601",
                "location": "string",
                "performed_by": "string or provider ID",
                "outcome": "string or UMLS_CUI"
            }
        ],
        "HPI": [
            {
                "summary": "string",
                "duration": "string or ISO8601",
                "onset": "string or ISO8601",
                "progression": "gradual | sudden | fluctuating | unknown",
                "associated_symptoms": ["UMLS_CUI or strings"],
                "alleviating_factors": ["UMLS_CUI or strings"],
                "exacerbating_factors": ["UMLS_CUI or strings"]
            }
        ],
        "ROS": [
            {
                "system": "constitutional | cardiovascular | respiratory | GI | GU | neuro | psych | etc.",
                "findings": ["UMLS_CUI or strings"]
            }
        ],
        "functional_status": [
            {
                "domain": "mobility | cognition | ADLs | IADLs | speech | hearing | vision",
                "description": "string",
                "score": "numeric (if validated scale used)",
                "scale": "Barthel Index | MMSE | MoCA | other"
            }
        ],
        "mental_status": [
            {
                "domain": "orientation | memory | judgment | mood | speech",
                "finding": "UMLS_CUI or string",
                "timestamp": "ISO8601"
            }
        ],
        "social_history": [
            {
                "category": "smoking | alcohol | drug use | housing | employment | caregiver | support",
                "status": "current | past | never | unknown",
                "description": "string"
            }
        ],
        "allergies": [
            {
                "substance": "UMLS_CUI or string",
                "reaction": "UMLS_CUI or string",
                "severity": "mild | moderate | severe | anaphylaxis",
                "date_recorded": "ISO8601"
            }
        ],
        "diagnoses": [
            {
                "code": "ICD10 or SNOMED or UMLS_CUI",
                "label": "string",
                "status": "active | resolved | historical | suspected",
                "onset_date": "ISO8601 or null"
            }
        ]
    }
    """,

    "edge_instructions": 
        """
    Each edge represents a change from one node to another.

    Guidelines for edges:
    - Create edges only when there is a clear clinical progression or change between nodes.
    - Maintain narrative or logical order — edges should flow from earlier to later events.
    - Combine co-occurring findings into the same node, not across multiple edges.

    Explicit guidelines for node boundaries:
    - edge_id (required): Unique identifier (Use format "node_id"_to_"node_id", such that the first "node_id" is the upstream node and the second "node_id" is the downstream node bounding the edge)
    - branch_flag (required): Boolean if this starts a side branch
    - content (required): All content related to the given node

    Optional structured field for edge-level transitions:
    transition_event = {
        "trigger_type": "procedure | lab_change | medication_change | symptom_onset | interpretation | spontaneous",
        "trigger_entities": ["UMLS_CUI_1", "UMLS_CUI_2"],  # e.g., C0025598 = Metformin, C0011581 = Chest Pain
        "change_type": "addition | discontinuation | escalation | deescalation | reinterpretation | resolution | progression | other",  # Nature of the change
        "target_domain": "medication | symptom | diagnosis | lab | imaging | procedure | functional_status | vital_sign",  # What category was affected
        "timestamp": "ISO 8601 datetime (e.g., "2025-03-01T10:00:00Z"), only include if explicitly given and can be converted to datetime"
    }
    """,

    "branch_instructions":
        """
    Branches arise when physiologic changes or complications aren't part of the main pathway but impact patient states. Specifically, we are thinking of ephemeral changes.

    Mark side branches clearly:
    - Edge initiating a new branch: branch_flag = True
    """
}


# =====================================
# SELECTED LLM
# =====================================

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
# dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

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

class branchClassify(dspy.Signature):
    content: str = dspy.InputField(desc="content section from either a node or an edge")
    branch_bool: bool = dspy.OutputField()

# =====================================
# FORM AND APPLY DOCSTRINGS
# =====================================

nodeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['node_instructions']
edgeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['edge_instructions']
branchClassify.__doc__ = docstring_dict["dag_primer"] + docstring_dict['branch_instructions']

# =====================================
# MODULES
# =====================================

class NodeEdgeGenerate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.node_module = dspy.Predict(nodeConstruct)
        self.edge_module = dspy.Predict(edgeConstruct)

    def generate_node(self, report_text):
        return self.node_module(report_text=report_text)

    def generate_edge(self, report_text, node_output):
        return self.edge_module(report_text=report_text, node_input=node_output)


class BranchClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(branchClassify)

    def forward(self, content):
        return self.program(content=content)


    """

# Pull examples from csv file

def load_branch_examples_from_csv(csv_path):
    examples = []
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for idx, row in enumerate(reader):

            bool_val = row["branch_bool"].strip().upper() == "TRUE"

            # Fully DSPy-compliant example
            ex = dspy.Example(content = row["content"], branch_bool = bool_val).with_inputs("content")
            # ex.input_keys = ["content"]
            # ex.output_keys = ["branch_bool"]
            # ex = ex.with_inputs(content=row["content"])
            # ex = ex.with_outputs(branch_bool=bool_val)

            if idx < 2:
                print(f"Loaded example {idx}")
                print("  Inputs:", ex.inputs())
                print("  Outputs:", ex.labels())

            examples.append(ex)
    return examples


def branching_accuracy(gold, pred,trace=None):
    return int(gold.branch_bool == pred.branch_bool)


# Calling CSV with examples of content and bool pairs
branch_examples = load_branch_examples_from_csv("src/data/branch_data.csv")
random.shuffle(branch_examples)

trainset = branch_examples[:85]
devset   = branch_examples[85:]

# Search labeled examples to select best demos fro few-shot prompts
teleprompter = BootstrapFewShot(
    metric=branching_accuracy,
    max_bootstrapped_demos=8,
    max_labeled_demos=85,
    max_rounds=1
)

# New version of BranchClassifier injected with best few-shot examples
optimizedBranchClassifier = teleprompter.compile(
    BranchClassifier(),
    trainset=trainset
)

# Print the examples that were selected for few-shot
selected_demos = teleprompter.get_params().get("demos", [])
if selected_demos:
    print("\nSelected few-shot demonstrations:")
    for i, ex in enumerate(selected_demos):
        print(f"Example {i+1}:")
        print("  Inputs:", ex.inputs)
        print("  Outputs:", ex.outputs)
else:
    print("No few-shot demonstrations found.")



# =====================================
# (OPTIONAL) EVALUATE FEW-SHOT
# =====================================

#run_eval = input("Run evaluation on dev set? (y to confirm): ").strip().lower()
run_eval = "n"

if run_eval == "y":
    evaluate = Evaluate(devset=devset, metric=branching_accuracy, display_progress=True)
    eval_result = evaluate(optimizedBranchClassifier)
    print("\nEvaluation result on dev set:", eval_result)
else:
    print("Skipping evaluation.")



# =====================================
# RUN PIPELINE
# =====================================

report_text = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")
#report_text = "A 64-year-old male with a history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history presented to the emergency department with progressive shortness of breath, dry cough, and unintentional weight loss over the past two months. He denied chest pain or hemoptysis."
#report_text = "A 64-year-old male with a past medical history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history initially presented to his primary care provider with a two-month history of progressive exertional dyspnea, dry cough, and unintentional weight loss. Initial outpatient labs, including a complete blood count and basic metabolic panel, were unremarkable. A chest X-ray revealed a left upper lobe opacity, prompting referral to pulmonology. High-resolution CT of the chest demonstrated a 4.2 cm spiculated mass in the left upper lobe with associated mediastinal lymphadenopathy. PET-CT confirmed hypermetabolic activity in the mass and mediastinal nodes without distant metastasis. Bronchoscopy with transbronchial biopsy was performed, and histopathology revealed poorly differentiated non-small cell lung carcinoma (NSCLC). Molecular testing showed no actionable mutations. The patient was staged as clinical stage IIB (T2bN1M0) and discussed at multidisciplinary tumor board. He was deemed a surgical candidate and underwent left upper lobectomy with mediastinal lymph node dissection via VATS. Pathology confirmed NSCLC with negative margins but 2/12 positive nodes, confirming stage IIB disease. He recovered well postoperatively without complications and was discharged home on postoperative day three. Following recovery, he was referred to medical oncology, and adjuvant cisplatin-based chemotherapy was initiated six weeks post-surgery. He completed four cycles of chemotherapy over three months without major adverse effects aside from mild fatigue and nausea. Surveillance imaging at three months post-treatment showed no evidence of disease recurrence. He continues routine follow-up every three months with thoracic surgery and oncology. The case highlights the importance of early symptom recognition, timely referral, and coordinated multidisciplinary care in managing operable lung cancer."


# Generate the initial nodes and edges
NodeEdgeGenerate = NodeEdgeGenerate()
node_result = NodeEdgeGenerate.generate_node(report_text)
edge_result = NodeEdgeGenerate.generate_edge(report_text, node_result["node_output"])



print("Nodes:")
pprint(node_result["node_output"])

print("\nEdges:")
pprint(edge_result["edge_output"])


# Extract content to use for classification --> 'content' for determine bool
# Only doing for edges because only running BranchClassifier on edges
if edge_result['edge_output']:
    transition_content = json.loads(edge_result['edge_output'])
    print(transition_content)
else:
    transition_content = ""
    print("No content found")

# Run BranchClassifier classifier --> optimizedBranchClassifier is an instance of BranchClassifier using BoostrapFewShot; wraps the dspy.Predict(branchClassify))
branch_result = optimizedBranchClassifier(
    content=transition_content,
)

print("\n Clinical transition being evaluated:")
print(transition_content)


print("\n Branch decision:", branch_result.branch_bool)

if branch_result.branch_bool and edge_result['edge_output']:
    print(" Branch triggered — updating edge with branch_flag = TRUE")
    edge_result['edge_output'][-1]['branch_flag'] = True
else:
    print(" No branch triggered or no edges available to update.")
