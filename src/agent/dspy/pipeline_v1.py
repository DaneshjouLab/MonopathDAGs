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
from src.agent.dspy.testing_more import (extract_paragraphs, recursively_decompose_to_atomic_sentences)




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

    WHEN LOOKING AT THE CASE REPORT INPUT, IGNORE THE REFERENCES.
    """,

    "node_instructions":
    """
    You are given a clinical case report. Your task is to extract a sequence of nodes representing the patient's evolving clinical state.

    Output format:
    Return a list of node dictionaries, each with:
    - node_id (e.g., "A", "B", "C")
    - node_step_index (integer for order)
    - content (concise clinical summary)
    - timestamp (optional, ISO8601)
    - clinical_data (optional, structured and UMLS-linked only)

    Example:
    [
      {
        "node_id": "A",
        "node_step_index": 0,
        "content": "The patient presented with bilateral painless testicular masses.",
        "clinical_data": {
          "imaging": [
            {
              "type": "C0030039",
              "body_part": "C0040580",
              "modality": "Ultrasound",
              "finding": "Multiple hyperechoic lesions",
              "impression": "Suggestive of Sertoli cell tumor",
              "date": "2015-12-01T00:00:00Z"
            }
          ]
        }
      }
    ]

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

    Guidelines:
    - Create one node per clinically meaningful state.
    - Combine co-occurring labs/imaging into the same node.
    - Use separate nodes for clearly sequential or distinct events.
    - Only include clinical_data fields when concepts are UMLS-mappable.
    - Do not return anything outside the list format.

    """,

    "edge_instructions": 
        """
    Each edge represents a change from one node to another.

    Guidelines for edges:
    - Create edges only when there is a clear clinical progression or change between nodes.
    - Maintain narrative or logical order — edges should flow from earlier to later events.
    - Combine co-occurring findings into the same node, not across multiple edges.

    Edge fields:
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

#lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
# dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

# =====================================
# DSPY SIGNATURES
# =====================================

class nodeConstruct(dspy.Signature):
    text_input: str = dspy.InputField(desc="body of text extracted from a case report")
    node_output = dspy.OutputField(type=list[dict], desc="A list of node dictionaries with node_id, node_step_index, content, optional timestamp and clinical_data")

class edgeConstruct(dspy.Signature):
    text_input: str = dspy.InputField(desc="body of text extracted from a case report")
    node_input: list[dict] = dspy.InputField(desc="A list of nodes with which to connect with edges")
    edge_output = dspy.OutputField(type=list[dict], desc="A list of edge dictionaries with edge_id, branch_flag, content, and optional transition_event")

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

    def generate_node(self, text_input):
        return self.node_module(text_input=text_input)

    def generate_edge(self, text_input, node_output):
        return self.edge_module(text_input=text_input, node_input=node_output)


class BranchClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(branchClassify)

    def forward(self, content):
        return self.program(content=content)


# =====================================
# FEW-SHOT OPTIMIZATION
# =====================================

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


"""
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


"""    

# =====================================
# EXTRACTION PROCESS FUNCTIONS
# =====================================


"""
def extract_stepwise_dag(text_input: str) -> dict:

    # Sequential breakdown of case report for better extraction to DAG
    # (We noticed that smaller LMs, at least, are not very good at extracting from large bodies of text)

    # PDF -> extracted text -> paragraphs -> atomic statements -> nodes/edges

    # text_input -> Should already be in str format; so if pdf path, use extract_text_frompdf first


    # Paragraph extraction from raw text
    paragraphs = extract_paragraphs(text_input)

    # Decompose to atomic statements
    atomic_statements = []
    for p in paragraphs:
        for sentence in p["paragraph"].split(". "):
            if sentence.strip():
                decomposed = recursively_decompose_to_atomic_sentences(sentence.strip())
                atomic_statements.extend(decomposed)

    # Join atomic content
    atomic_text = "\n".join(atomic_statements)

    # Generate node and edge outputs
    generator = NodeEdgeGenerate()
    node_result = generator.generate_node(text_input=atomic_text)
    edge_result = generator.generate_edge(text_input=atomic_text, node_output=node_result["node_output"])
    
    # Convert nodes and edges to JSON strings for consistent handling
    nodes_json = json.dumps(node_result["node_output"])
    edges_json = json.dumps(edge_result["edge_output"])
    
    return {
        "nodes": nodes_json,
        "edges": edges_json,
        "atomic_statements": atomic_statements
    }

"""


def decompose_content_to_atomic_statements(content_block: str) -> list[str]:
    """
    Given a content string from a node or edge, decompose it into atomic clinical statements.

    Intended for use *after* nodes or edges are generated from the full case report.

    Parameters:
    - content_block (str): A single content string from a node or edge.

    Returns:
    - List of atomic clinical sentences (List[str])
    """
    atomic_statements = []

    for sentence in content_block.split(". "):
        if sentence.strip():
            decomposed = recursively_decompose_to_atomic_sentences(sentence.strip())
            atomic_statements.extend(decomposed)

    return atomic_statements



def chunk_and_generate_nodes(case_report: str, generator: NodeEdgeGenerate, max_chunk_size: int = 25) -> list[dict]:
    """
    Breaks a long case report into chunks and applies node generation sequentially.

    Parameters:
    - case_report: Full case report text.
    - generator: An instance of NodeEdgeGenerate.
    - max_chunk_size: Max characters per chunk.

    Returns:
    - Combined list of all generated nodes.
    """
    paragraphs = extract_paragraphs(case_report)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        p_text = para["paragraph"].strip()
        if len(current_chunk) + len(p_text) < max_chunk_size:
            current_chunk += "\n" + p_text
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p_text
    if current_chunk:
        chunks.append(current_chunk.strip())

    all_nodes = []
    for chunk in chunks:
        result = generator.generate_node(text_input=chunk)
        nodes = result.get("node_output", [])

        # Try to parse if model returned a JSON string
        if isinstance(nodes, str):
            try:
                nodes = json.loads(nodes)
            except json.JSONDecodeError:
                print("Warning: Node output not valid JSON.")
                continue

        if isinstance(nodes, list):
            all_nodes.extend(nodes)

    return all_nodes




# =====================================
# RUN PIPELINE
# =====================================

#case_report = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")
#case_report = "A 64-year-old male with a history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history presented to the emergency department with progressive shortness of breath, dry cough, and unintentional weight loss over the past two months. He denied chest pain or hemoptysis."
#case_report = "A 64-year-old male with a past medical history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history initially presented to his primary care provider with a two-month history of progressive exertional dyspnea, dry cough, and unintentional weight loss. Initial outpatient labs, including a complete blood count and basic metabolic panel, were unremarkable. A chest X-ray revealed a left upper lobe opacity, prompting referral to pulmonology. High-resolution CT of the chest demonstrated a 4.2 cm spiculated mass in the left upper lobe with associated mediastinal lymphadenopathy. PET-CT confirmed hypermetabolic activity in the mass and mediastinal nodes without distant metastasis. Bronchoscopy with transbronchial biopsy was performed, and histopathology revealed poorly differentiated non-small cell lung carcinoma (NSCLC). Molecular testing showed no actionable mutations. The patient was staged as clinical stage IIB (T2bN1M0) and discussed at multidisciplinary tumor board. He was deemed a surgical candidate and underwent left upper lobectomy with mediastinal lymph node dissection via VATS. Pathology confirmed NSCLC with negative margins but 2/12 positive nodes, confirming stage IIB disease. He recovered well postoperatively without complications and was discharged home on postoperative day three. Following recovery, he was referred to medical oncology, and adjuvant cisplatin-based chemotherapy was initiated six weeks post-surgery. He completed four cycles of chemotherapy over three months without major adverse effects aside from mild fatigue and nausea. Surveillance imaging at three months post-treatment showed no evidence of disease recurrence. He continues routine follow-up every three months with thoracic surgery and oncology. The case highlights the importance of early symptom recognition, timely referral, and coordinated multidisciplinary care in managing operable lung cancer."
case_report = "Testicular tumefaction is a common concern in urology. Most causes can be easily identified through anamnesis, clinical examination, blood tests, or ultrasonography. However, for testicular masses, differential diagnosis can be challenging. The 2022 WHO classification of tumors of the urinary system and the male genital organs lists 43 different types of testicular tumors, of which 8 are categorized as having unspecified, borderline, or uncertain behavior. Given that testicular cancers primarily affect young males, accurate diagnosis and assessment of pathological progression are crucial for determining the most appropriate therapeutic strategy. However, data on the management of many types of testicular tumors are scarce, and existing case studies are often only partially comparable. Consequently, it can be difficult to choose between a more radical treatment option and favoring the conservation of fertility and endocrine function, especially face-to-face to a young patient. In this clinical case, we share our experience with a 37-year-old patient with a multifocal, bilateral testicular LCCSCT presenting as painless testicular masses that took an unexpected course, resulting in a fatal outcome. A 37-year-old man was referred to our institution in December 2015. He noticed bilateral painless testicular masses. Physical examination revealed no gynecomastia or skin anomalies. Ultrasound showed bilateral macro-orchitis with multiple intratesticular hyperechoic lesions with acoustic shadowing. These lesions were 2.4 cm and 2.1 cm on the left and right testicles, respectively. A pelvic MRI was performed, showing bilateral intratesticular lesions with clear hypointense T1 and T2 signals, which take intense contrast enhancement after gadolinium injection. The MRI revealed 10 lesions on the left testicle and 5 lesions on the right. Serum alpha-fetoprotein (AFP), beta-hCG, LDH, testosterone, estradiol, and gonadotropin levels were in normal range. A surgical testicular exploration was performed, and a right testicular nodule was enucleated and sent to pathology. Histology revealed a benign large-cell calcifying Sertoli cell tumor under 5 cm, with no mitosis, cytological atypia, vascular permeation, or necrosis. Immunohistochemistry was positive for vimentin, calretinin, inhibin, and cytokeratin AE1/AE3. Given the benign histology, radical orchiectomy was not pursued; instead, regular follow-up was initiated. Follow-up consisted of testicular exams and sonography every 6 months for 2 years, then annually. No genetic testing was performed due to absence of syndromic features or family history. Endocrine tests ruled out glucose, thyroid, adrenal, and pheochromocytoma abnormalities. In June 2021, testicular exam revealed left-sided induration; MRI confirmed tumor infiltration into the spermatic cord. A left radical orchiectomy was performed. Pathology revealed a multifocal large-cell calcifying Sertoli cell tumor (largest 6.5 cm) with spermatic cord invasion and neoplastic vascular permeation. CT scan showed para-aortic lymphadenopathy, pulmonary nodules, and sclerotic spinal lesions. PET confirmed left inguinal and iliac lymphadenopathy and a penile lesion, but no lung uptake. The patient underwent lymph node dissection and right radical orchiectomy, revealing metastatic disease and multifocal LCCSCT. By November 2021, local extension into the penis and inguinal canal had occurred, with widespread metastases by December. Chemotherapy with vinblastine, cisplatin, and ifosfamide was ineffective. Paclitaxel was then given, but the disease progressed. The patient enrolled in a trial with Axitinib and pazopanib but died 7 months later in palliative care. This case highlights that despite benign histological features, LCCSCTs can be aggressive, especially in sporadic bilateral cases. Although most are benign, large, multifocal, sporadic tumors may behave malignantly. WHO classification lists size >5 cm, necrosis, pleomorphism, and invasive growth as risk factors. Some studies suggest lowering size threshold to >2.4 cm. Diagnosis requires histopathology and immunohistochemistry. Sertoli cell tumors express inhibin, calretinin, vimentin, and keratin, but typically lack beta-catenin nuclear staining. Testis-sparing surgery (TSS) may be considered for tumors with 0–1 risk factor. Retroperitoneal lymph node dissection (RPLND) offers benefit for regional spread, but systemic chemotherapy and radiation have low efficacy. Bilateral orchiectomy should have been the initial strategy in this case."

"""
# Generate the initial nodes and edges
generator = NodeEdgeGenerate()
node_result = generator.generate_node(case_report)
edge_result = generator.generate_edge(case_report, node_result["node_output"])

# Store nodes and edges as Python objects for further processing
nodes_obj = node_result["node_output"]
edges_obj = edge_result["edge_output"]

"""

generator = NodeEdgeGenerate()

# Generate nodes in chunks to avoid overload for smaller models
nodes_obj = chunk_and_generate_nodes(case_report, generator)

# Generate edges once all nodes are collected
edge_result = generator.generate_edge(text_input=case_report, node_output=nodes_obj)
edges_obj = edge_result["edge_output"]

print("Nodes:")
pprint(nodes_obj)

print("\nEdges:")
pprint(edges_obj)

# Loop through nodes and edges and extract atomic statements from content output
for node in nodes_obj:
    content_str = node.get("content", "")
    atomic_sents = decompose_content_to_atomic_statements(content_str)

    print(f"\nAtomic breakdown of content from node_id {node.get('node_id', 'unknown')}:")
    for s in atomic_sents:
        print(f"- {s}")

for edge in edges_obj:
    content_str = edge.get("content", "")
    atomic_sents = decompose_content_to_atomic_statements(content_str)

    print(f"\nAtomic breakdown of content from edge_id {edge.get('edge_id', 'unknown')}:")
    for s in atomic_sents:
        print(f"- {s}")

"""

# Extract content to use for classification — only doing for edges
if edges:
    transition_content = json.dumps(edges)
else:
    transition_content = ""
    print("No content found")

# Run BranchClassifier classifier
branch_result = optimizedBranchClassifier(
    content=transition_content,
)

print("\nClinical transition being evaluated:")
print(transition_content)

print("\nBranch decision:", branch_result.branch_bool)

if branch_result.branch_bool and edges:
    print("Branch triggered — updating edge with branch_flag = TRUE")
    edges[-1]['branch_flag'] = True
else:
    print("No branch triggered or no edges available to update.")


"""