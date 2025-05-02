
from __future__ import annotations
import csv
import random
import dspy
import json
import time
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from pprint import pprint
from tqdm import tqdm
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


# Make sure chunking is working - DONE
# Need the content extraction to be more exhuastive - Does not work, but probs a model problem
# Then convert content to atomic - yah but model problem
# And then view the atomic statements in aggregate to fill in clinical data - yah
# Clinical data needs to be standalone signature and module - DONE

# NOW NEED TO MAKE BRANCH CLASSIFY
# MAKE EDGES FINE AND CORRECT LABELS
# MAKE NEAT
# Make atomic less intensive please
# Right now commented out atomic so it doesn't keep killing my computer

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

    Text extraction guidance:
    - When looking at the case report input, ignore the references, background, conclusions etc. sections
    - We only want to extract on content relating to the specific patient discussed in the case report


    """,

    "node_instructions":
    """
    You are given a clinical case report. Your task is to extract a sequence of nodes representing the patient's evolving clinical state.

    Guidelines:
    - Create one node per clinically meaningful state.
    - Combine co-occurring labs/imaging into the same node.
    - Use separate nodes for clearly sequential or distinct events.
    - Do not return anything outside the list format. Should be in JSON compatible style.
    - Keep imaging content packaged in one node if no clear temporal change is indicated
    - Keep pathology / histology content packaged in one node if no clear temporal change is indicated
    - node_memory is a running memory that updates as we add new nodes; use it to preserve context, merge overlapping details, and avoid redundant or stale states



    Output format:
    Return a list of node dictionaries, in this order from top to bottom, each with:
    - node_id (In ascending alphabetical order, e.g., "A", "B", "C")
    - node_step_index (integer for order)
    - content (exhuastive clinical content, include all relevant details for the given node)
    - timestamp (optional, ISO8601)

    Example:
    [
      {
        "node_id": "A",
        "node_step_index": 0,
        "content": "The patient presented with bilateral painless testicular masses."
      }
    ]


    """,

    "edge_instructions": 
        """
    Each edge represents a change from one node to another. There should be one edge in between every pair of adjacent nodes.

    Guidelines for edges:
    - Create edges only when there is a clear clinical progression or change between nodes.
    - Maintain narrative or logical order — edges should flow from earlier to later events.
    - Combine co-occurring findings into the same node, not across multiple edges.

    Output format:
    Return a list of edge dictionaries, in this order, each with:
    - edge_id: Unique identifier (Use format "node_id"_to_"node_id", such that the first "node_id" is the upstream node and the second "node_id" is the downstream node bounding the edge)
    - branch_flag: Boolean if this starts a side branch, default = True
    - content: Exhuastive clinical content, include all relevant details for the given node

    Optional structured field for edge-level transitions:
    transition_event = {
        "trigger_type": "procedure | lab_change | medication_change | symptom_onset | interpretation | spontaneous",
        "trigger_entities": ["UMLS_CUI_1", "UMLS_CUI_2"],  # e.g., C0025598 = Metformin, C0011581 = Chest Pain
        "change_type": "addition | discontinuation | escalation | deescalation | reinterpretation | resolution | progression | other",  # Nature of the change
        "target_domain": "medication | symptom | diagnosis | lab | imaging | procedure | functional_status | vital_sign",  # What category was affected
        "timestamp": "ISO 8601 datetime (e.g., "2025-03-01T10:00:00Z"), only include if explicitly given and can be converted to datetime"
    }

    """,

    "node_clinical_data_instructions": 
        """
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
from dotenv import load_dotenv
load_dotenv(".config/.env")
import os
gemini_api_key = os.environ.get("GEMINI_APIKEY")

print(gemini_api_key)
#lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
#lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
#lm = dspy.LM(model='ollama_chat/meta-llama-3-8b-instruct', api_base='http://localhost:11434', api_key='') # 8B parameter

lm = dspy.LM('gemini/gemini-2.0-flash', api_key=gemini_api_key,temperature=0.2)

print

#dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

# =====================================
# DSPY SIGNATURES
# =====================================

class nodeConstruct(dspy.Signature):
    text_input: str = dspy.InputField(desc="body of text extracted from a case report")
    node_memory: list[dict] = dspy.InputField(optional=True, desc="List of nodes generated so far (memory). Use as context when generating new nodes so there is no duplication.") # Sliding window, grows as we make new nodes
    node_output = dspy.OutputField(type=list[dict], desc="A list of node dictionaries with node_id, node_step_index, content, optional timestamp and clinical_data")


class edgeConstruct(dspy.Signature):
    #text_input: str = dspy.InputField(desc="body of text extracted from a case report")
    node_input: list[dict] = dspy.InputField(desc="Ordered list of node dictionaries as generated by nodeConstruct")
    edge_output: list[dict] = dspy.OutputField(desc="A list of edge dictionaries with edge_id, branch_flag, content, and optional transition event. There should be one edge in between every adjacent node.")

class nodeClinicalDataExtract(dspy.Signature):
    # Optional: You can comment this out to rely only on atomic_sentences
    content: str = dspy.InputField(desc="Narrative clinical content from a node")
    atomic_sentences: list[str] = dspy.InputField(desc="List of atomic-level clinical statements derived from the node content")
    clinical_data: dict = dspy.OutputField(desc="Structured clinical data dictionary with fields like medications, labs, imaging, etc.")

class branchClassify(dspy.Signature):
    content: str = dspy.InputField(desc="content section from either a node or an edge")
    branch_bool: bool = dspy.OutputField()

# =====================================
# FORM AND APPLY DOCSTRINGS
# =====================================

nodeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['node_instructions']
edgeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['edge_instructions']
nodeClinicalDataExtract.__doc__ = docstring_dict["dag_primer"] + docstring_dict["node_clinical_data_instructions"]
branchClassify.__doc__ = docstring_dict["dag_primer"] + docstring_dict['branch_instructions']

# =====================================
# MODULES
# =====================================


class NodeEdgeGenerate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.node_module = dspy.Predict(nodeConstruct)
        #self.edge_module = dspy.Predict(edgeConstruct)
        self.edge_module = dspy.Predict(edgeConstruct, examples=edge_fewshot_examples) # With some examples to reinforce structure


    def generate_node(self, text_input, node_memory=None):
        # Call the LLM to get raw nodes
        result = self.node_module(
            text_input=text_input,
            node_memory=node_memory or []
        )
        nodes = result.get("node_output", [])

        # If the LLM returns a JSON string, parse it
        if isinstance(nodes, str):
            try:
                parsed = json.loads(nodes)
                if isinstance(parsed, list):
                    nodes = parsed
                else:
                    raise ValueError("Parsed node output is not a list.")
            except Exception as e:
                print(f"Warning: Node output not valid JSON. Error: {e}")
                nodes = []

        # If have prior memory, merge new with old
        if node_memory:
            nodes = merge_memory_nodes(node_memory, nodes)

        # Return the unified list under the same key
        return {"node_output": nodes}


    def generate_edge(self, node_input):
        # For edge generation, the node_input is the dict of nodes that are output from generate_node
        result = self.edge_module(node_input=node_input)
        edges = result.get("edge_output", [])

        # Handle stringified JSON if returned as text
        if isinstance(edges, str):
            try:
                parsed = json.loads(edges)
                if isinstance(parsed, list):
                    edges = parsed
                else:
                    raise ValueError("Parsed edge output is not a list.")
            except Exception as e:
                print(f"Warning: Edge output not valid JSON. Error: {e}")
                edges = []

        return {"edge_output": edges}



class ClinicalDataExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(nodeClinicalDataExtract)

    def forward(self, content: str = "", atomic_sentences: list[str] = []):
        result = self.extractor(content=content, atomic_sentences=atomic_sentences)
        clinical_data = result.get("clinical_data", {})
        
        if not isinstance(clinical_data, dict):
            print("Warning: clinical_data is not a valid dictionary. Skipping.")
            clinical_data = {}

        return clinical_data


class BranchClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(branchClassify)

    def forward(self, content):
        return self.program(content=content)




# ---------------------------------------
# FEW-SHOT EXAMPLES - EDGE CONSTRUCTION
# ---------------------------------------

# Structure not coming out right with edges so doing these to reinforce output

edge_example_1 = dspy.Example(
    node_input=[{"node_id": "A"}, {"node_id": "B"}],
    edge_output=[{
        "edge_id": "A_to_B",
        "branch_flag": True,
        "content": "Severe headache led to CT finding of subdural hematoma.",
        "transition_event": {
            "trigger_type": "symptom_onset",
            "trigger_entities": ["C0018681", "C0038454"],  # e.g., UMLS CUIs
            "change_type": "progression",
            "target_domain": "imaging"
        }
    }]
).with_inputs("node_input")

edge_example_2 = dspy.Example(
    node_input=[{"node_id": "B"}, {"node_id": "C"}],
    edge_output=[{
        "edge_id": "B_to_C",
        "branch_flag": True,
        "content": "Rash developed after antibiotics.",
        "transition_event": {
            "trigger_type": "medication_change",
            "trigger_entities": ["C0003232", "C0037274"],  # Antibiotics, Rash
            "change_type": "adverse_effect",
            "target_domain": "symptom"
        }
    }]
).with_inputs("node_input")

edge_example_3 = dspy.Example(
    node_input=[{"node_id": "C"}, {"node_id": "D"}],
    edge_output=[{
        "edge_id": "C_to_D",
        "branch_flag": False,
        "content": "PET-CT showed regression of disease.",
        "transition_event": {
            "trigger_type": "interpretation",
            "trigger_entities": ["C0151744"],  # Regression
            "change_type": "resolution",
            "target_domain": "diagnosis"
        }
    }]
).with_inputs("node_input")

edge_fewshot_examples = [edge_example_1, edge_example_2, edge_example_3]



# =====================================
# FEW-SHOT EXAMPLES - BRANCH CLASSIFY
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


# Calling CSV with examples of content and bool pairs
branch_examples = load_branch_examples_from_csv("src/data/branch_data.csv")
random.shuffle(branch_examples)

trainset = branch_examples[:85]
devset   = branch_examples[85:]

# Search labeled examples to select best demos for few-shot prompts
teleprompter = BootstrapFewShot(
    metric=branching_accuracy,
    max_bootstrapped_demos=4,
    max_labeled_demos=40,
    max_rounds=1
)

# Try bootstrapping few-shot examples
teleprompter.compile(
    dspy.Predict(branchClassify),  # compile only returns optimized Predict, not needed here
    trainset=trainset
)

# Extract selected demos
selected_demos = teleprompter.get_params().get("demos", [])

if selected_demos:
    print("\nSelected few-shot demonstrations:") # Check if any selected via bootstrap
    for i, ex in enumerate(selected_demos):
        print(f"Example {i+1}:")
        print("  Inputs:", ex.inputs)
        print("  Outputs:", ex.outputs)
else: # if none selected, just select the first X examples
    num_examples = 2
    print("No few-shot demonstrations found.")
    print(f"Using fallback: selecting the first {num_examples} examples from trainset.")
    selected_demos = trainset[:num_examples]

# Final classifier using selected or fallback examples
optimizedBranchClassifier = dspy.Predict(branchClassify, examples=selected_demos)
    

# =====================================
# EXTRACTION PROCESS AND ORG FUNCTIONS
# =====================================


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


# Move these up to the right section later


# Filter out irrelevant content (author list, conclusion, references, etc.)
# Right now the model is too week and this is just devolving into summarization, so not going to use for now

class FilterIrrelevantSections(dspy.Signature):
    full_text: str = dspy.InputField(desc="The full text of a clinical case report, including any irrelevant sections like background, discussion, and references.")
    cleaned_text: str = dspy.OutputField(desc="The original text with only irrelevant sections (authors, references, conclusion etc.) removed. Do not summarize or paraphrase. Preserve all remaining formatting and content exactly.")


class FilterIrrelevantSectionsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(FilterIrrelevantSections)

    def forward(self, full_text: str):
        result = self.program(full_text=full_text)
        return result.cleaned_text


# Use dspy to chunk and and generate nodes intelligently
class ChunkedNodeGenerator(dspy.Signature):
    case_report: str = dspy.InputField(desc="Full clinical case report text")
    max_words_per_chunk: int = dspy.InputField(desc="Maximum number of words per chunk", default=250)
    max_chunks: int = dspy.InputField(desc="Maximum number of chunks to process (optional)", default=None)
    node_output: list[dict] = dspy.OutputField(desc="List of generated node dictionaries")


class ChunkingNodeModule(dspy.Module):
    def __init__(self, generator: NodeEdgeGenerate):
        super().__init__()
        self.generator = generator

    def forward(self, case_report: str, max_words_per_chunk: int, max_chunks: int = None):
        paragraphs = extract_paragraphs(case_report)

        # Build text chunks
        chunks = []
        current_chunk = []
        word_count = 0

        for para in paragraphs:
            p_text = para["paragraph"].strip()
            p_words = p_text.split()

            if len(p_words) > max_words_per_chunk:
                # Oversized paragraph – split up
                sub_chunk = []
                sub_count = 0
                for w in p_words:
                    sub_chunk.append(w)
                    sub_count += 1
                    if sub_count >= max_words_per_chunk:
                        chunks.append(" ".join(sub_chunk).strip())
                        sub_chunk = []
                        sub_count = 0
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk).strip())
                continue

            if word_count + len(p_words) <= max_words_per_chunk:
                current_chunk.append(p_text)
                word_count += len(p_words)
            else:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = [p_text]
                word_count = len(p_words)

        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())

        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        # Phase 1: sliding‐window memory across chunks
        memory_nodes: List[Dict] = []
        failed_chunks = 0
        start_time = time.time()

        for chunk in tqdm(chunks, desc="Generating nodes", unit="chunk"):
            try:
                out = self.generator.generate_node(
                    text_input=chunk,
                    node_memory=memory_nodes
                )
                memory_nodes = out["node_output"]
            except Exception as e:
                print(f"Warning: node generation failed on chunk. Error: {e}")
                failed_chunks += 1

        end_time = time.time()
        print(f"\nNode generation completed in {end_time - start_time:.2f} seconds.")
        print(f"Failed chunks: {failed_chunks}/{len(chunks)}")

        return {"node_output": memory_nodes}


"""

#For sentence chunking, integrate this
# Also get rid of the few shot examples for ephemeral becuase now we are just doing comparison
# Also do this moving window for edges as well because it's too confusing for the model; but only show two at a time and then content is what changed between the two

import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model: en_core_web_sm...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text, n=3):
    "Split text into sentences and join every n sentences into one string."
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return [' '.join(sentences[i:i+n]) for i in range(0, len(sentences), n)]

"""


# Creates a dynamic memory to give nodes context as it is constructing
def merge_memory_nodes(prev_nodes: List[Dict], new_nodes: List[Dict]) -> List[Dict]:
    """
    Merge `new_nodes` into `prev_nodes` by:
      1. Appending truly new nodes.
      2. For matching node_ids, concatenating content and merging clinical_data.
      3. Re-assigning node_ids (“A”, “B”, …) and step_indices.
    """
    # shallow-copy previous
    merged = [n.copy() for n in prev_nodes]
    lookup = {n["node_id"]: n for n in merged}

    for cand in new_nodes:
        cid = cand["node_id"]
        if cid in lookup:
            base = lookup[cid]
            # merge content
            if cand["content"] not in base["content"]:
                base["content"] += " " + cand["content"]
            # merge clinical_data
            for cat, items in cand.get("clinical_data", {}).items():
                bucket = base.setdefault("clinical_data", {}).setdefault(cat, [])
                for itm in items:
                    if itm not in bucket:
                        bucket.append(itm)
        else:
            merged.append(cand.copy())

    # re-index
    for idx, node in enumerate(merged):
        node["node_step_index"] = idx
        node["node_id"] = chr(ord("A") + idx)

    return merged


# Does final pass through completed sequence of nodes and helps to organize them


class ReorganizeNodes(dspy.Signature):
    """
    Take a full list of extracted nodes, deduplicate or merge any
    overlapping/fragmented entries, then reindex them cleanly.
    """
    nodes_sequence: List[Dict] = dspy.InputField(
        desc="List of node dicts to be cleaned and merged"
    )
    node_output: List[Dict] = dspy.OutputField(
        desc="Reorganized list of nodes (deduped, merged, re‐indexed A, B, C…)"
    )




# =====================================
# RUN PIPELINE
# =====================================

case_report = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")
#case_report = "A 64-year-old male with a history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history presented to the emergency department with progressive shortness of breath, dry cough, and unintentional weight loss over the past two months. He denied chest pain or hemoptysis."
#case_report = "A 64-year-old male with a past medical history of hypertension, type 2 diabetes mellitus, and a 40-pack-year smoking history initially presented to his primary care provider with a two-month history of progressive exertional dyspnea, dry cough, and unintentional weight loss. Initial outpatient labs, including a complete blood count and basic metabolic panel, were unremarkable. A chest X-ray revealed a left upper lobe opacity, prompting referral to pulmonology. High-resolution CT of the chest demonstrated a 4.2 cm spiculated mass in the left upper lobe with associated mediastinal lymphadenopathy. PET-CT confirmed hypermetabolic activity in the mass and mediastinal nodes without distant metastasis. Bronchoscopy with transbronchial biopsy was performed, and histopathology revealed poorly differentiated non-small cell lung carcinoma (NSCLC). Molecular testing showed no actionable mutations. The patient was staged as clinical stage IIB (T2bN1M0) and discussed at multidisciplinary tumor board. He was deemed a surgical candidate and underwent left upper lobectomy with mediastinal lymph node dissection via VATS. Pathology confirmed NSCLC with negative margins but 2/12 positive nodes, confirming stage IIB disease. He recovered well postoperatively without complications and was discharged home on postoperative day three. Following recovery, he was referred to medical oncology, and adjuvant cisplatin-based chemotherapy was initiated six weeks post-surgery. He completed four cycles of chemotherapy over three months without major adverse effects aside from mild fatigue and nausea. Surveillance imaging at three months post-treatment showed no evidence of disease recurrence. He continues routine follow-up every three months with thoracic surgery and oncology. The case highlights the importance of early symptom recognition, timely referral, and coordinated multidisciplinary care in managing operable lung cancer."
#case_report = "Testicular tumefaction is a common concern in urology. Most causes can be easily identified through anamnesis, clinical examination, blood tests, or ultrasonography. However, for testicular masses, differential diagnosis can be challenging. The 2022 WHO classification of tumors of the urinary system and the male genital organs lists 43 different types of testicular tumors, of which 8 are categorized as having unspecified, borderline, or uncertain behavior. Given that testicular cancers primarily affect young males, accurate diagnosis and assessment of pathological progression are crucial for determining the most appropriate therapeutic strategy. However, data on the management of many types of testicular tumors are scarce, and existing case studies are often only partially comparable. Consequently, it can be difficult to choose between a more radical treatment option and favoring the conservation of fertility and endocrine function, especially face-to-face to a young patient. In this clinical case, we share our experience with a 37-year-old patient with a multifocal, bilateral testicular LCCSCT presenting as painless testicular masses that took an unexpected course, resulting in a fatal outcome. A 37-year-old man was referred to our institution in December 2015. He noticed bilateral painless testicular masses. Physical examination revealed no gynecomastia or skin anomalies. Ultrasound showed bilateral macro-orchitis with multiple intratesticular hyperechoic lesions with acoustic shadowing. These lesions were 2.4 cm and 2.1 cm on the left and right testicles, respectively. A pelvic MRI was performed, showing bilateral intratesticular lesions with clear hypointense T1 and T2 signals, which take intense contrast enhancement after gadolinium injection. The MRI revealed 10 lesions on the left testicle and 5 lesions on the right. Serum alpha-fetoprotein (AFP), beta-hCG, LDH, testosterone, estradiol, and gonadotropin levels were in normal range. A surgical testicular exploration was performed, and a right testicular nodule was enucleated and sent to pathology. Histology revealed a benign large-cell calcifying Sertoli cell tumor under 5 cm, with no mitosis, cytological atypia, vascular permeation, or necrosis. Immunohistochemistry was positive for vimentin, calretinin, inhibin, and cytokeratin AE1/AE3. Given the benign histology, radical orchiectomy was not pursued; instead, regular follow-up was initiated. Follow-up consisted of testicular exams and sonography every 6 months for 2 years, then annually. No genetic testing was performed due to absence of syndromic features or family history. Endocrine tests ruled out glucose, thyroid, adrenal, and pheochromocytoma abnormalities. In June 2021, testicular exam revealed left-sided induration; MRI confirmed tumor infiltration into the spermatic cord. A left radical orchiectomy was performed. Pathology revealed a multifocal large-cell calcifying Sertoli cell tumor (largest 6.5 cm) with spermatic cord invasion and neoplastic vascular permeation. CT scan showed para-aortic lymphadenopathy, pulmonary nodules, and sclerotic spinal lesions. PET confirmed left inguinal and iliac lymphadenopathy and a penile lesion, but no lung uptake. The patient underwent lymph node dissection and right radical orchiectomy, revealing metastatic disease and multifocal LCCSCT. By November 2021, local extension into the penis and inguinal canal had occurred, with widespread metastases by December. Chemotherapy with vinblastine, cisplatin, and ifosfamide was ineffective. Paclitaxel was then given, but the disease progressed. The patient enrolled in a trial with Axitinib and pazopanib but died 7 months later in palliative care. This case highlights that despite benign histological features, LCCSCTs can be aggressive, especially in sporadic bilateral cases. Although most are benign, large, multifocal, sporadic tumors may behave malignantly. WHO classification lists size >5 cm, necrosis, pleomorphism, and invasive growth as risk factors. Some studies suggest lowering size threshold to >2.4 cm. Diagnosis requires histopathology and immunohistochemistry. Sertoli cell tumors express inhibin, calretinin, vimentin, and keratin, but typically lack beta-catenin nuclear staining. Testis-sparing surgery (TSS) may be considered for tumors with 0–1 risk factor. Retroperitoneal lymph node dissection (RPLND) offers benefit for regional spread, but systemic chemotherapy and radiation have low efficacy. Bilateral orchiectomy should have been the initial strategy in this case."


# Start lobal timer
global_start = time.time()

# Instantiate the filter module
filter_module = FilterIrrelevantSectionsModule()
# Run it and get the cleaned case report text
cleaned_text = filter_module(full_text=case_report)
# Print cleaned result for debugging
print("\n🧹 Cleaned Case Report (filtered output):")
print("=" * 60)
print(cleaned_text)
print("=" * 60)


generator = NodeEdgeGenerate()
chunked_node_module = ChunkingNodeModule(generator) # For node, requires chunking


"""
# Generate nodes in chunks to avoid overload for smaller models, use chunk module
node_result = chunked_node_module(case_report=case_report, max_words_per_chunk=450, max_chunks=None)
nodes_obj = node_result["node_output"]

# Generate edges once all nodes are collected
# The input into the edge generator is simply the list of dict of nodes from node_generate
edge_result = generator.generate_edge(node_input=nodes_obj)
edges_obj = edge_result["edge_output"]
"""

# Memory-enabled extraction of nodes
node_result = chunked_node_module(
    case_report=case_report,
    max_words_per_chunk=450,
    max_chunks=None
)
nodes_obj = node_result["node_output"]

# Pure-Python final merge
nodes_obj = merge_memory_nodes([], nodes_obj)

# LLM-based reorg
nodes_obj = dspy.Predict(ReorganizeNodes)(nodes_sequence=nodes_obj).node_output

# Edge generation off the clean nodes
edge_result = generator.generate_edge(node_input=nodes_obj)
edges_obj   = edge_result["edge_output"]


# Loop through nodes and edges and extract atomic statements from content output

"""
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


# Use atomic statements and content to generate clinical data dict to append to nodes

clinical_data_extractor = ClinicalDataExtractor()

for node in nodes_obj:
    content_str = node.get("content", "")
    #atomic_sents = decompose_content_to_atomic_statements(content_str)

    # You can comment out `content=content_str` if you want atomic_sents only
    clinical_data = clinical_data_extractor(
        content=content_str,
        atomic_sentences=None # Currently none because we have atomic_sents commented out jut to reduce run time
    )
    
    node["clinical_data"] = clinical_data  # append to original node

print("=" * 60)
print("Nodes (with clinical data):")
print("=" * 60)

for i, node in enumerate(nodes_obj):
    print(f"\n Node {i+1}:")
    pprint(node)
    print("-" * 60)


print("=" * 60)
print("Edges:")
print("=" * 60)

for i, edge in enumerate(edges_obj):
    print(f"\n Edge {i+1}: edge_id = {edge.get('edge_id', 'N/A')}")
    pprint(edge)
    print("-" * 60)



# Evaluate and update the first edge that triggers a branch
if edges_obj:
    branch_triggered = False

    for edge in edges_obj:
        # Extract the content used to evaluate branching logic
        transition_content = edge.get("content", "")
        if not transition_content:
            continue  # Skip edge if no content is available

        # Run branching classifier on this specific edge content
        branch_result = optimizedBranchClassifier(content=transition_content)

        # Logging evaluation info
        print(f"\nEvaluating edge: {edge.get('edge_id', 'unknown')}")
        print(f"Content: {transition_content}")
        print(f"Branch decision: {branch_result.branch_bool}")

        if branch_result.branch_bool:
            # Update branch_flag only for the first edge that qualifies
            print(f"Branch triggered — setting branch_flag = True on edge {edge['edge_id']}")
            edge['branch_flag'] = True
            branch_triggered = True
            break  # Stop after marking the first branching edge

    if not branch_triggered:
        print("No edges met branching criteria.")
else:
    print("No edges available to evaluate.")



# End global timer
global_end = time.time()
print(f"\n=== Full DAG generation pipeline completed in {global_end - global_start:.2f} seconds ===")