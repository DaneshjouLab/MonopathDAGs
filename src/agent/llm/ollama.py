import os
from typing import List, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import dspy
from dotenv import load_dotenv
# ========== LLM Configuration ==========

load_dotenv(".config/.env")

gemini_api_key = os.environ.get("GEMINI_APIKEY")
llm_timeline = lm = dspy.LM('gemini/gemini-2.0-flash', api_key=gemini_api_key,temperature=0.21,cache=False)
# dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='',cache=False)
llm_extractor = llm_timeline
# dspy.LM('ollama_chat/llama3.3', api_base='http://localhost:11434', api_key='')

# ========== DSPy Signatures ==========

@dataclass
class PatientTimeline(dspy.Signature):
    """
    Your task is to extract structured patient timeline from a clinical paragraph. First evaluate if the information is part of the clinical case, or just introduction.
    Your goal is to verify that it is a part of the clinical case and keep patient management and care in order. 
    your goal is to include everything including lab findings
    INCLUDE ALL LAB FINDINGS VERBATIM AND ALL IMAGING FINDINGS VERBATIM
    ONLY INCLUDE PATIENT DATA
    DO NOT BULLET POINT, this is a str output, keep specific lab and imaging in as well as procedure values. No explanations
    """
    paragraph: str = dspy.InputField(desc="Paragraph of clinical text to extract patient information from.")
    previous_memory: List[str] = dspy.InputField(desc="List of patient history extracted so far.")
    pt_timeline: str = dspy.OutputField(desc="Concatenated highly specific human-readable patient timeline including new events extracted in order.")

@dataclass
class ExtractCancerFacts(dspy.Signature):
    """
    You are given a full patient timeline generated from a clinical case report.
    Your task is to extract structured cancer-related findings.

    Specifically:
    - Identify if cancer is mentioned at all.
    - List general cancer mentions (e.g., 'malignancy', 'cancer').
    - Extract specific cancer diagnoses (e.g., 'pancreatic adenocarcinoma').
    - Determine if metastases are described.
    - If so, extract each metastasis as a pair: (cancer type, metastatic site).

    Do not infer. Use only explicit evidence from the patient timeline.
    """

    patient_timeline: List[str] = dspy.InputField(
        desc="List of timeline entries capturing patient clinical events."
    )
    cancers: List[str] = dspy.OutputField(
        desc="General cancer mentions (e.g., 'cancer', 'tumor', 'malignancy')."
    )
    specific_cancers: List[str] = dspy.OutputField(
        desc="Specific cancer diagnoses (e.g., 'glioblastoma', 'breast carcinoma')."
    )
    has_metastasis: bool = dspy.OutputField(
        desc="True if metastatic spread is described; False otherwise."
    )
    metastasis_locations: List[Tuple[str, str]] = dspy.OutputField(
        desc="List of (cancer type, metastasis site) pairs from the timeline."
    )

# ========== Preprocessing ==========

def preprocess_pmc_article_text(html_path: str) -> str:
    def is_stop_section(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in ("references", "bibliography"))

    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")

    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    for abstract_tag in soup.find_all(["div", "section"], class_=lambda c: c and "abstract" in c.lower()):
        abstract_tag.decompose()

    body = soup.find("body") or soup.find("article") or soup.find("div", class_="body")
    if body is None:
        body = soup

    lines = []
    for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table"]):
        text = tag.get_text(strip=True)
        if not text:
            continue
        if is_stop_section(text):
            break
        if tag.name == "table":
            table_text = []
            for row in tag.find_all("tr"):
                row_text = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
                if row_text:
                    table_text.append("\t".join(row_text))
            if table_text:
                lines.append("\n".join(table_text))
        else:
            lines.append(text)

    return "\n\n".join(lines)

# ========== Paragraph Splitting ==========

def split_into_sentences(text: str, n: int = 3) -> List[str]:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return [' '.join(sentences[i:i + n]) for i in range(0, len(sentences), n)]

def generate_paragraphs(raw_text: str, split_length: int = 10) -> List[str]:
    return split_into_sentences(raw_text, split_length)

# ========== Timeline Generation ==========

def generate_patient_timeline(paragraphs: List[str]) -> List[str]:
    dspy.configure(lm=llm_timeline)
    predictor = dspy.Predict(PatientTimeline)

    memory: List[str] = []
    for para in paragraphs:
        result = predictor(paragraph=para, previous_memory=memory[-3:])
        if result.pt_timeline.strip():
            memory.append(result.pt_timeline.strip())
            print("-" * 30)
            print(result.pt_timeline.strip())
    return memory

# ========== Cancer Fact Extraction ==========

def extract_cancer_information(timeline_text: List[str]) -> dict:
    with dspy.context(lm=llm_extractor):
        extractor = dspy.Predict(ExtractCancerFacts)
        result = extractor(patient_timeline=timeline_text)
        return {
            "cancers": result.cancers,
            "specific_cancers": result.specific_cancers,
            "has_metastasis": result.has_metastasis,
            "metastasis_locations": result.metastasis_locations,
        }

# ========== Full Pipeline ==========

def process_article_for_extraction(html_path: str) -> dict:
    raw_text = preprocess_pmc_article_text(html_path)
    paragraphs = generate_paragraphs(raw_text, split_length=10)
    timeline = generate_patient_timeline(paragraphs)
    structured_output = extract_cancer_information(timeline)
    return {
        "timeline": timeline,
        "structured": structured_output
    }

# ========== CLI Debug ==========
import csv
import os
from pathlib import Path
from typing import List

import pandas as pd


# === File paths ===
METADATA_FILE = Path("./webapp/static/graphs/mapping/graph_metadata.csv")
OUTPUT_FILE = Path("./webapp/static/graphs/mapping/graph_html_metadata.csv")
MAX_TOTAL = 546

# === Output Schema ===
OUTPUT_FIELDS = [
    "graph_id", "source_file",
    "timeline_1", "timeline_2", "timeline_3",
    "cancers", "specific_cancers", "has_metastasis", "metastasis_locations"
]


def load_existing_output() -> set:
    """Return a set of graph_ids already processed in the output file."""
    if not OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    return set(df["graph_id"].tolist())


def load_metadata() -> pd.DataFrame:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata input: {METADATA_FILE}")
    return pd.read_csv(METADATA_FILE)


import json

def process_row(row: dict) -> dict:
    html_path = row["source_file"]
    graph_id = row["graph_id"]
    file_name = Path(html_path).name

    try:
        result = process_article_for_extraction(html_path)
        timeline = result["timeline"][-3:]
        timeline += [""] * (3 - len(timeline))  # pad to 3
        cancer_info = result["structured"]

        return {
            "graph_id": graph_id,
            "source_file": file_name,
            "timeline_1": timeline[0],
            "timeline_2": timeline[1],
            "timeline_3": timeline[2],
            "cancers": json.dumps(cancer_info["cancers"]),
            "specific_cancers": json.dumps(cancer_info["specific_cancers"]),
            "has_metastasis": cancer_info["has_metastasis"],
            "metastasis_locations": json.dumps(cancer_info["metastasis_locations"])  # List[Tuple[str, str]]
        }

    except Exception as e:
        print(f"❌ Failed to process {graph_id}: {e}")
        return None



def write_new_rows(rows: List[dict]):
    write_header = not OUTPUT_FILE.exists()
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
from tqdm import tqdm

def generate_output_table():
    metadata_df = load_metadata()
    processed_ids = load_existing_output()
    processed_count = len(processed_ids)

    # Filter and limit rows to process
    remaining_rows = metadata_df[~metadata_df["graph_id"].isin(processed_ids)]
    remaining_rows = remaining_rows.head(MAX_TOTAL - processed_count)

    for _, row in tqdm(remaining_rows.iterrows(), total=len(remaining_rows), desc="Processing articles"):
        graph_id = row["graph_id"]

        result = process_row(row)
        if result:
            write_new_rows([result])  # write immediately
            processed_count += 1
            tqdm.write(f"✅ Wrote row for {graph_id}")
        else:
            tqdm.write(f"⚠️ Skipped {graph_id} due to processing error")


if __name__ == "__main__":
    generate_output_table()
