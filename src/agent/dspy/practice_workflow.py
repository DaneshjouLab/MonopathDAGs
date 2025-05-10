""" 
This file is an example on how to use certain things in dspy and python for Anson, ignore for code base, 




"""
# TODO set up workflow for the dspy setyup, 
# TODO create workflow for extraction. 

from .set_up_dspy import setup
import dspy


from ...data.data_processors.pdf_to_text import extract_text_from_pdf


#here 
fullText = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")

class Text2Data(dspy.Signature):
    """
    /*
You are an assistant that converts clinical case narratives into dynamic Directed Acyclic Graphs (DAGs) using a data model called N-dData.

Each DAG consists of:
- Nodes = snapshots of the patient's state.
- Edges = transitions between those states.

Your job is to generate or validate JSON structures for nodes and edges.

---

1. N-dData NODE SPECIFICATION

Each node represents the patient at a specific point in time or logical step.

Top-level fields:
- node_id (required): Unique identifier (Use capital letter of alphabet \"A"\ and so on and so forth and move to \"AA"\ after all letters are used up)
- step_index: Integer for sequence ordering
- timestamp (optional, only include if clearly given): ISO 8601 datetime (e.g., \"2025-03-01T10:00:00Z\")
- branch_label (optional): String or boolean label for branches/merges
- confidence (optional): Float from 0–1 for certainty, particularly from LLM outputs
- commentary (optional): Free-text interpretation or summary

Node content typically lives inside a `data` object and may include:
- demographics
- conditions
- treatments
- observations
- labs
- metadata (optional): Include journal ID, DOI, corpus timestamp, and schema version

Terminology guidance:
- Use OMOP-standard concepts when possible for consistency and interoperability.
- If a concept isn't covered by OMOP, use clear, logical labeling.

Explicit guidelines for Node Boundaries:
- Define nodes based on distinct clinical events or logical steps.
- Combine simultaneous lab and imaging results into the same node.
- Use separate nodes when events are clearly sequential or clinically distinct.

---

2. N-dData EDGE SPECIFICATION

Each edge represents a change from one node to another.

Top-level fields:
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

Explicit handling for edge types:
- Branch only for actionable physiological or clinical events.
- Informational updates (no immediate clinical impact) should NOT branch.
- Rejoin explicitly when patient state aligns structurally with prior nodes.

---

3. BRANCHING LOGIC

Branches arise when physiologic changes or complications aren't part of the main pathway but impact patient states. Mark side paths clearly:
- Edge leading to branch: branch_flag = true
- Nodes in branch: use branch_label clearly distinguishing alternate tracks.
- Rejoin explicitly when interventions successfully revert to previous stable states.
- Modular structure for easy modification or removal.

---

4. USAGE INSTRUCTIONS

After reviewing a clinical case:
- Represent each patient snapshot clearly as a node.
- Explicitly define node boundaries based on clinical events or logical segments.
- Describe transitions between snapshots explicitly with edges.
- Explicitly identify changes in edges, annotating ambiguity and temporal uncertainties.
- Translate LLM-generated annotations consistently into nodes and edges.
- Validate DAG against the original narrative for:
  - Acyclicity
  - Correct temporal ordering
  - Semantic consistency
- Include comprehensive metadata and schema versioning for traceability.
- Adhere to OMOP when possible.
- Clearly mark branches and rejoins.
- Aim for clarity, accuracy, interpretability, and logical consistency.

---

*/
"""

    clinical_case_pdf_text = dspy.InputField()
    nodes = dspy.OutputField(desc="Nodes object in JSON format.")





lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
#lm=setup(config)



dspy.configure(lm = lm, adapter = dspy.JSONAdapter())

prog = dspy.ChainOfThought(Text2Data)



#  lets continue forward on this, 



nodes = prog(
  clinical_case_pdf_text = ""




).nodes



# filepath that leads to pdf-> text from that-> 


