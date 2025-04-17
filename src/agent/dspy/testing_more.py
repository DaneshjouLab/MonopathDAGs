from typing import List, Dict
import dspy
import dspy.predict

# 1. Load text from PDF
from src.data.data_processors.pdf_to_text import extract_text_from_pdf
raw_text = extract_text_from_pdf("./samples/pdfs/am_journal_case_reports_2024.pdf")

# 2. Use paragraph extractor
def extract_paragraphs(text: str) -> List[dict]:
    import re
    section_headers = [
        "Abstract", "Introduction", "Case Presentation", "Case Description",
        "Investigation", "Diagnosis", "Treatment", "Follow-up", "Outcome",
        "Discussion", "Conclusion", "Background", "Clinical Findings",
        "Therapeutic Intervention"
    ]
    section_pattern = re.compile(rf"^\s*({'|'.join(section_headers)}):?\s*$", re.IGNORECASE)
    current_section = "Unknown"
    blocks = []
    raw_paragraphs = re.split(r"\n\s*\n", text)

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        header_match = section_pattern.match(para)
        if header_match:
            current_section = header_match.group(1).title()
            continue
        if len(para.split()) < 5:
            continue
        blocks.append({
            "section": current_section,
            "paragraph": para
        })
    return blocks

paragraphs = extract_paragraphs(raw_text)

# 3. Configure DSPy LLM and module
lm = dspy.LM('ollama_chat/llama3.3', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

class clinicalEventExtract(dspy.Signature):
    """
    You are a medical extraction assistant.

    Your task is to read one paragraph from a clinical case report and extract a list of distinct, ordered, and atomic clinical events that relate directly to the patient.

    You must include:
    - Events that are explicitly stated or reasonably inferred.
    - Clinical facts, diagnostic actions, interventions, interpretations, adverse effects, and administrative decisions — all must be patient-specific.
    - Ephemeral changes (e.g., lab value changes, symptoms) and non-ephemeral state transitions (e.g., diagnosis, progression, remission).
    - Any referenced prior medical events (e.g., past surgeries) — these should appear early in the timeline.
    - All measurements, units, time references, and agents (if stated).
    - Ambiguous, indeterminate, or speculative findings are valid and should be included if they influence the patient timeline.

    For each extracted event, return a dictionary with:
    - step_index (int): Order in which the event occurred within this paragraph
    - description (str): A clear, atomic statement of what happened
    - temporal_reference (optional, str): Any stated or implied time expression (e.g., "day 5", "two weeks later")
    - event_type (optional, str): A tag like "diagnosis", "treatment", "observation", "procedure", "response", "administrative", etc.
    - agents (optional, list of str): All named roles or actors involved (e.g., "oncologist", "radiologist")
    - value (optional): Any quantitative or qualitative measurement (e.g., "CEA 108 ng/mL", "tumor 1.2 cm")
    - confidence (float): Confidence in the accuracy of this extraction (0–1)
    - source_sentence (str): Exact sentence or phrase this event came from

    Each event should be atomic — if one sentence includes multiple actions or outcomes, split them into multiple entries.

    Do not include general background facts, only information about this specific patient.

    Do not normalize or rephrase time references — include them as stated.
    """

    report_text: str = dspy.InputField(desc="A paragraph of clinical narrative from a case report")
    events: List[Dict] = dspy.OutputField(desc="List of distinct, ordered, atomic clinical events")



# what do you feel would work step wise, =->
extractor = dspy.Predict(clinicalEventExtract)

# result2=extractor(raw_text)
# 4. Run LLM on each paragraph
# for i, p in enumerate(paragraphs):
#     # print(f"\n[{p['section']}] Paragraph {i+1}:\n{p['paragraph']}\n")
#     result = extractor(report_text=p['paragraph'])
#     print(result)
#     for event in result.events:
#         print(event)
        # print(f"- ({event['step_index']}) {event['description']}")
# the following should be done in this cs

# decompose_signature-> outputs should be decomposable-> 
# once it is decomposable might be easier, sentence by sentence
# byte pair encoding versoin of composable facts, 
#

class decomposeToAtomicSentences(dspy.Signature):
    """
    Given a single complex sentence from a clinical case report,
    return a list of atomic clinical sentences.

    Each atomic sentence must:
    - Contain only one clinical event, action, or state
    - Be clear, self-contained, and refer only to the patient
    - Preserve all quantitative values, agents, and time references
    - Be directly traceable to the original sentence (no hallucination)
    - Be suitable for downstream structuring into graph nodes or facts

    Do NOT summarize, group, or abstract across multiple concepts.

    If the sentence is already atomic, return it as a single-item list.

    Output: list of atomic clinical sentences (strings)
    """

    sentence: str = dspy.InputField(desc="A complex clinical sentence from a case report")
    context: str = dspy.InputField(optional=True, desc="Optional failure reason or context to improve decomposition")
    atomic_sentences: List[str] = dspy.OutputField(desc="List of decomposed atomic sentences")
# atomic_sent=dspy.Predict(decomposeToAtomicSentences)
decomposeToAtomicSentences.__doc__ = """
    Given a single complex sentence from a clinical case report,
    return a list of atomic clinical sentences.

    Each atomic sentence must:
    - Contain only one clinical event, action, or state
    - Be clear, self-contained, and refer only to the patient
    - Preserve all quantitative values, agents, and time references
    - Be directly traceable to the original sentence (no hallucination)
    - Be suitable for downstream structuring into graph nodes or facts
    Do NOT summarize, group, or abstract across multiple concepts.

    If the sentence is already atomic, return it as a single-item list.

If this sentence was previously marked as non-atomic because: {context}, be especially careful to avoid that issue.
"""


class checkIfAtomic(dspy.Signature):
    """
    Determine whether a clinical sentence is atomic.

    A sentence is atomic if:
    - It describes exactly one clinical event, action, or state affecting the patient
    - It does not combine multiple separate actions or changes
    - It is self-contained and unambiguous
    - It includes relevant detail (e.g., time, agent, value) if present in the original source

    Return:
    - is_atomic (bool): True if sentence is atomic
    - reason (str): Why it passed or failed
    """

    sentence: str = dspy.InputField()
    is_atomic: bool = dspy.OutputField()
    reason: str = dspy.OutputField()
checkIfAtomic.__doc__ = """
You are a clinical validator whose job is to strictly determine whether a sentence is atomic.

A sentence is **atomic** ONLY IF ALL of the following are true:
1. It describes exactly ONE clinical action, state, event, or observation.
2. It does NOT include multiple verbs or actions ("and", "then", "also", "while", "which", etc.).
3. It is NOT a compound sentence.
4. It is self-contained and clearly refers to a single patient-specific event.
5. It includes detail about the agent, value, or time if such detail is present in the source.

❗If there are multiple actions, effects, causes, or interpretations — it is NOT atomic.

Be harsh. If you're uncertain, the answer is FALSE.

Return:
- is_atomic (bool): True ONLY if the sentence is minimal and describes exactly one clinical change.
- reason (str): Explain why it is or isn’t atomic.

Examples:
- "The patient underwent a PET scan." → ✅ atomic
- "The scan showed lesions and chemotherapy was started." → ❌ not atomic (two actions)
- "He developed fever while on pembrolizumab." → ❌ not atomic (symptom + treatment context)
"""

decomposeToAtomicSentences_module = dspy.Predict(decomposeToAtomicSentences)
checkIfAtomic_module = dspy.Predict(checkIfAtomic)

def recursively_decompose_to_atomic_sentences(sentence: str, depth: int = 0, max_depth: int = 2, reason: str = "") -> List[str]:
    if depth > max_depth:
        return [sentence]

    # Step 1: Decompose current sentence (add reason to prompt if needed)
    if reason:
        prompt_override = f"The sentence was rejected as atomic because: {reason}"
        atomic_candidates = decomposeToAtomicSentences_module(sentence=sentence, context=prompt_override).atomic_sentences
    else:
        atomic_candidates = decomposeToAtomicSentences_module(sentence=sentence).atomic_sentences

    output = []
    for atomic_candidate in atomic_candidates:
        check = checkIfAtomic_module(sentence=atomic_candidate)

        if check.is_atomic:
            output.append(atomic_candidate)
        else:
            print(f"↪ Re-decomposing: {atomic_candidate} – {check.reason}")
            decomposed = recursively_decompose_to_atomic_sentences(
                sentence=atomic_candidate,
                depth=depth + 1,
                max_depth=max_depth,
                reason=check.reason
            )
            output.extend(decomposed)

    return output


input_sentence = """ At 17 years post-initial diagnosis, he developed 
an oligometastatic liver lesion and a lung lesion and underwent partial hepatectomy"""

# this is not working, 
atomic_results = recursively_decompose_to_atomic_sentences(input_sentence)

for i, s in enumerate(atomic_results, 1):
    print(f"{i}. {s}")

