"""Doctor / Patient consultation loop for 3MDBench cases (text-only).

Prompts and control flow ported verbatim from the upstream 3mdbench repo
(agents/doctor_agent.py, agents/patient_agent.py, utils/dialogue_utils.py,
benchmarking/run_dialogue.py, benchmarking/run_diagnoses_obtaining.py) --
only the image-handling parts are dropped, per the decision to run 3MDBench
text-only. Unlike AI Hospital there is no Reporter/Examiner role here: just
Doctor and Patient, and Patient speaks first.
"""
from __future__ import annotations

from typing import Any, Dict, List

from smart_trial.models.model_client import ModelClient

MAX_DIAG_LEN = 28  # same constant as upstream utils/dialogue_utils.py

DIAGNOSIS_LIST = [
    "eczema", "hives", "herpes", "allergic contact dermatitis", "contact dermatitis",
    "seborrheic dermatitis", "rosacea", "conjunctivitis", "tonsillitis", "chronic lichen",
    "keratosis pilaris", "molluscum contagiosum", "lichen planus", "periodontitis", "caries",
    "psoriasis", "ingrown nail", "stye", "acne", "onychomycosis", "seborrheic keratosis",
    "chalazion", "vitiligo", "shingles", "mycosis", "actinic keratosis", "onycholysis",
    "chickenpox", "dental calculus", "warts", "stomatitis", "abscess", "gingivitis",
    "nail dystrophy",
]

DOCTOR_SYSTEM_PROMPT = (
    "You are a highly experienced doctor with extensive medical expertise in a simulated world.\n"
    "Your task is to diagnose the patient and provide the recommendations.\n"
    f"The diagnosis should be chosen from this list: {DIAGNOSIS_LIST}\n\n"
    "In a conversation, you need to provide a single diagnosis. If you do not have sufficient "
    "information yet, then inquire this information from the patient. Ask only one question at a time."
)

CHARACTERS_DICT = {
    "symptom_description": {
        "sanguine": "Actively describes symptoms with vivid details, often adding anecdotes or humor. Occasionally minimizes severity, focusing on positive aspects.",
        "choleric": "Direct and assertive when describing symptoms. Complains openly and expects swift solutions. May express frustration if not understood.",
        "melancholic": "Provides detailed and precise descriptions of symptoms but may emphasize severity or worry about potential complications.",
        "phlegmatic": "Provides information only when directly asked, keeping responses brief and to the point. Tends to minimize the significance of symptoms or avoid adding extra details unless specifically encouraged.",
    },
    "asking_questions": {
        "sanguine": "Asks multiple questions, curious about the doctor’s opinion and alternative treatments. Engages in a conversational tone.",
        "choleric": "Focused on practical outcomes. Asks direct, outcome-oriented questions and expects clear answers.",
        "melancholic": "Inquires about details of the diagnosis and treatment, often seeking reassurance or clarification.",
        "phlegmatic": "Rarely asks questions, instead passively listens to the doctor’s advice. Prefers straightforward communication.",
    },
    "communication_style": {
        "sanguine": "Frequently veers into unrelated topics, sharing personal stories or jokes. Creates a lively and informal atmosphere.",
        "choleric": "Maintains a focused and authoritative tone. Keeps the conversation goal-oriented, occasionally cutting off unnecessary details.",
        "melancholic": "Stays on-topic but may overanalyze the situation. Occasionally mentions worries or hypothetical scenarios.",
        "phlegmatic": "Reserved and calm. Sticks to the topic without adding extra details. Prefers to keep the interaction brief.",
    },
    "attitude_towards_treatment": {
        "sanguine": "Open to treatment but may prefer methods perceived as 'natural' or 'easy.' May express interest in lifestyle adjustments over medication.",
        "choleric": "Prefers fast-acting solutions. Advocates for specific treatments, often insisting on personal preferences.",
        "melancholic": "Accepts treatment but with hesitation. May overthink side effects and require additional reassurance.",
        "phlegmatic": "Accepts advice without much questioning. Generally compliant but prefers minimal intervention.",
    },
    "emotional_involvement": {
        "sanguine": "Expresses emotions openly, often switching between optimism and slight concern. Rarely appears overly worried.",
        "choleric": "Displays frustration or impatience if progress is slow. May get irritated when things don’t go their way.",
        "melancholic": "Highly emotionally involved, often expressing worry, fear, or sadness about their condition.",
        "phlegmatic": "Appears emotionally detached or neutral, rarely showing visible concern about their health.",
    },
}

EXTRACTION_SYSTEM_PROMPT = """
You are a text analysis engine that processes doctor-patient consultation transcripts. Your task is to identify and extract the final diagnosis that the doctor has decided to assign to the patient. Follow these instructions carefully:

1. Identify the Relevant Sentence:
   - Search the entire transcript for the sentence in which the doctor explicitly communicates the final diagnosis.
   - Note that doctors can express diagnoses in many different ways; it does not have to be in the form "your diagnosis is...". Look for alternative phrasing, searching for other wording that indicates a definitive conclusion.
   - Only extract the sentence if you are confident it contains the final diagnosis, not merely a provisional or hypothetical opinion.

2. Extract the Diagnosis:
   - From the identified sentence, extract the diagnosis. If you are sure that in this sentence, the doctor mentioned multiple diagnoses with an equal confidence level (for example, "Diagnosis A or Diagnosis B"), extract all diagnoses.
   - Ensure that the diagnoses you extract are the ones the doctor confirms as final.
   - Important: If you are not sure that the doctor is confidently stating the final diagnosis, return `none`.

3. Output Format:
   - Provide the extracted diagnosis or diagnoses as a comma-separated list, without any patticles like "or".
   - Do not include any additional text, context, or commentary in your output.

Examples:
- If the sentence is: "After reviewing your tests, I have concluded that you have pneumonia," your output should be:
  `pneumonia`
- If the sentence is: "Your condition is either bronchitis or pneumonia," your output should be:
  `bronchitis, pneumonia`
- If no sentence confidently states a final diagnosis, or if you are not sure that the doctor is expressing a confident final diagnosis, your output should be:
  `none`

Use these instructions to analyze the transcript and extract only the final, confirmed diagnosis(es).
"""


def build_patient_system_prompt(case: Dict[str, Any]) -> str:
    char = (case.get("personality") or "phlegmatic").strip().lower()
    if char not in CHARACTERS_DICT["symptom_description"]:
        char = "phlegmatic"
    return (
        "You are a patient at an online consultation with the general practitioner.\n\n"
        "Your main complaint:\n"
        f"{case.get('general_complaint')}\n"
        "Additional symptoms:\n"
        f"{case.get('complaints')}\n\n"
        f"Your personality corresponds to {char}. That includes:\n"
        f"- {CHARACTERS_DICT['symptom_description'][char]}\n"
        f"- {CHARACTERS_DICT['asking_questions'][char]}\n"
        f"- {CHARACTERS_DICT['attitude_towards_treatment'][char]}\n"
        f"- {CHARACTERS_DICT['emotional_involvement'][char]}\n"
        f"- {CHARACTERS_DICT['communication_style'][char]}\n\n"
        "During the conversation you should behave according to your personality.\n\n"
        "Your goal in this conversation is to understand the cause of the symptoms, the diagnosis, and the treatment.\n"
        "You cannot self-diagnose, you only tell the doctor about your symptoms.\n"
        "Do not provide any analysis, inference, or implications.\n"
        "Use only the information that is provided in the symptoms and complaints list or which you can infer from it.\n\n"
        "Start the conversation with the ONLY main complaint. Remember that you are typing, thus, keep your texts short.\n"
        "If you think that the conversation can be finished and you obtained all the needed information from the doctor, respond with BREAK.\n"
        "If you have already said thanks to the doctor and there are no new questions, finish the dialogue by responding with BREAK.\n"
        "If you said goodbyes to the doctor, finish the dialogue by responding with BREAK."
    )


def run_consultation(
    case: Dict[str, Any],
    *,
    doctor_client: ModelClient,
    patient_client: ModelClient,
    temperature: float = 0.6,
) -> Dict[str, Any]:
    """Run one Doctor/Patient consultation. Patient speaks first.

    Returns {"dialog_history": [...], "dialogue_ended": bool}.
    """
    patient_system = build_patient_system_prompt(case)

    doctor_history: List[Dict[str, str]] = []
    patient_history: List[Dict[str, str]] = []

    # Upstream calls the Patient agent with no user turn at all (system prompt
    # alone tells it to open with the main complaint). Most chat APIs require
    # at least one message, so we add a minimal generic kickoff turn.
    kickoff = [{"role": "user", "content": "Begin the conversation."}]
    patient_utterance = patient_client.chat(kickoff, system_prompt=patient_system, temperature=temperature)
    patient_history.append({"role": "user", "content": "Begin the conversation."})
    patient_history.append({"role": "assistant", "content": patient_utterance})

    dialog_history: List[Dict[str, Any]] = [{"turn": 0, "role": "Patient", "content": patient_utterance}]
    dialogue_ended = False

    for turn in range(1, MAX_DIAG_LEN // 2 + 1):
        doctor_history.append({"role": "user", "content": patient_utterance})
        doctor_utterance = doctor_client.chat(
            doctor_history, system_prompt=DOCTOR_SYSTEM_PROMPT, temperature=temperature
        )
        doctor_history.append({"role": "assistant", "content": doctor_utterance})
        dialog_history.append({"turn": turn, "role": "Doctor", "content": doctor_utterance})

        patient_history.append({"role": "user", "content": doctor_utterance})
        patient_utterance = patient_client.chat(
            patient_history, system_prompt=patient_system, temperature=temperature
        )
        patient_history.append({"role": "assistant", "content": patient_utterance})
        dialog_history.append({"turn": turn, "role": "Patient", "content": patient_utterance})

        if "BREAK" in patient_utterance:
            dialogue_ended = True
            break

    return {"dialog_history": dialog_history, "dialogue_ended": dialogue_ended}


def extract_diagnosis(extractor_client: ModelClient, dialog_history: List[Dict[str, Any]]) -> str:
    doctor_utterances = [t["content"] for t in dialog_history if t["role"] == "Doctor"]
    prompt = f"Doctor's replics: {' '.join(doctor_utterances)}"
    response = extractor_client.chat(
        [{"role": "user", "content": prompt}], system_prompt=EXTRACTION_SYSTEM_PROMPT, temperature=0.0
    )
    return response.strip()
