"""MediQ expert prompts (self-contained in smart_trial; not imported from src/)."""

EXPERT_SYSTEM = {
    "meditron_system_msg": (
        "You are a medical doctor trying to reason through a real-life clinical case. "
        "Based on your understanding of basic and clinical science, medical knowledge, "
        "and mechanisms underlying health, disease, patient care, and modes of therapy, "
        "respond according to the task specified by the user. Base your response on the "
        "current and standard practices referenced in medical guidelines."
    ),
    "question_word": "Doctor Question",
    "answer_word": "Patient Response",
    "implicit": (
        "Given the information so far, if you are confident to pick an option correctly and "
        "factually, respond with the letter choice and NOTHING ELSE. Otherwise, if you are "
        "not confident to pick an option and need more information, ask ONE SPECIFIC ATOMIC "
        "QUESTION to the patient. The question should be bite-sized, NOT ask for too much at "
        "once, and NOT repeat what has already been asked. In this case, respond with the "
        "atomic question and NOTHING ELSE."
    ),
    "implicit_RG": (
        "Given the information so far, if you are confident to pick an option correctly and "
        "factually, respond in the format:\n"
        "REASON: a one-sentence explanation of why you are choosing a particular option.\n"
        "ANSWER: the letter choice and NOTHING ELSE. Otherwise, if you are not confident to "
        "pick an option and need more information, ask ONE SPECIFIC ATOMIC QUESTION to the "
        "patient. The question should be bite-sized, NOT ask for too much at once, and NOT "
        "repeat what has already been asked. In this case, respond in the format:\n"
        "REASON: a one-sentence explanation of why you should ask the particular question.\n"
        "QUESTION: the atomic question and NOTHING ELSE."
    ),
    "atomic_question_improved": (
        "If there are missing features that prevent you from picking a confident and factual "
        "answer to the inquiry, consider which features are not yet asked about in the "
        "conversation log; then, consider which missing feature is the most important to ask "
        "the patient in order to provide the most helpful information toward a correct medical "
        "decision. You can ask about any relevant information about the patient's case, such "
        "as family history, tests and exams results, treatments already done, etc. Consider "
        "what are the common questions asked in the specific subject relating to the patient's "
        "known symptoms, and what the best and most intuitive doctor would ask. Ask ONE SPECIFIC "
        "ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT "
        "ask for too much at once. Make sure to NOT repeat any questions from the above "
        "conversation log. Answer in the following format:\n"
        "ATOMIC QUESTION: the atomic question and NOTHING ELSE.\n"
        "ATOMIC QUESTION: "
    ),
    "answer": (
        "Assume that you already have enough information from the above question-answer pairs "
        "to answer the patient inquiry, use the above information to produce a factual "
        "conclusion. Respond with the correct letter choice (A, B, C, or D) and NOTHING "
        "ELSE.\nLETTER CHOICE: "
    ),
    "grounded_answer": (
        "Using ONLY information explicitly stated in the conversation log above, pick the "
        "best letter (A, B, C, or D) for the inquiry. If the conversation does not contain "
        "enough factual information to differentiate the options, respond with ABSTAIN and "
        "NOTHING ELSE.\nLETTER CHOICE: "
    ),
    "grounded_answer_RG": (
        "Using ONLY information explicitly stated in the conversation log above, pick the "
        "best letter (A, B, C, or D) for the inquiry. You MUST cite patient dialogue.\n"
        "If the conversation does not contain enough factual information, respond:\n"
        "REASON: <one sentence>\nABSTAIN\n"
        "Otherwise respond:\n"
        "REASON: <one sentence citing specific patient facts from the log>\n"
        "ANSWER: <single letter A, B, C, or D and NOTHING ELSE>"
    ),
    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
OPTIONS: {}
YOUR TASK: {}""",
}

STRATEGY_PRECEDENCE = (
    "The global clinical task rules below define required information types and MCQ context. "
    "Your stage strategy defines emphasis and ordering only; it does not exempt you from "
    "asking about labs, prior workup, medications, or other missing features needed to "
    "differentiate the options."
)


def _format_mcq_options(case: dict) -> str:
    opts = case.get("options") or {}
    if not isinstance(opts, dict):
        return "A: (unknown) B: (unknown) C: (unknown) D: (unknown)"
    parts = []
    for letter in "ABCD":
        parts.append(f"{letter}: {opts.get(letter, opts.get(letter.lower(), ''))}")
    return ", ".join(parts)


def build_safe_patient_initial_info(case: dict) -> str:
    """Initial visit context only — no vignette labs, imaging, or atomic facts."""
    age = case.get("age", "unknown")
    gender = case.get("gender", "unknown")
    chief = (case.get("chief_complaint") or "unknown presentation").strip()
    return f"Age: {age}; Gender: {gender}; Chief complaint: {chief}"


def build_global_clinical_task_block(case: dict) -> str:
    """SMART global layer borrowed from MediQ implicit / atomic_question_improved."""
    inquiry = (case.get("question") or "Which option best fits this clinical scenario?").strip()
    options_text = _format_mcq_options(case)
    patient_info = build_safe_patient_initial_info(case)
    implicit = EXPERT_SYSTEM["implicit"]
    atomic_hint = EXPERT_SYSTEM["atomic_question_improved"]
    return f"""## Global Clinical Task (MCQ — MediQ-aligned)

You are working toward answering a multiple-choice clinical question. You do NOT yet have
full chart data; gather information through the dialogue.

PATIENT INFORMATION (initial visit only):
{patient_info}

INQUIRY (do not assume unstated exam or lab results are already known):
{inquiry}

OPTIONS:
{options_text}

Information-seeking rules (from MediQ):
{implicit}

When you need more information, prefer questions that address missing features important
for differentiating the options. {atomic_hint.split('Answer in the following format')[0].strip()}

{STRATEGY_PRECEDENCE}

Rules:
- Ask exactly ONE atomic question per turn unless concluding with [DIAGNOSIS].
- Do not repeat questions already in the conversation log.
- You may ask about labs, imaging, prior tests, treatments, family history, and exam findings
  the patient may know from prior care — even if unusual in real life, this patient may have that information.
- Do not reveal the option letters or your leading diagnosis while still questioning."""
