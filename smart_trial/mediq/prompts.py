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
    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
OPTIONS: {}
YOUR TASK: {}""",
}
