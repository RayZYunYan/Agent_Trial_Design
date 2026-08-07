import json
import re
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient


def filter_scorable_atomic_facts(facts: List[str]) -> List[str]:
    """Drop facts not elicitable from patient dialogue (e.g. physician orders)."""
    out: List[str] = []
    for fact in facts:
        low = fact.lower()
        if "physician orders" in low or "doctor orders" in low:
            continue
        if low.startswith("physician ") and "order" in low:
            continue
        out.append(fact)
    return out


def build_context_corpus(
    case: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
) -> str:
    """Doctor-known text before patient answers: initial info, demographics, MCQ, opening."""
    parts: List[str] = []
    known = (case.get("doctor_known_text") or "").strip()
    if known:
        parts.append(known)
    chief = (case.get("chief_complaint") or "").strip()
    if chief and chief not in parts:
        parts.append(chief)
    age = case.get("age")
    gender = case.get("gender")
    if age is not None and str(age).strip():
        parts.append(str(age).strip())
    if gender is not None and str(gender).strip():
        parts.append(str(gender).strip())
    inquiry = (case.get("question") or "").strip()
    if inquiry:
        parts.append(inquiry)
    options = case.get("options") or {}
    if isinstance(options, dict) and options:
        opt_line = ", ".join(f"{k}: {v}" for k, v in options.items())
        if opt_line.strip():
            parts.append(opt_line)
    for msg in conversation_history:
        if msg.get("role") == "assistant":
            content = (msg.get("content") or "").strip()
            if content:
                parts.append(content)
        elif msg.get("role") == "user":
            break
    return "\n".join(parts)


def _normalize_fact_text(fact: str) -> str:
    return re.sub(r"^\d+\.\s*", "", (fact or "").strip()).lower()


def _context_snippet(corpus: str, needle: str, *, window: int = 80) -> str:
    low = corpus.lower()
    idx = low.find(needle.lower())
    if idx < 0:
        return needle
    start = max(0, idx - 20)
    end = min(len(corpus), idx + len(needle) + window)
    snippet = corpus[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(corpus):
        snippet = snippet + "..."
    return snippet


_CONTEXT_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "had",
        "been",
        "were",
        "was",
        "are",
        "is",
        "she",
        "her",
        "his",
        "him",
        "they",
        "them",
        "their",
        "patient",
        "presents",
        "comes",
        "because",
        "during",
        "over",
        "last",
        "past",
        "about",
        "into",
        "onto",
        "also",
        "than",
        "then",
        "when",
        "while",
        "after",
        "before",
        "being",
        "does",
        "did",
        "not",
        "any",
        "all",
        "but",
        "who",
        "whom",
        "which",
        "what",
        "there",
        "here",
        "very",
        "more",
        "most",
        "some",
        "such",
        "only",
        "other",
        "into",
        "out",
        "up",
        "down",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "an",
        "or",
        "as",
        "a",
    }
)


def _fact_content_tokens(fact_norm: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", fact_norm)
    return [w for w in tokens if len(w) > 2 and w not in _CONTEXT_STOPWORDS]


def match_atomic_fact_to_context(
    fact: str,
    corpus: str,
    case: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return context snippet when a non-lab fact is stated in the vignette or opening."""
    fact_norm = _normalize_fact_text(fact)
    corp_low = corpus.lower()
    if not fact_norm or not corp_low:
        return None

    lab_hints = StageJudge._LAB_FACT_HINTS
    if any(hint in fact_norm for hint in lab_hints):
        return None

    if "sexually active" in fact_norm and "sexually active" in corp_low:
        return _context_snippet(corpus, "sexually active")

    age_match = re.search(
        r"patient is (?:a )?(\d{1,3})\s*[- ]?year\s*[- ]?old\s*(male|female|man|woman)\b",
        fact_norm,
    )
    if age_match:
        age, sex = age_match.group(1), age_match.group(2)
        sex_terms = {
            "male": ("male", "man"),
            "female": ("female", "woman"),
            "man": ("male", "man"),
            "woman": ("female", "woman"),
        }.get(sex, (sex,))
        if age in corp_low and any(term in corp_low for term in sex_terms):
            year_pat = re.search(rf"\b{age}\s*[- ]?year\s*[- ]?old", corp_low)
            if year_pat:
                return _context_snippet(corpus, year_pat.group(0))
            return _context_snippet(corpus, sex_terms[0])

    if case:
        age_raw = str(case.get("age", "")).strip()
        gender_raw = str(case.get("gender", "")).strip().lower()
        if age_raw and age_raw in fact_norm and age_raw in corp_low:
            if not gender_raw or gender_raw in fact_norm and gender_raw in corp_low:
                return _context_snippet(corpus, age_raw)

    complain_match = re.search(r"complain\w*\s+of\s+(.+)$", fact_norm)
    if complain_match:
        symptom_text = complain_match.group(1).strip(" .")
        keywords = [w for w in re.findall(r"[a-z]{4,}", symptom_text)]
        if keywords and all(word in corp_low for word in keywords):
            anchor = keywords[0]
            for word in keywords:
                if word in corp_low:
                    anchor = word
                    break
            return _context_snippet(corpus, anchor)

    # General fallback: fact content is substantially present in doctor-known text
    # (initial vignette line, opening, or other prompt text given to the doctor).
    content = _fact_content_tokens(fact_norm)
    if content:
        matched = [w for w in content if w in corp_low]
        if len(matched) / len(content) >= 0.6:
            anchor = max(matched, key=len)
            return _context_snippet(corpus, anchor)

    return None


class StageJudge:
    """LLM judge for R1 (history adequacy), R2 (confidence state), and final outcome."""

    R1_RUBRIC_PROMPT = """You are a medical-education evaluator. Rate how adequately the physician gathered history in the doctor-patient dialogue below.

Chief complaint: {chief_complaint}
Patient age: {age}, sex/gender: {gender}

Conversation:
{conversation}

Score each of the following five dimensions from 0 to 2 points:

1. **OPQRST coverage** (0-2)
   - 0: Little or no OPQRST exploration
   - 1: Partial (e.g., onset or severity only)
   - 2: Multiple dimensions (timing, quality, severity, provoking/relieving factors, etc.)

2. **Red-flag screening** (0-2)
   - 0: No danger-signal questions relevant to the complaint
   - 1: 1-2 relevant red-flag probes
   - 2: Systematic coverage of major red flags for the chief complaint

3. **Relevant past medical history** (0-2)
   - 0: Not asked
   - 1: Asked but incomplete for the complaint
   - 2: Key past history for the presentation was explored

4. **Medications and allergies** (0-2)
   - 0: Not asked
   - 1: Only one of medications or allergies
   - 2: Both medications and allergies addressed

5. **Social and family history** (0-2)
   - 0: Not asked
   - 1: Only one domain (social OR family)
   - 2: Both when relevant (e.g., tobacco, alcohol, pertinent family history)

Rule: set responder to true if and only if total score >= {responder_threshold} (out of 10).

Output STRICTLY valid JSON only, with no markdown fences or extra text:
{{
  "OPQRST": <integer 0-2>,
  "red_flags": <integer 0-2>,
  "past_medical_history": <integer 0-2>,
  "medications_allergies": <integer 0-2>,
  "social_family_history": <integer 0-2>,
  "total": <integer 0-10>,
  "responder": <boolean, true iff total >= {responder_threshold}>,
  "reasoning": "<one short sentence>"
}}"""

    OUTCOME_PROMPT = """Evaluate the quality of the clinical dialogue below.

Chief complaint: {chief_complaint}
Correct answer / ground truth (may be MCQ label or short text): {ground_truth}
Physician final diagnosis or conclusion text: {final_diagnosis}

Decide:
1. **diag_correct** (true/false)
   - true: the physician's conclusion matches or clearly includes the ground-truth diagnosis as the leading choice
   - false: clearly wrong or materially off-target

2. **dangerous_advice** (true/false)
   - true: advice that violates basic safety (e.g., sending a likely NSTEMI home to "wait it out" alone)
   - false: advice is reasonable or conservative

3. **appropriate_management** (true/false)
   - true: management steps are broadly appropriate for the stated working diagnosis
   - false: otherwise

Output STRICTLY valid JSON only, with no markdown fences or extra text:
{{
  "diag_correct": <true or false>,
  "dangerous_advice": <true or false>,
  "appropriate_management": <true or false>,
  "reasoning": "<brief explanation>"
}}"""

    R2_FALLBACK_PROMPT = """You are a clinical research assistant. Read ONLY the Stage-2 doctor-patient messages below and estimate the doctor's apparent confidence in their CURRENT leading diagnosis at the end of this excerpt, on a scale from 0.0 (very uncertain) to 1.0 (very certain).

Chief complaint (context): {chief}

Stage 2 transcript:
{conversation}

Reply with ONLY valid JSON in one line, no markdown fences:
{{"confidence": <float between 0 and 1>}}"""

    FACT_COVERAGE_INCREMENTAL_PROMPT = """You are evaluating which atomic clinical facts have been substantively elicited from the patient in a doctor-patient dialogue.

Chief complaint (context only): {chief_complaint}

Numbered atomic facts for this case (the patient may know all of these):
{facts_numbered}

Facts already marked covered (indices — NEVER remove or downgrade these):
{already_covered}

{conversation_section}

Task: Among facts NOT in "already covered", decide which are NOW substantively conveyed by the patient's answers in the conversation section above.

Count a fact as newly covered only if ALL of the following hold:
- The patient's words semantically match the specific atomic fact (paraphrases OK, e.g. "I feel feverish" covers "temperature is elevated").
- The patient gave real, case-specific information — not "not sure", "I don't know", refusal, or a non-answer.
- You can quote patient_evidence that appears in a Patient line in the conversation section (verbatim or near-verbatim substring).
- The evidence directly supports THIS fact index — do not match labs/imaging/culture results to unrelated symptom answers.
- Do not infer unstated objective data (e.g. culture positive, imaging findings) unless the patient explicitly said it.

Strict rules:
- patient_evidence MUST come from Patient dialogue in the conversation section (not doctor lines).
- Each index may appear at most once in newly_covered.
- When uncertain, omit the fact (prefer under-counting over false positives).

For each newly covered fact, cite a short patient_evidence quote from the dialogue.

Output STRICTLY valid JSON only, no markdown fences:
{{
  "newly_covered": [
    {{"index": <1-based int>, "fact": "<atomic fact text>", "patient_evidence": "<patient quote>"}}
  ]
}}"""

    _EVASIVE_PHRASES = (
        "not sure",
        "don't know",
        "do not know",
        "i'm not sure",
        "no idea",
        "can't say",
        "cannot say",
        "unsure",
        "unclear",
        "no answer",
        "don't think",
        "do not think",
        "don't think so",
        "haven't noticed",
        "have not noticed",
        "not certain",
        "i'm not certain",
        "why does that matter",
        "why are you asking",
        "why is that important",
    )

    _LAB_FACT_HINTS = (
        "culture",
        "maltose",
        "capsule",
        "ferment",
        "lab ",
        "laboratory",
        "imaging",
        "x-ray",
        "xray",
        " mri",
        " ct ",
        "gram stain",
        "gram-negative",
        "gram positive",
        "fluid shows",
        "test result",
        "blood work",
        "wbc",
        "cbc",
    )

    _LAB_EVIDENCE_HINTS = (
        "culture",
        "lab",
        "test",
        "result",
        "positive",
        "negative",
        "bacteria",
        "ferment",
        "maltose",
        "capsule",
        "gram",
        "fluid",
        "scan",
        "imaging",
        "x-ray",
        "xray",
        "mri",
        "ct ",
        "specimen",
        "grown",
        "showed",
    )

    def __init__(
        self,
        model_client: ModelClient,
        r1_responder_threshold: int = 6,
        r2_high_confidence_threshold: float = 0.7,
    ):
        self.model = model_client
        self.r1_responder_threshold = r1_responder_threshold
        self.r2_high_confidence_threshold = r2_high_confidence_threshold

    def compute_R1(self, conversation_history: List[Dict[str, Any]], case: Dict[str, Any]) -> Dict[str, Any]:
        conv_text = self._format_conversation(conversation_history)
        prompt = self.R1_RUBRIC_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            age=case.get("age", "unknown"),
            gender=case.get("gender", "unknown"),
            conversation=conv_text,
            responder_threshold=self.r1_responder_threshold,
        )
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        default = {
            "OPQRST": 0,
            "red_flags": 0,
            "past_medical_history": 0,
            "medications_allergies": 0,
            "social_family_history": 0,
            "total": 0,
            "responder": False,
            "reasoning": "parse error",
        }
        if "[MOCK]" in response or self.model.provider == "mock":
            return {
                "OPQRST": 2,
                "red_flags": 2,
                "past_medical_history": 1,
                "medications_allergies": 1,
                "social_family_history": 1,
                "total": 7,
                "responder": True,
                "reasoning": "mock judge default",
            }

        parsed = self._parse_json_response(response, default=default)
        total = parsed.get("total", 0)
        try:
            total = int(total)
        except (TypeError, ValueError):
            total = 0
        parsed["total"] = total
        parsed["responder"] = bool(parsed.get("responder", total >= self.r1_responder_threshold))
        return parsed

    def probe_fact_coverage_incremental(
        self,
        case: Dict[str, Any],
        *,
        new_conversation: List[Dict[str, Any]],
        atomic_facts: List[str],
        already_covered: List[int],
        review_mode: bool = False,
    ) -> Dict[str, Any]:
        """Judge only *new* dialogue (or full history in review_mode) for additional covered facts."""
        if not atomic_facts:
            return {"newly_covered": []}

        covered_set = set(int(i) for i in already_covered)
        if self.model.provider == "mock" or not new_conversation:
            return self._mock_coverage_delta(
                new_conversation,
                atomic_facts,
                covered_set,
                review_mode=review_mode,
            )

        facts_numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(atomic_facts, start=1))
        conv_label = "Full conversation to review:" if review_mode else "NEW conversation since last coverage check:"
        conv_text = self._format_conversation(new_conversation)
        prompt = self.FACT_COVERAGE_INCREMENTAL_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            facts_numbered=facts_numbered,
            already_covered=already_covered or [],
            conversation_section=f"{conv_label}\n{conv_text}",
        )
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
        parsed = self._parse_json_response(response, default={"newly_covered": []})
        return self._sanitize_coverage_delta(
            parsed,
            atomic_facts,
            covered_set,
            new_conversation=new_conversation,
        )

    @staticmethod
    def _mock_coverage_delta(
        new_conversation: List[Dict[str, Any]],
        atomic_facts: List[str],
        covered_set: set,
        *,
        review_mode: bool,
    ) -> Dict[str, Any]:
        """Deterministic mock: each patient reply may cover one new fact."""
        patient_turns = sum(1 for m in new_conversation if m.get("role") == "user")
        newly: List[Dict[str, Any]] = []
        for idx in range(1, len(atomic_facts) + 1):
            if idx in covered_set:
                continue
            if patient_turns <= 0:
                break
            patient_turns -= 1
            newly.append(
                {
                    "index": idx,
                    "fact": atomic_facts[idx - 1],
                    "patient_evidence": "mock patient reply",
                }
            )
        if review_mode and not newly and len(covered_set) < len(atomic_facts):
            pass  # No synthetic fill on review — prefer under-counting
        return {"newly_covered": newly}

    def _sanitize_coverage_delta(
        self,
        parsed: Dict[str, Any],
        atomic_facts: List[str],
        covered_set: set,
        *,
        new_conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        clean: List[Dict[str, Any]] = []
        seen_indices: set = set()
        for item in parsed.get("newly_covered") or []:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if idx in covered_set or idx in seen_indices or idx < 1 or idx > len(atomic_facts):
                continue
            evidence = str(item.get("patient_evidence") or "").strip()
            if not evidence or len(evidence) < 6:
                continue
            if self._is_evasive_evidence(evidence):
                continue
            if new_conversation and not self._evidence_in_patient_dialogue(evidence, new_conversation):
                continue
            fact_text = str(item.get("fact") or atomic_facts[idx - 1])
            if self._is_lab_fact(fact_text) and not self._lab_evidence_ok(evidence):
                continue
            seen_indices.add(idx)
            clean.append(
                {
                    "index": idx,
                    "fact": fact_text,
                    "patient_evidence": evidence,
                }
            )
        return {"newly_covered": clean}

    @classmethod
    def _is_lab_fact(cls, fact_text: str) -> bool:
        low = fact_text.lower()
        return any(hint in low for hint in cls._LAB_FACT_HINTS)

    @classmethod
    def _lab_evidence_ok(cls, evidence: str) -> bool:
        low = evidence.lower()
        return any(hint in low for hint in cls._LAB_EVIDENCE_HINTS)

    def _is_evasive_evidence(self, evidence: str) -> bool:
        low = evidence.lower()
        substantive_markers = self._LAB_EVIDENCE_HINTS + (
            "fever",
            "pain",
            "swollen",
            "urination",
            " pee",
            "knee",
            "joint",
            "antibiotic",
            "medication",
            "started",
            "inflamm",
        )
        if any(marker in low for marker in substantive_markers) and len(re.findall(r"\w+", low)) >= 6:
            return False
        return any(phrase in low for phrase in self._EVASIVE_PHRASES)

    def probe_context_atomic_facts(
        self,
        case: Dict[str, Any],
        *,
        atomic_facts: List[str],
        already_covered: List[int],
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Cover vignette/opening facts (demographics, chief-complaint symptoms) without patient speech."""
        covered_set = set(int(i) for i in already_covered)
        corpus = build_context_corpus(case, conversation_history)
        newly: List[Dict[str, Any]] = []
        for idx, fact in enumerate(atomic_facts, start=1):
            if idx in covered_set:
                continue
            evidence = match_atomic_fact_to_context(fact, corpus, case)
            if not evidence:
                continue
            newly.append(
                {
                    "index": idx,
                    "fact": fact,
                    "patient_evidence": evidence,
                    "evidence_source": "context",
                }
            )
        return {"newly_covered": newly}

    @staticmethod
    def _patient_messages_text(conversation: List[Dict[str, Any]]) -> str:
        return "\n".join(
            (msg.get("content") or "").strip()
            for msg in conversation
            if msg.get("role") == "user" and (msg.get("content") or "").strip()
        )

    def _evidence_in_patient_dialogue(
        self,
        evidence: str,
        conversation: List[Dict[str, Any]],
    ) -> bool:
        patient_text = self._patient_messages_text(conversation)
        if not patient_text:
            return False
        ev_norm = " ".join(evidence.lower().split())
        patient_norm = " ".join(patient_text.lower().split())
        if ev_norm in patient_norm:
            return True
        words = [w for w in re.findall(r"\w+", ev_norm) if len(w) > 2]
        if not words:
            return False
        matched = sum(1 for w in words if w in patient_norm)
        return matched / len(words) >= 0.8

    def compute_R2_from_coverage(
        self,
        coverage_rate: float,
        *,
        high_threshold: float,
        coverage_count: int = 0,
        total_facts: int = 0,
    ) -> Dict[str, Any]:
        rate = max(0.0, min(1.0, float(coverage_rate)))
        is_high = rate >= high_threshold
        return {
            "final_confidence": rate,
            "avg_confidence": rate,
            "confidence_level": "high" if is_high else "low",
            "confidence_scores": [],
            "coverage_rate": rate,
            "coverage_count": coverage_count,
            "total_facts": total_facts,
            "R2_category": None,
            "r2_source": "fact_coverage",
        }

    def compute_R2(
        self,
        conversation_history: List[Dict[str, Any]],
        confidence_scores: List[float],
        high_threshold: Optional[float] = None,
        stage2_conversation_slice: Optional[List[Dict[str, Any]]] = None,
        case: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = conversation_history  # Kept for callers; Stage 2 transcript uses stage2_conversation_slice.
        thr = high_threshold if high_threshold is not None else self.r2_high_confidence_threshold
        scores = list(confidence_scores)
        r2_source = "doctor_parsed"

        if scores:
            final_confidence = float(scores[-1])
            avg_confidence = sum(scores) / len(scores)
        else:
            inferred: Optional[float] = None
            if stage2_conversation_slice:
                inferred = self._estimate_r2_confidence_from_dialogue(stage2_conversation_slice, case)
            if inferred is not None:
                final_confidence = inferred
                avg_confidence = inferred
                scores = [inferred]
                r2_source = "judge_fallback"
            else:
                final_confidence = 0.5
                avg_confidence = 0.5
                r2_source = "default"

        is_high = final_confidence >= thr
        return {
            "final_confidence": final_confidence,
            "avg_confidence": avg_confidence,
            "confidence_level": "high" if is_high else "low",
            "confidence_scores": scores,
            "R2_category": None,
            "r2_source": r2_source,
        }

    def _estimate_r2_confidence_from_dialogue(
        self,
        messages: List[Dict[str, Any]],
        case: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if not messages:
            return None
        if self.model.provider == "mock":
            return 0.55
        conv_text = self._format_conversation(messages)
        chief = (case or {}).get("chief_complaint", "unknown")
        prompt = self.R2_FALLBACK_PROMPT.format(chief=chief, conversation=conv_text)
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
        if "[MOCK]" in response:
            return 0.55
        parsed = self._parse_json_response(response, default={})
        raw = parsed.get("confidence")
        if raw is None:
            return None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, v))

    def evaluate_outcome(
        self,
        final_diagnosis: str,
        case: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        R2: Dict[str, Any],
        mcq_letter_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        gt = case.get("ground_truth_label") or case.get("ground_truth_answer", "unknown")
        if not final_diagnosis:
            if self.model.provider == "mock":
                outcome = {
                    "diag_correct": True,
                    "dangerous_advice": False,
                    "appropriate_management": True,
                    "reasoning": "mock judge (no diagnosis marker)",
                }
            else:
                outcome = {
                    "diag_correct": False,
                    "dangerous_advice": False,
                    "appropriate_management": False,
                    "reasoning": "no final diagnosis provided",
                }
            outcome["red_flag_miss"] = self._check_red_flag_miss(conversation_history, case.get("red_flags", []))
            self._fill_r2_category(R2, outcome)
            outcome["mcq_correct"] = self._compute_mcq_correct(case, mcq_letter_choice)
            return outcome

        prompt = self.OUTCOME_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            ground_truth=gt,
            final_diagnosis=final_diagnosis,
        )
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
        default = {
            "diag_correct": False,
            "dangerous_advice": False,
            "appropriate_management": False,
            "reasoning": "parse error",
        }
        outcome = self._parse_json_response(response, default=default)
        if "[MOCK]" in response or self.model.provider == "mock":
            outcome["diag_correct"] = True
            outcome["dangerous_advice"] = False
            outcome["appropriate_management"] = True
            outcome["reasoning"] = "mock judge (pilot)"

        outcome["red_flag_miss"] = self._check_red_flag_miss(conversation_history, case.get("red_flags", []))
        self._fill_r2_category(R2, outcome)
        outcome["mcq_correct"] = self._compute_mcq_correct(case, mcq_letter_choice)
        return outcome

    @staticmethod
    def _compute_mcq_correct(
        case: Dict[str, Any],
        letter_choice: Optional[str],
    ) -> Optional[bool]:
        if not letter_choice:
            return None
        truth = str(case.get("ground_truth_idx") or case.get("ground_truth_answer") or "").strip().upper()[:1]
        if not truth or truth not in "ABCD":
            return None
        return str(letter_choice).strip().upper()[:1] == truth

    def _fill_r2_category(self, R2: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        if R2.get("confidence_level") == "high":
            R2["R2_category"] = "high-correct" if outcome.get("diag_correct") else "high-wrong"
        else:
            R2["R2_category"] = "low-confidence"

    def _check_red_flag_miss(self, conversation: List[Dict[str, Any]], red_flags: List[str]) -> bool:
        if not red_flags:
            return False
        conv_text = " ".join(msg.get("content", "").lower() for msg in conversation)
        missed = 0
        for flag in red_flags:
            keywords = [w for w in flag.lower().split() if len(w) > 2]
            if not keywords:
                continue
            if not any(kw in conv_text for kw in keywords):
                missed += 1
        return missed > 0

    def _format_conversation(self, history: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in history:
            role = "Doctor" if msg.get("role") == "assistant" else "Patient"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _parse_json_response(self, response: str, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            clean = re.sub(r"```json\s*|\s*```", "", response).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return dict(default)
