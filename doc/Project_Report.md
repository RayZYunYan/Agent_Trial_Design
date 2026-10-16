# SMART Design for Medical LLM Agents

**Status.** Stopped after the last cluster run. The results were not strong enough to continue, and Prof. Liu ended the project. This note is the record of what was built, what was measured, and where the code and logs live. The work was done at the USC Viterbi School of Engineering.

**People.** Advisor: Liu, Ruishan. Guidance and study design: Li, Dongze. Project members: Zhang, Rui (Ray); Xia, Zihan (Zoe). Research assistant: Li, Jiawei.

**Repository.** [github.com/RayZYunYan/Agent_Trial_Design](https://github.com/RayZYunYan/Agent_Trial_Design), forked in spirit from [stellalisy/mediQ](https://github.com/stellalisy/mediQ). The original MediQ benchmark remains under `src/`. Copyright, the research-use disclaimer, and third-party license terms are in `LICENSE` at the repository root.

**How to read this note.** Sections 1–4 are the question and the system. Sections 5–6 are the SMART / Q-learning line. Sections 7–9 are the later multi-agent experiments and the reason the project stopped. Section 10 is work that was written down as a plan and was not run as a finished experiment. Section 11 is a short side exploration. Section 12 is for someone who wants to continue from this repository.

**Which numbers to trust.** The lab notebook is `doc/Agent Trial SMART Design.pdf`. Inside that notebook, tables are ordered newest first. When two tables report the same comparison, use the one higher in the notebook. Completed run logs from the last experiments are on the advisor’s CARC cluster (USC Discovery). They were not copied back into git. A partial local 3MDBench snapshot that used to sit at `clinic_experiment/outputs/mdbench/X0/` was removed: agent E stopped at 58 cases, and its accuracies did not match the notebook.

---

## Suggested GitHub description

Adaptive clinical-dialogue experiments on top of MediQ. The repository contains a two-stage SMART simulator (re-randomization, retrieval, trajectory logs, offline Q-learning / A-learning), a later multi-doctor fact and cross-diagnosis study on MediQ, AI Hospital, and 3MDBench, and the lab notebook. The project stopped after the last CARC run: no single questioning policy beat giving the model the record directly, and interviewer–diagnostician compatibility did not yield a result strong enough to continue. The default branch `cross-diagnosis` contains the latest study and the earlier SMART code. The work was done at the USC Viterbi School of Engineering. This is a research record, not a clinical tool. Start with `doc/Project_Report.md` and `LICENSE`.

---

## 1. Question we started with

The original question was whether a medical dialogue agent should be evaluated as one block, or as a sequence of decisions. Existing benchmarks (MediQ, AgentClinic, and the single-randomization chatbot trials we read) assign one policy for the whole encounter and score the ending. A clinical conversation is sequential: how to take the history, whether to look anything up, and how to close. The statistical tool for that structure is a Sequential Multiple Assignment Randomized Trial (SMART; Murphy 2005) and the dynamic treatment regime (DTR) estimated from it by Q-learning and A-learning.

The concrete plan, written in April–May 2026, was:

- One text encounter. The doctor sees age, sex, and the chief complaint. The patient simulator sees the full case.
- Three fixed decision points. Stage 1, turns 1–4, history-taking. Stage 2, turns 5–10, retrieval. Stage 3, turns 11–20, how to conclude.
- Three arms at each stage. History: comprehensive review of systems (A1a), focused chief complaint (A1b), red-flag first (A1c). Retrieval: always retrieve (A2a), retrieve only when confidence is low (A2b), parametric knowledge only (A2c). Conclusion: one diagnosis plus a plan (A3a), a differential with shared decision-making (A3b), or escalate (A3c).
- After each stage, an intermediate score re-draws the next arm from a restricted pool. Stage 1 responders (information score R1 ≥ 6) may receive any Stage 2 arm. Non-responders may not receive “parametric only.” High versus low confidence restricts the Stage 3 pool.
- Primary endpoint: diagnostic accuracy against the case’s ground truth. Secondary endpoints that were specified and mostly not analyzed: missed red flags, dangerous advice, turn count.

The design spec, the 12-day build plan, and the Q-learning analysis plan are in the PDF.

Patient simulation followed MediQ’s Fact-Select patient: the model answers from atomic facts in the case and does not volunteer the rest. Cases came from MediQ’s iMEDQA release (`stellalisy/mediQ`, CC-BY 4.0). iMEDQA has no red-flag field, so red flags were added by us, first as a category list and later as a cached per-case file.

---

## 2. What changed

Three design changes matter more than the rest.

**Stage 3 was removed.** The Stage 3 score mixed the doctor’s confidence with whether the diagnosis was already correct. That second part is only known after the encounter, so the Stage 3 draw was not a proper response-adaptive randomization. From early June 2026 the trial is two-stage. The final diagnosis is given inside Stage 2. R2 (confidence at the end of Stage 2) is logged and is not used to assign an arm. The current tree has Stage 1 and Stage 2 arm files only; the Stage 3 YAML files from the May plan are gone.

**The estimand moved.** The two-stage SMART did run, and Q-learning did return a regime (red-flag-first, then always retrieve, for patients who did not clear the Stage 1 bar). Accuracy stayed far below the same items when the model was handed the full record. The cases span many specialties, and the sample was too small for a specialty-specific regime. The September synthesis in the notebook states that conclusion directly: a general optimal questioning policy was not identified.

**The last experiments ask a different question.** Once we re-ran plain MediQ and then the same doctors on AI Hospital and 3MDBench, the stable pattern was not “which script is best.” It was that the model that asked the questions and the model that read the transcript are not interchangeable. Self-play (same model asks and answers) and cross-play (one model’s transcript, another model’s diagnosis) differ, and the difference is not explained by “the better interviewer plus the better diagnostician.” That is the result the last CARC run was meant to pressure-test. It did not come out strongly enough to keep going.

---

## 3. Repository map

Use `cross-diagnosis`. It is the GitHub default branch. It contains the latest multi-doctor study and the earlier SMART simulator, retrieval code, Q-learning analysis, and MediQ experiment code. `smart-trial` is an earlier snapshot of that same line, at the 10 August 2026 `atomic facts` commit. It is the old `main` tip under a new name. It is not a second copy of the results.

Older branch names are not on GitHub. Their commits are inside `cross-diagnosis`. The names below are a map of that history, so a result table can be tied to the study that produced it.

The local checkout of `cross-diagnosis` can be behind `origin/cross-diagnosis`. The remote tip includes a 3MDBench output directory that was removed from this working tree because it did not match the notebook. Read remote tips before assuming the working tree is the last commit.

| Name | Where it sits in the history | What it records |
|---|---|---|
| `cross-diagnosis` | Default branch. Latest tip. Formerly `new_dataset`. | Five-doctor MediQ rerun plus `clinic_experiment/` for AI Hospital and 3MDBench. Also contains `smart_trial/` and `mediq_experiment/`. |
| `smart-trial` | Earlier snapshot. Formerly `main`. | May simulator through the June data and eval merge, up to the August atomic-fact commit. |
| `output` | Commits inside `cross-diagnosis`. | One-pass data build; `data/all_mediq_craft_merged.jsonl` (2,685 MediQ + CRAFT-MD cases). |
| `q-learning-framework`, `2stage` | Commits inside `cross-diagnosis`. | Two-stage trial, BM25 retrieval, offline Q-learning / A-learning. `smart_trial/q_learning/RESULTS.md` is the 12 June 2026 analysis (n = 200). |
| `eval` | Commits inside `cross-diagnosis`. | Fixed 50-case cohort: baseline, 50×9 static grid, offline adaptive lookup, closed-loop adaptive. Claude smoke tests and CARC SLURM scripts. Section 5. |
| `detail` | Commits inside `cross-diagnosis`. | Stricter coverage, question de-duplication, early stop. |
| `MediQ` | Commits inside `cross-diagnosis`. | Five-doctor MediQ rerun and fact ablations (`mediq_experiment/`). Section 7. |

`src/` is the upstream MediQ benchmark (single interactive loop, multiple-choice judge). It was left in the tree on purpose. `smart_trial/` does not import it. Its license is separate from the original code in this repository. See `LICENSE`.

---

## 4. The simulator that was built

By 11 May 2026 a mock encounter ran end to end: opening, four Stage 1 turns, R1, six Stage 2 turns, R2, Stage 3 through a conclusion or the turn cap, outcome, JSONL. By 13 May the same path ran on Groq (`llama-3.1-8b-instant` for doctor, patient, and judge), with `[CONFIDENCE: 0.XX]` parsed from the doctor and a judge fallback if parsing failed. One checked case (`medqa_0000`, seed 42) took A1b → A2a → A3b with R2 = 0.60 from the doctor’s own confidence.

Later pieces, all under `smart_trial/`:

- **Arms.** `config/arms/stage1_A1a.yaml`, `A1b`, `A1c`, `stage2_A2a.yaml`, `A2b`, `A2c`. Each arm is a system-prompt injection plus a retrieval flag. Stage 1 is four turns. Stage 2 is up to six turns and ends on a `[DIAGNOSIS]` marker, or is forced to conclude on the last turn.
- **Randomization.** `core/randomizer.py`. Stage 1 is uniform over {A1a, A1b, A1c}, stratified by case category. If R1 ≥ 6, Stage 2 is uniform over {A2a, A2b, A2c}. Otherwise Stage 2 is uniform over {A2a, A2b}.
- **Judge.** `core/judge.py`. R1 is a 0–10 rubric (OPQRST, red flags, past history, medications and allergies, social and family history; responder at 6). Outcome fields include `diag_correct` and, in the logger, red-flag miss, dangerous advice, and appropriateness. From July, MediQ-style runs also score a multiple-choice letter against `ground_truth_idx`, which is deterministic. The free-text LLM judge remained the SMART outcome and is more subjective.
- **Patient personas.** `core/persona.py` and `config/personas/`. Zoe merged an earlier four-axis persona sketch with a health-literacy axis (Nutbeam: functional / interactive / critical vocabulary and jargon comprehension), plus personality, recall, and emotional state. Presets: `default`, `low_literacy`, `anxious`, `reserved`, `difficult`. A jargon short-circuit asks the patient to request a plain-language clarification instead of answering. In practice the doctor prompt already required plain language, so the short-circuit rarely fired during history-taking; jargon showed up mainly in the concluding turn, after which the encounter ended. Personas are recorded on the trajectory so a run can be replayed.
- **Retrieval.** Stage 2 arms A2a and A2b emit `[RETRIEVAL QUERY: ...]`. The orchestrator strips that marker before the patient sees the question, runs retrieval, and injects the top passages into the next doctor system prompt. A2c never retrieves. The first index was BM25 over the MedRAG textbooks corpus (9 clinical textbooks, about 98,799 passages, top 3). StatPearls was the original target and could not be redistributed. Zoe then added Contriever (`facebook/contriever`) and a hybrid ranker (reciprocal rank fusion of BM25 and Contriever, following the LaMP setup). Mode is `rag.retriever_mode` in `trial_config.yaml`: `bm25`, `contriever`, or `hybrid`. The index files are local caches and are not the scientific result.
- **MediQ doctor loop inside the SMART shell.** `smart_trial/mediq/`. When enabled, the doctor uses MediQ’s BasicExpert pattern: each turn either asks one question or commits to an A–D letter, with the SMART arm text added as a constraint. Stage 1 can suppress an early answer. This was the bridge used when we put MediQ’s interactive setup next to our arms. It is off unless `mediq.enabled` is set. Experts other than BasicExpert, MediQ’s own patient, vLLM log-probabilities, and the original `max_questions=30` loop were not ported.
- **Logs.** One JSON object per encounter. The May schema (arms, R1, R2, trajectory, outcome) is the Q-learning input. Later MediQ fields include letter choice and fact coverage. Entry point: `run_encounter.py`. Summary: `python -m smart_trial.scripts.summarize_encounters`.

A known data limitation, written down on 25 June: a large share of iMEDQA items turn on laboratory or study results. A patient who only answers history questions cannot surface those facts, so a SMART-style interview is capped on those items. MediQ’s own patient can fall back to a third-person disclosure. Our patient did not, and the doctor prompt was only later pushed to ask about labs.

---

## 5. SMART results

Two analyses share a date and must be kept apart. They use different samples and different estimators.

### 5.1 Fixed 50-case comparison (eval plan, notebook entry of 12 June 2026)

Fifty cases were fixed as the test cohort. The comparison was:

1. **Baseline.** No stages, no arm text, at most 11 turns. One encounter per case.
2. **50 × 9 grid.** Each case run under all nine (A1, A2) pairs. Average accuracy of each pair on the 50 cases.
3. **Offline adaptive policy.** Q-learning fit on that grid; for each case, look up the outcome of the path the fitted rule would have chosen. This does not re-run the simulator.
4. **Closed-loop adaptive.** The rule chooses the Stage 1 arm from the Stage 2 state and the encounter is actually re-run. Separate from the lookup number.

Average accuracy:

| Path | Accuracy |
|---|---|
| A1a → A2a | 0.46 |
| A1a → A2b | 0.28 |
| A1a → A2c | 0.52 |
| A1b → A2a | 0.44 |
| A1b → A2b | 0.34 |
| A1b → A2c | 0.32 |
| A1c → A2a | 0.44 |
| A1c → A2b | 0.30 |
| A1c → A2c | 0.46 |
| Baseline (no stage, ≤ 11 turns) | 0.26 |
| Q-learning adaptive, lookup on the grid, not a held-out test | 0.56 |
| Closed-loop adaptive | 0.48 |
| MediQ full record, all information at once | 0.797 |

For scale, the notebook also records the published MediQ interactive numbers we were comparing against: initial (no interaction) 0.545, interactive basic 0.554, interactive best 0.661. Our full-record figure of 0.797 is higher than those published interactive numbers because it gives the model the record directly.

The best static pair on this grid is A1a → A2c at 0.52. The lookup adaptive number is 0.56. The closed loop, which is the one that matches a real deployment, is 0.48. All of them sit well below 0.797.

### 5.2 Offline Q-learning and A-learning on 200 simulated encounters

`smart_trial/q_learning/RESULTS.md` analyzes 200 two-stage encounters logged on branch `2stage` (analysis date 12 June 2026). Reproduction is `python -m smart_trial.q_learning.run_main_analysis`. Source tables are under `smart_trial/q_learning/outputs/`.

This is a randomized SMART sample, not the 50 × 9 grid. Stage 1 is uniform on three arms. Stage 2 is uniform on the feasible pool (three arms if R1 ≥ 6, two arms otherwise). The outcome is correct diagnosis. Mean outcome in the sample is 0.260. Only 24 of 200 encounters (12%) were Stage 1 responders. The sparsest cell, A1c → A2c, has n = 1. The best observed cell is A1c → A2a (n = 32, mean outcome 0.406).

| Estimand | Estimate | 95% interval |
|---|---|---|
| Value of the estimated optimal regime, Q-learning, cross-fit | 0.525 | percentile bootstrap (0.423, 0.699); m-out-of-n (0.416, 0.697); CMS projection (0.253, 0.720) |
| Same value, A-learning plug-in | 0.441 | — |
| Best static pair (A1c → A2a) | 0.415 | — |
| Sample mean outcome | 0.260 | — |

The plug-in values sit above both the sample mean and the best observed cell. An oracle simulation with a known regime still showed about 0.2 upward bias at this sample size, because the estimator maximizes over extrapolated Q-values. The report in `RESULTS.md` says to treat the level of V̂ as optimistic. The more stable signals are the Q-versus-A agreement and the gap between the adaptive rule and the best static pair (0.110 on this analysis; an earlier three-stage analysis had a gap of about 0.016).

Adjusted Stage 1 values from Q-learning, after an optimal Stage 2 action: A1c 0.497, A1a 0.365, A1b 0.279. Stage 2 contrasts are fragile. A2c is feasible for only 24 subjects and was actually assigned to 13. Q-learning and A-learning do not rank A2c the same way.

The regime both methods support is for the 176 non-responders (88% of the sample): **A1c at Stage 1, then A2a at Stage 2.** That matches the best observed cell. Responder subgroups have 4–10 subjects. Their recommended switches (A2b or A2c) are directional only. Q-learning’s fitted values in those cells (some above 0.7) are not a result to quote; the A-learning values there sit around 0.36–0.44.

Rule agreement between Q-learning and A-learning on the same 200 encounters: Stage 1, 0.875; Stage 2, 0.935; both stages, 0.835. The earlier three-stage analysis agreed on both stages only about 0.39 of the time. Dropping Stage 3 made the two estimators agree. It did not make the regime recoverable.

Oracle check on synthetic two-stage data (8 replicates): at n = 300, the fraction of replicates that recovered the entire true regime was 0.188. The estimated value stayed above the achievable mean. `RESULTS.md` therefore calls the fitted rule provisional.

Secondary outcomes that were specified on 5 June (all relevant red flags asked, no guideline-violating advice, encounter of at most 10 turns) were not the endpoints of this analysis.

---

## 6. What that line established

The September synthesis, written before the project stopped, is the right summary of Sections 5.1 and 5.2:

- A two-stage randomization with a logged intermediate score is implementable on a MediQ-style simulator, and the offline estimators can be run.
- On the 50-case grid, arm choice moves accuracy (roughly 0.28 to 0.52) relative to an unstructured baseline (0.26).
- The adaptive rules land at 0.48 (closed loop) or 0.56 (in-sample lookup). The same items with the full record are at 0.797.
- On the 200-encounter SMART sample, both estimators point at A1c then A2a for non-responders, and they agree much more often than they did in three stages. The value estimates are optimistic, the oracle does not recover the true regime at n = 300, and responder arms are too sparse to interpret.
- Cases cover many specialties. Nothing in these samples supports a single questioning script, or a stable per-specialty script, that would close the gap to the full record.

That is why the work shifted to measuring what the dialogue actually contains, and who is reading it.

---

## 7. MediQ rerun and fact ablations

Code: `mediq_experiment/`. Cases 100–199 (100 cases). Doctors in the final five-way setup, as named in the notebook:

| Id | Model |
|---|---|
| A | gpt-5.4 |
| B | claude-sonnet-4-6 |
| C | gpt-5.6-luna |
| D | Qwen3.5-4B-OptiQ-4bit, local |
| E | Meta-Llama-3.1-8B-Instruct-4bit, local |

Patient and judge for this block were Claude Haiku 4.5. D and E were run on a Mac mini via MLX after a vLLM attempt; an earlier candidate for E was DeepSeek-R1-0528-Qwen3-8B, replaced by Llama once access was available. The checked-in `clinic_experiment/config.yaml` later points some ids at a relay and different model strings. Cite the notebook’s model list above for the tables in this section, not whatever the YAML says on a given checkout.

### 7.1 Two doctors, interactive (17 July 2026)

Agents A and B, end to end, no SMART stages. Fact coverage here is the fraction of atomic facts mentioned. On 7 August we found that this coverage score ignored facts already present in the question stem, so every coverage number in this subsection is preliminary.

| | Final accuracy | Mean fact coverage | The other doctor reading this transcript |
|---|---|---|---|
| A (gpt-5.4) | 0.79 | 0.074 | B reading A: 0.60 |
| B (claude-sonnet-4-6) | 0.67 | 0.055 | A reading B: 0.69 |

The dialogue × diagnosis matrix is the same four cells: A on A 0.79, B on A 0.60, A on B 0.69, B on B 0.67.

From those four cells the notebook computes: A’s dialogues average 0.695 and B’s average 0.680 (1.5 points). A’s diagnoses average 0.740 and B’s average 0.635 (10.5 points). Self-play averages 0.73 and cross-play averages 0.645 (8.5 points). The interaction contrast is 0.79 − 0.60 − 0.69 + 0.67 = 0.17. A’s advantage is mostly diagnosis, not interviewing, and the pairing term is large relative to the interviewing gap. Proposed mechanisms, not tested as a separate experiment: each model asks for cues it knows how to use; the transcript’s style fits its author; the two roles form an implicit protocol.

Interactive runs on these 100 cases also stopped very early. With the MediQ rule that a confidence of at least 4 may end the interview, gpt-5.4 and Claude Sonnet 4.6 typically stopped after two or three questions. Low fact coverage is partly that early stop, and partly the stem-scoring bug above.

### 7.2 Random facts, same 100 cases (23 July 2026, Zoe)

For each case, physician-order facts were removed. For k in {1, 2, 4, 8, all}, k facts were drawn at random, three repeats, and the same draw was given to A and B as a static prompt (facts + question + options, letter only). Both models’ accuracy rose with k. A led B at every k. The gap was about 0.3 points at k = 1 and 8.3 points at k = all, where A was 0.91 and B was 0.83. The interactive gap in Section 7.1 was 12 points (0.79 vs 0.67). Giving both models the same full fact list shrinks the gap and does not remove it, so part of A’s lead is reasoning on identical information, not only information gathering. The per-k table was written to `mediq_experiment/outputs/fact_coverage_experiment/` on the machine that ran it; do not reconstruct intermediate k values that are not in the notebook.

### 7.3 Five doctors on MediQ

The newer MediQ table adds C, D, and E. Rows are the information source. Columns are who diagnoses. Values are accuracy on the same 100 cases.

| Information | A | B | C | D | E |
|---|---|---|---|---|---|
| Base prompt only | 0.43 | 0.47 | 0.48 | 0.33 | 0.40 |
| 1 random fact | 0.63 | 0.51 | 0.63 | 0.42 | 0.49 |
| 2 most important, Claude-ranked | 0.89 | 0.87 | 0.93 | 0.64 | 0.59 |
| 2 most important, GPT-ranked | 0.84 | 0.82 | 0.92 | 0.63 | 0.55 |
| Dialogue A | 0.79 | 0.60 | 0.85 | 0.47 | 0.44 |
| Dialogue B | 0.69 | 0.67 | 0.76 | 0.42 | 0.44 |
| Dialogue C | 0.74 | 0.59 | 0.81 | 0.40 | 0.43 |
| Dialogue D | 0.75 | 0.58 | 0.75 | 0.49 | 0.35 |
| Dialogue E | 0.77 | 0.79 | 0.85 | 0.41 | 0.53 |
| All facts | 0.93 | 0.83 | 0.95 | 0.64 | 0.57 |

Reading this table, which supersedes Section 7.1 for A and B:

- Two curated facts already reach the high 0.8s or low 0.9s for A, B, and C. All facts add a smaller further gain (A 0.84–0.89 → 0.93; C 0.92–0.93 → 0.95). D and E stay near 0.55–0.64 even with all facts.
- A full dialogue is not uniformly better than two curated facts for the strong models. A reading its own dialogue scores 0.79, below A reading two Claude-ranked facts (0.89). C is the exception among the strong models: C reading A’s or E’s dialogue scores 0.85, still under C with all facts (0.95) and under C with two curated facts (0.92–0.93).
- Self-play is the diagonal of the dialogue block: A 0.79, B 0.67, C 0.81, D 0.49, E 0.53. Several off-diagonal cells are higher than the author’s own diagnosis (B reading E is 0.79; A reading E is 0.77; C reading E is 0.85). The “self-play wins” pattern from the 2 × 2 does not hold for every pair once five models are in the matrix. E’s transcripts are unusually useful to A, B, and C, while E’s own diagnoses stay weak.
- A later note gives mean fact coverage on these interactive runs as A 0.074, B 0.055, C 0.082, D 0.123, E 0.326, and observes that E’s higher coverage did not make E a better diagnostician, while B’s accuracy jumped when B read E. Treat those coverage figures as the pre-correction scorer (stem facts omitted).

Claude and GPT were each asked to rank atomic facts. On the overlap table at the top of the notebook (newest ranking comparison):

| Tier | Claude facts | Overlap with GPT | Repetition | Jaccard |
|---|---|---|---|---|
| Most important | 200 | 152 | 76.0% | 61.3% |
| Important | 253 | 162 | 64.0% | 47.1% |
| Medium | 293 | 175 | 59.7% | 42.6% |
| Low | 322 | 244 | 75.8% | 61.0% |

The two rankers agree most on the top tier and on the bottom tier, and least in the middle. In the MediQ accuracy table, Claude-ranked and GPT-ranked pairs are close for every doctor. The choice of ranker is not the main effect. The jump from one random fact to two top-ranked facts is.

---

## 8. AI Hospital and 3MDBench

Code: `clinic_experiment/`. The pipeline runs several doctors against one shared patient (and, for AI Hospital, one shared reporter), then builds a matrix: base vignette only, one random fact, two top-ranked facts, each doctor’s transcript, and all facts, each re-diagnosed by every doctor. AI Hospital was sampled from 506 cases; 3MDBench from 2,996 text cases, images dropped. The working sample in the notes is 100 cases per dataset.

Scoring differs by dataset. AI Hospital follows that benchmark’s five-axis rubric, binarized so grades A or B count as correct. 3MDBench maps the free-text diagnosis onto its closed label set. Cross-diagnosis freezes the transcript, strips the author’s own diagnosis, and asks another model to diagnose from that transcript.

An earlier two-model pass used gpt-4o-mini and Claude Haiku 4.5 (Zoe, 31 July), and a DeepSeek versus gpt-5.6-luna pass (24 July). Those are intermediate. The five-agent matrices below are the ones to cite. Model ids match Section 7.

### 8.1 AI Hospital, five doctors

Accuracy in percent.

| Information | A | B | C | D | E |
|---|---|---|---|---|---|
| Base prompt only | 0 | 0 | 0 | 0 | 0 |
| One random fact | 3 | 5 | 5 | 4 | 0 |
| Two most important, Claude-ranked | 38 | 35 | 38 | 21 | 8 |
| Two most important, GPT-ranked | 45 | 49 | 43 | 33 | 16 |
| Dialogue A | 53 | 66 | 63 | 62 | 55 |
| Dialogue B | 55 | 45 | 57 | 48 | 40 |
| Dialogue C | 66 | 70 | 51 | 55 | 44 |
| Dialogue D | 44 | 42 | 45 | 23 | 18 |
| Dialogue E | 30 | 36 | 30 | 19 | 11 |
| All facts | 79 | 76 | 81 | 57 | 27 |

On this dataset the vignette alone is not enough for anyone. Two curated facts recover a large part of the all-facts accuracy for A, B, and C (all-facts 79 / 76 / 81; two GPT-ranked facts 45 / 49 / 43) and much less for D and E. Dialogues sit between those two conditions. The best dialogue cells are not all on the diagonal: B reading A is 66, A reading C is 66, B reading C is 70, while B reading B is 45 and C reading C is 51. E’s dialogues are the weakest source for every reader, including E.

The older DeepSeek versus luna 2 × 2 (100 cases per cell) is a different model pair and should not be averaged with this table: DeepSeek’s own dialogue 40% (luna reading it 48%); luna’s own dialogue 51% (DeepSeek reading it 69%). The older gpt-4o-mini versus Haiku 2 × 2 was lower still (about 0.28–0.33).

Failure notes from Zoe, after reading individual AI Hospital transcripts:

- The common miss is a correct main diagnosis with a missed co-diagnosis. AI Hospital cases often have two to four concurrent labels, and a hit on the main one is still graded down.
- Nearby anatomy and similar mechanisms get swapped (for example Achilles insertion tendinitis versus retrocalcaneal bursitis).
- One symptom with two branches (jaundice as hepatitis versus a pancreatic head mass) is often unresolved because the interview never separates them.
- Some records list more diagnoses than a short interview can cover. One reviewed case had 17 labels.
- Models sometimes refuse to diagnose when the expected study is missing even though another study in the record is informative. In one thyroid case both models waited for an ultrasound and TI-RADS grade; the record had a contrast CT that already suggested malignancy, and the reporter correctly said there was no ultrasound.

### 8.2 3MDBench, five doctors, fact conditions

The newest 3MDBench table in the notebook is the fact-condition block. The notebook states that the dialogue cross for this five-doctor setup had not been entered there. The group’s account is that the remaining runs did finish on CARC and were not copied into the repository. This report therefore quotes only the cells written in the notebook. Anyone continuing the project needs the CARC logs before citing a 3MDBench dialogue matrix.

Accuracy in percent. Images were not shown to the models.

| Information | A | B | C | D | E |
|---|---|---|---|---|---|
| Base prompt only | 2 | 38 | 13 | 32 | 13 |
| One random fact | 33 | 28 | 34 | 30 | 22 |
| Two most important, Claude-ranked | 77 | 77 | 75 | 65 | 46 |
| Two most important, GPT-ranked | 79 | 75 | 73 | 62 | 45 |
| All facts | 83 | 82 | 81 | 75 | 53 |

Two top-ranked facts are close to the full record for A, B, and C (GPT-ranked 79 / 75 / 73 versus all-facts 83 / 82 / 81). D gains more from the rest of the record (62–65 → 75). E remains near half even with all facts (53).

The older text-only 2 × 2 (gpt-4o-mini vs Haiku) did include dialogues: GPT transcript diagnosed at 0.57 by GPT and 0.63 by Haiku; Haiku transcript diagnosed at 0.67 by GPT and 0.69 by Haiku. DeepSeek versus luna on 3MDBench was higher and almost flat (67–70% in all four cells). Those pairs are not the five-doctor system.

Zoe’s 3MDBench failure notes, from the two-model transcripts:

- Dermatology confusions are the strongest pattern and line up with dropping images. Herpes versus eczema is the example: the text describes grouped clear vesicles, and the picture is what separates them. The doctors also stopped after a few questions and did not ask the history items that would have helped.
- Some “errors” are scoring strictness. Dental calculus plus gingivitis was marked wrong against a gold label of dental calculus alone.
- Single-mechanism, concrete findings (nail dystrophy, abscess) were stable hits, as were simple single-diagnosis cases on AI Hospital.

Across both datasets, textbook single-cause cases are where the models agree and succeed. Multi-morbidity and missing modalities are where they fail together. When the task is the bottleneck, swapping the reader changes less than the case does.

A partial local snapshot of this run used to be at `clinic_experiment/outputs/mdbench/X0/`. It stopped early and did not match the table above, so it was removed from the tree. Do not restore it and cite it as the 3MDBench result.

---

## 9. Why the project stopped

The last experiment after the September synthesis was run to completion on CARC. The results were not what would be needed to keep a paper going, and Prof. Liu stopped the work. The newest tables in the notebook are the record we have of that period: the five-doctor MediQ matrix, the AI Hospital matrix, the 3MDBench fact-condition matrix, and the Claude/GPT fact-rank overlap.

Taken together, those tables support a narrower claim than the one we started with, and not a clean positive result:

- On MediQ, handing a strong model the full record, or even two facts that another strong model ranked as most important, beats an interactive interview. On AI Hospital the full record is above the interviews (about 76–81, against dialogue cells mostly between 45 and 70). Two ranked facts on AI Hospital are about 35–49, below most of those dialogue cells.
- On 3MDBench, two ranked facts are already close to the full record for the largest models. That matches Hypothesis 1 in the 14 August note (“most important facts are sufficient”). The five-doctor dialogue cross for this dataset was not written into the notebook, so these fact cells do not show whether an interview would have reached the same accuracy, and they do not show that agents fail to retrieve those facts (Hypothesis 2). The retrieval-gap experiment sketched at the end of the notebook was not completed as a separate study.
- Interviewer and diagnostician do interact. The 2 × 2 interaction contrast of 0.17 is real on that pair. The 5 × 5 matrices show the interaction is pair-specific: some transcripts help a different reader more than they help their author, and some transcripts (E on AI Hospital) help nobody. That is enough to say self-play confounds interviewing with diagnosis. It is not a stable ranking of interviewers, and it is not a training method.
- Small local models (the 4B and 8B doctors) remain far behind on every information condition, including the full record. A large part of the spread is diagnostic competence given the same text.

No further experiment was run after that CARC job.

---

## 10. Written down, not run

The notebook ends with two design notes that should not be described as results.

**Cross-play evaluation and training.** After the 2 × 2, there is a proposal (in the notebook under the heading that begins “Who Asked, Who Diagnosed?”) to split every agent into an interaction policy and a diagnosis model, evaluate the full matrix, define portable acquisition value, diagnostic robustness, and a compatibility residual, and train the interviewer against a random frozen diagnostician rather than against itself. Suggested theory pieces were a non-identifiability argument for the self-play diagonal, a “no universal interviewer” claim, and a cross-play generalization bound. Suggested mechanism ablations were raw dialogue, patient answers only, canonical facts, full information, and evidence coverage. None of that training or theory was carried out. The matrices in Sections 7 and 8 are the empirical fragment that motivated the proposal.

**Fact-tier sufficiency protocol.** A later outline proposes a gold set of “most important” facts, an agent that must extract those facts during the interview, and a comparison of gold facts versus agent-extracted facts, with a target of reaching 85% of all-facts accuracy from the top tier alone. The accuracy tables already compare two ranked facts against all facts and against dialogues. The agent-extraction arm, the recall of gold facts inside those extractions, and the turn-count threshold were not run as that protocol.

Other items that were specified and left unfinished: human adjudication of 50 LLM judgments, a clinician check of the patient simulator, secondary SMART endpoints as a formal analysis, sample-size calculations for a 4,000-encounter trial, pre-registration, and embedding features as a robustness check on Q-learning.

---

## 11. Appendix: sepsis side exploration

In mid-July Zoe connected [SepsisAgent](https://github.com/FreedomIntelligence/SepsisAgent) to the public ICU-Sepsis MDP benchmark as an external check, separate from the dialogue trial. The reference policies on that benchmark came out at about 0.781 (random), 0.783 (expert), and 0.875 (π*). The adapter turns each of 713 non-terminal discrete states into a prompt, caches `state_id → action_id`, and evaluates return and survival locally from the cache.

CARC produced 114 of 713 valid actions with SepsisAgent-4B. On those 114 states the policy was fluid-heavy, used vasopressors sparingly, and raised vasopressor intensity as SOFA increased. A V100 vLLM backend failed and the script fell back. There is no full-benchmark score. This appendix is only so the cache scripts are not mistaken for a finished evaluation.

---

## 12. Continuing from this repository

Read this file, then the PDF. Stay on `cross-diagnosis`. The PDF is the lab notebook (newest tables first). This file is the narrative and the warning about which files not to cite. `LICENSE` states what may be reused.

**If the goal is to reproduce a number in this report**

| Number | Where it comes from | What is in git |
|---|---|---|
| 50-case arm grid, baseline 0.26, closed-loop 0.48, lookup 0.56 | Notebook, 12 June; code from the old `eval` line, now in this tree | SLURM and configs under `smart_trial/config/eval/`. Those configs point at `data/all_dev_good.jsonl`, which is not in this tree. The encounter logs for that grid are not what `clinic_experiment/outputs/` contains. |
| n = 200 Q-learning / A-learning | `smart_trial/q_learning/RESULTS.md` | Re-run `python -m smart_trial.q_learning.run_main_analysis` if `smart_trial/outputs/encounters.jsonl` on that branch is the same 200 encounters. Quote the caveats in that file with the point estimates. |
| MediQ, AI Hospital, 3MDBench matrices | Newest tables in the PDF | Raw JSONL is on CARC. The partial local `mdbench/X0` copy was removed because it did not match these tables. |
| Fact-coverage means near 0.05–0.33 | Older notebook entries | The scorer omitted stem facts. Do not build a new claim on those means until the logs are re-scored. |

**If the goal is to extend the science**

The SMART simulator is usable as infrastructure: fixed arms, known propensities, JSONL trajectories, and a Q-learning / A-learning stack with an oracle check. A new trial needs a larger sample than 200 before the regime estimate is more than provisional, a patient who can disclose studies and labs, and a pre-declared endpoint that is not an in-sample lookup. The 50-case closed loop (0.48) is the deployment-style number; the lookup (0.56) and the Q-learning plug-in (0.525) are not.

The cross-play matrices are the other usable product. A continuation that only adds more self-play accuracy will not answer the question the last run left open: whether an interviewer has value that survives a change of diagnostician, after the facts in the transcript are held fixed. The unfinished piece is the mechanism split in Section 10 (raw transcript versus canonical facts versus full record), on logs that already exist on CARC. Training a cross-play interviewer was proposed and not started.

**Practical entry points on the current tree**

```bash
pip install -r smart_trial/requirements.txt
python run_encounter.py --n 10 --seed 42
python -m smart_trial.scripts.summarize_encounters
python -m pytest smart_trial/tests/ -q
```

Tests use mock models unless `SMART_TRIAL_LIVE_API=1`. API keys live in `.env` (see `.env.example` if present on the branch; one `detail` commit deleted a copy of that example and should not be treated as a reason to commit secrets). CARC jobs for the Claude eval are documented in `smart_trial/config/eval/README.md`. MediQ ablations: `mediq_experiment/README.md`. Clinic pipeline: `python -m clinic_experiment.run_pipeline --dataset ai_hospital` and `--dataset mdbench`.

The default branch is `cross-diagnosis`. `smart-trial` is the earlier snapshot. Point readers at this file and at `LICENSE`.

---

## 13. Documents in this folder

| File | Role |
|---|---|
| `doc/Project_Report.md` | This report. |
| `doc/Agent Trial SMART Design.pdf` | Lab notebook from the start of the project through the last recorded tables. Newest experimental tables are at the front. Also contains the original SMART specification, the 12-day plan, the Q-learning analysis plan, and the unimplemented CrossDx and fact-tier proposals. |
| `LICENSE` | Copyright, research-use disclaimer, and third-party terms. Repository root, not this folder. |

---

## 14. Copyright and disclaimer

The full text is `LICENSE` at the repository root.

This repository is a research record. It is not a medical device, and its outputs are not a diagnosis, a treatment recommendation, or clinical advice. Nothing in the code, the notebook, or this report has been validated for use with patients.

The original code, this report, and the lab notebook keep their authors' rights. Viewing the repository and citing it with attribution are allowed. Copying the original materials into another project, commercial use, and clinical use are not granted by this repository.

`src/` and the MediQ / iMEDQA cases stay under CC-BY 4.0 from their original authors. CRAFT-MD, AI Hospital, 3MDBench, and the MedRAG textbooks stay under their own upstream terms. This repository does not relicense those materials. Model weights are not included.
