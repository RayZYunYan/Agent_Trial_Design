# Q-learning Analysis Package

Offline analysis of SMART trajectories already logged under
`smart_trial/outputs/encounters/`. Implements the plan in three layers:

```
load.py        encounters JSONL  →  pandas DataFrame
features.py    DataFrame         →  H1 / H2 / H3 hand-crafted feature matrices
models.py      sklearn-style StageRegressor with arm × state interactions
q_learning.py  backward induction → fitted Q̂1/Q̂2/Q̂3 + optimal rule per row
bootstrap.py   percentile (regular targets) + m-out-of-n (value of π̂)
a_learning.py  doubly-robust cross-check  [STUB — W3]
oracle.py      synthetic recovery validation  [STUB — W2]
rules.py       extract DTR rules, static-vs-adaptive comparison
figures.py     interaction plot + value-comparison bar (working examples)
run_main_analysis.py    end-to-end pipeline
```

## Run

```bash
# from repo root
python -m smart_trial.q_learning.run_main_analysis
```

Outputs land in `smart_trial/q_learning/outputs/`.

## What's working vs stubbed

Working end-to-end on the data already logged:
- Trajectory loading + flat DataFrame
- Feature engineering (H1, H2, H3) per plan §1
- Linear Q-learning backward induction per plan §2
- Rule extraction + adaptive-vs-static value comparison (plan §5 deliverables 1, 2, 3)
- Percentile bootstrap (regular) and m-out-of-n bootstrap (non-regular) CIs per plan §4
- Interaction plot + value-comparison bar (plan §5)

Stubbed with detailed docstrings (intern fills in per plan timeline):
- `a_learning.py` — algorithm spelled out in module docstring, W3 deliverable
- `oracle.py` — synthetic DGP outline in docstring, W2 deliverable

## Caveats

- Sample size is the binding constraint right now: only 12 encounters logged
  for one case_id. Running the pipeline will produce point estimates and CIs,
  but the linear regressions have far more parameters than data points until
  many more encounters are run. Do the oracle validation in parallel.
- All randomization propensities are known by design — the trial is randomized
  within each stage's pool. Use those known weights in A-learning rather than
  estimating propensities.
- Feature engineering deliberately stays interpretable. Embedding features are
  a §7 robustness ablation; do not put them on the main path.
