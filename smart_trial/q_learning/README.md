# Q-learning Analysis Package

Offline analysis of SMART trajectories logged under `smart_trial/outputs/`.
Implements statistical (SMART) Q-learning with A-learning cross-check and
oracle simulation validation.

## Run

```bash
# from repo root — auto-detects encounters.jsonl or encounters/ directory
python -m smart_trial.q_learning.run_main_analysis

# oracle recovery check
python -c "from smart_trial.q_learning.oracle import recovery_rate; print(recovery_rate([100,300], n_replicates=5))"
```

Outputs land in `smart_trial/q_learning/outputs/`.

## Pipeline

```
load.py         encounters JSONL (file or dir) → DataFrame
features.py     H1 / H2 + counterfactual overrides
models.py       reference-arm Ridge / logistic StageRegressor
q_learning.py   backward induction + counterfactual π̂ + cross-fitted V(π̂)
a_learning.py   doubly-robust blip estimator (Q/A cross-check)
oracle.py       synthetic DGP + recovery rate vs n
bootstrap.py    percentile, m-out-of-n, CMS projection CIs
diagnostics.py  n vs p, arm cell counts
run_main_analysis.py   end-to-end
```

## Design (two-stage SMART)

- Decision points: Stage 1 (A1a/b/c) and Stage 2 (A2a/b/c, pool gated by R1).
- The final diagnosis is delivered within Stage 2; there is no Stage 3.
- R2 (post-Stage-2 confidence) is post-treatment: kept in the DataFrame for
  descriptive analysis only, never used as a decision-state feature.

## Key improvements

- Counterfactual H2(a1) for stage-1 decisions
- Logistic Q at stage 2 (binary `diag_correct`)
- Reference-arm contrast coding (fewer collinear dummies)
- Single-file `encounters.jsonl` loading + category name mapping
- Cross-fitted V(π̂) to reduce in-sample optimism
- A-learning + oracle validation no longer stubbed

## Caveats

- Sample size remains the binding constraint for real data (see `diagnostics.csv`).
- Literacy features use `literacy_id` derived from the log `persona` block
  (`vocabulary_register` → literacy_F/I/C) or legacy `literacy_persona`.
- Counterfactual H holds R1 fixed when swapping A1 (documented structural assumption).
- Pre-redesign three-stage logs (with `stage3_arm`) still parse, but their
  outcomes were produced by Stage-3 arms and are not valid under the two-stage
  design — rerun encounters instead of mixing old and new data.
