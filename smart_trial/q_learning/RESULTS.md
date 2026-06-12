# Statistical Results: Optimal DTR Estimation by Q-learning and A-learning
## Two-Stage SMART, n = 200 Simulated Encounters (branch `2stage`, analysis of 2026-06-12)

All numbers reproduce via `python -m smart_trial.q_learning.run_main_analysis` plus the
companion extraction script; source CSVs in `smart_trial/q_learning/outputs/`.

---

### Table 1. Cohort and design descriptives

| Characteristic | Value |
|---|---|
| Encounters analyzed, n | 200 |
| Outcome: correct diagnosis, mean(Y) | 0.260 |
| Stage-1 responders (R₁ ≥ 6), n (%) | 24 (12.0%) |
| Stage-1 arms, n (observed Y̅) | A1a 75 (0.240) · A1b 67 (0.224) · A1c 58 (0.328) |
| Stage-2 arms, n (observed Y̅) | A2a 95 (0.274) · A2b 92 (0.250) · A2c 13 (0.231) |
| Observed Y̅ by response | non-responders 0.267 (n=176) · responders 0.208 (n=24) |
| Largest (A1,A2) cell | A1b→A2b: n=41, Y̅=0.220 |
| Best observed (A1,A2) cell | A1c→A2a: n=32, Y̅=0.406 |
| Sparsest cell | A1c→A2c: n=1 |

*Randomization: A1 uniform 1/3; A2 uniform over the feasible pool — responders {A2a,A2b,A2c}
at 1/3, non-responders {A2a,A2b} at 1/2. Propensities known by design.*

---

### Table 2. Estimated value of the optimal regime V(π̂) and comparators

| Estimand / method | Estimate | 95% CI | Notes |
|---|---|---|---|
| V̂(π̂), Q-learning plug-in (in-sample) | 0.525 | — | mean of stage-1 optimal pseudo-outcomes |
| V̂(π̂), Q-learning cross-fitted (5-fold) | 0.525 | (0.423, 0.699) | percentile bootstrap, B=200 |
| V̂(π̂), m-out-of-n bootstrap | 0.525 | (0.416, 0.697) | m = n³ᐟ⁴; non-regularity robust |
| V̂(π̂), CMS projection interval | 0.525 | (0.253, 0.720) | adaptive to near-ties |
| V̂(π̂), A-learning plug-in | 0.441 | — | blip-based backward induction |
| Best static regime (A1c→A2a) | 0.415 | — | max over 9 feasible (a₁,a₂) pairs |
| Adaptive − best-static gap | 0.110 | — | was 0.016 in the 3-stage analysis |
| Observed mean outcome, mean(Y) | 0.260 | — | benchmark |

*Interpretation caveat: both plug-in values sit well above mean(Y) and the best observed cell
(0.406). The oracle simulation (Table 5) shows a residual upward bias of ≈ 0.2 at this n from
argmax-over-extrapolated-predictions; treat V̂ levels as optimistic and the Q–A and
adaptive–static **differences** as the more reliable signals.*

---

### Table 3. Adjusted mean outcome by arm (regression-standardized over the sample)

**Stage 1** — value of assigning a₁ to everyone, then acting optimally at stage 2:
E_n[max_{a₂} Q̂₂(H₂(a₁), a₂)] (Q-learning).

| Stage-1 arm | Q-learning | rank |
|---|---|---|
| A1a | 0.365 | 2 |
| A1b | 0.279 | 3 |
| A1c | **0.497** | 1 |

**Stage 2** — adjusted mean outcome of assigning a₂ over feasible subjects:
E_n[Q̂₂(H₂, a₂)] (Q) and E_n[ν̂(H₂) + H₂ψ̂_{a₂}] (A); A-learning blip = average advantage
over reference arm A2a.

| Stage-2 arm | n feasible | Q-learning | A-learning | A-learning blip vs A2a (mean / median) |
|---|---|---|---|---|
| A2a (ref) | 200 | 0.254 | 0.259 | 0 (ref) |
| A2b | 200 | 0.276 | 0.253 | −0.035 / −0.046 |
| A2c | 24 | 0.304 | 0.210 | +0.041 / +0.066 |

*Per-subject rules (Table 4) are argmaxes of subject-specific predictions, not of these
averages — the methods can agree per subject while ordering the population means differently.
The A2c column rests on 24 feasible (13 treated) subjects; its mean/median blip discrepancy
across methods reflects that sparsity.*

---

### Table 4. Estimated optimal decision rules by subgroup — Q-learning vs A-learning

Modal recommendation within subgroup (category × R₁ response); Q̂/Â = mean predicted value
at the recommended regime.

| Subgroup | n | Q: π̂₁ | Q: π̂₂ | Q̂ | A: π̂₁ | A: π̂₂ | Â | Rules concur |
|---|---|---|---|---|---|---|---|---|
| Cardiology · non-resp | 49 | A1c | A2a | 0.469 | A1c | A2a | 0.445 | ✓ |
| Neuro · non-resp | 21 | A1c | A2a | 0.482 | A1a | A2a | 0.368 | π₂ only |
| Other · non-resp | 75 | A1c | A2a | 0.495 | A1c | A2a | 0.452 | ✓ |
| Pediatrics · non-resp | 20 | A1c | A2a | 0.464 | A1c | A2a | 0.452 | ✓ |
| Psychiatry · non-resp | 11 | A1c | A2b | 0.611 | A1c | A2b | 0.527 | ✓ |
| Cardiology · resp | 4 | A1a | A2c | 0.956† | A1c | A2c | 0.397 | π₂ only |
| Neuro · resp | 4 | A1a | A2c | 0.562† | A1a | A2c | 0.362 | ✓ |
| Other · resp | 10 | A1c | A2b | 0.846† | A1c | A2b | 0.436 | ✓ |
| Pediatrics · resp | 6 | A1a | A2b | 0.715† | A1c | A2a | 0.439 | ✗ |

† Responder-subgroup Q̂ values rest on n = 4–10 subjects and are not statistically supported;
the A-learning values (0.36–0.44) are the more conservative read.

**Headline regime (both methods):** for the 88% who do not respond at stage 1 —
**A1c at stage 1, then A2a at stage 2** — coinciding with the best observed cell
(A1c→A2a: Y̅ = 0.406, n = 32). Responder recommendations (switch to A2c/A2b) are directional
only.

---

### Table 5. Method agreement and estimator validation

**(a) Q-learning vs A-learning per-encounter rule agreement (n = 200)**

| Quantity | Agreement | 3-stage analysis (reference) |
|---|---|---|
| Stage-1 rule, π̂₁ | 0.875 | 0.735 |
| Stage-2 rule, π̂₂ | 0.935 | 0.540 |
| Both stages | **0.835** | 0.390 |

**(b) Oracle recovery — synthetic 2-stage DGP with known π\* (8 replicates per n)**

| n | agree_all | agree π₁ | agree π₂ | V̂ − E[Y] (value gap) |
|---|---|---|---|---|
| 50 | 0.100 | 0.248 | 0.545 | −0.331 |
| 100 | 0.176 | 0.360 | 0.559 | −0.341 |
| 300 | 0.188 | 0.317 | 0.523 | −0.220 |

*The negative value gap (V̂ above the achievable mean) quantifies the optimism bias noted in
Table 2. Recovery improves with n in the 2-stage design (it did not in the 3-stage version)
but remains far from 1 — rule estimates should be reported as provisional until the oracle
test passes (agree_all → 1, gap → 0).*

---

### Methods summary (one paragraph for the report text)

We estimated the optimal dynamic treatment regime π* = argmax_π E[Y^π] for a two-stage SMART
by (i) **Q-learning**: backward induction with a logistic stage-2 Q-function regressed on the
arm-interacted history design [H₂ | 1{A₂=a}·H₂], the stage-2 pseudo-outcome
ỹ₁ = max_{a∈𝒜₂(H₂)} Q̂₂(H₂,a), and stage-1 rules chosen by maximizing Q̂₂ at the
counterfactual history H₂(a₁); and (ii) **A-learning**: doubly-robust blip estimation
Q_t = ν_t(H) + Σ_a 1{A=a}·Hψ_a, with blips identified by propensity-centered regression of
residuals on (1{A=a} − π_a)·H using the known randomization probabilities (stage 1: 1/3;
stage 2: 1/3 responders, 1/2 non-responders). Feasible-set restrictions were enforced at every
argmax. Value inference used percentile, m-out-of-n (m = n³ᐟ⁴), and CMS-projection bootstrap
intervals (B = 200) to accommodate the non-regularity of the optimal-value functional.
Estimator validity was assessed by Q–A rule agreement and by recovery of a known regime on a
synthetic oracle data-generating process.

---

*Files: `value_ci.csv`, `rule_summary.csv`, `qa_agreement_summary.csv`, `arm_cells.csv`,
`coefficients.csv`, `static_vs_adaptive.csv`, `diagnostics.csv` under
`smart_trial/q_learning/outputs/`.*
