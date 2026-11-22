# Pre-registered predictions and outcomes

The experiments ran in rounds on Kaggle in 2026 (round 6 contained no Hawkes experiments). Before each round, its predictions and
pass/fail thresholds were written into the project plan and into the round's config file (`configs/`). They were not
changed afterwards. Failed predictions are reported as failures in the paper. There is no third-party timestamp for
these records; the configs are the record.

The outcomes listed here are the ones recorded when each round finished. The paper's numbers are regenerated from
`paper/data/` by `paper/scripts/make_numbers.py` and are authoritative where they differ. IDs for rounds 3–5 are added here for reference; the others are the original
labels.

## Rounds 1–2: does attention implement the kernel? (`configs/hawkes_pilot.yaml`, `configs/hawkes_main.yaml`)

| ID | Prediction | Outcome |
|---|---|---|
| H1 | Late-context gap fraction (share of the information beyond a Poisson rate still missing) ≤ 0.3 | Pass (pilot 0.056; 0.048–0.060 across the eight final models) |
| H2 | At least one head has an exponential time profile (fit R² ≥ 0.7) whose decay tracks the true β, or several heads form a fixed bank of decays | Fail: no head has an exponential profile whose decay tracks β (best exponential fits R² 0.39–0.74); the model without time-aware attention is as accurate |
| H3 | α, log β and the live intensity are linearly decodable mid-network (R² ≥ 0.5) | α (R² 0.83) and the intensity (0.96) pass; the true log β is less identifiable (round 5) |
| H4 | The gap grows as α → 1, most sharply on the unseen range 0.90–0.98 | Not supported: the gap fraction on the unseen near-critical range is 0.017–0.034, below the 0.048–0.060 overall (appendix, α bins) |
| H5 | Removing the null attention slot degrades the model or forces a workaround | Fail: the no-null model matches the others |
| — | Intensity tracking: within-sequence Spearman ≥ 0.8 | Pass (0.97–0.98 pooled; 0.92–0.95 within sequences) |
| — | A head whose span adapts to β (Spearman ≥ 0.5) | Appeared in one seed only; dropped |

## Round 3: are the parameter estimates causal? (`transplant` stage)

| ID | Prediction | Outcome |
|---|---|---|
| T1 | Editing the decoded estimate of α, μ or β moves the forecast with slope ≥ 0.3 relative to the true process | Pass (slopes 0.7–1.2; random edits move it 4–9× less) |
| T2 | Specificity: the change follows the edited parameter's own pattern with correlation ≥ 0.5 | Fail for α in every run (0.46–0.50); β specific in 4 of 5 runs, μ in 2 of 5 |

## Round 4: fair baselines (`baseline` stage; third seed; power-law kernel)

| ID | Prediction | Outcome |
|---|---|---|
| B1 | Within 0.02 nats of the Bayes-optimal in-context forecaster from 8 events on | Pass: within 0.002 nats at every context length 2–128 in all five exponential runs |

## Round 5: posterior, not parameters (`posterior` stage; two more power-law seeds)

| ID | Prediction | Outcome |
|---|---|---|
| P-a | Posterior-mean R² ≥ 0.85 | Pass in all 8 runs (0.99 / 0.99 / 0.97–0.98 for log μ, α, log β) |
| P-b | Posterior-mean R² ≥ true-value R² + 0.05 | Pass |
| P-c | True-value R² ≥ 90% of the information ceiling | Pass for exponential runs (93–98%); fail for power-law log β (89%) and γ (42–62% of a tiny ceiling) |

## Round 7: Bayesian or just accurate? (`configs/hawkes_r7.yaml`)

| ID | Prediction | Outcome |
|---|---|---|
| M1 | On data from the other kernel family, KL(Bayes under own prior ‖ model) < 0.5 × KL(Bayes under true prior ‖ model) in ≥ 3/4 of models | Pass 8/8 |
| M2 | Decoded α correlates more with the own-prior posterior mean than with the true value | Pass (uninformative: same pattern in distribution) |
| I1 | Swapping the 3 probe directions between sequences gives transfer ≥ 0.5 and ≥ 3× a random subspace | Fail (transfer 0.002–0.008) |
| K1 | Decayed counts at 7 fixed rates suffice for the posterior means (MLP R² within 0.03 of the best residual probe) | Fail (0.84 / 0.64 / 0.11–0.17 against 0.99 / 0.99 / 0.97) |
| K2 | A linear map of those counts recovers ≥ 0.7 of the loss that count and time features alone cause | Pass 7/8 |
| FIM-PP | Exploratory, no threshold: is the pretrained FIM-PP model near Bayes in this setting? | Far from Bayes with few context sequences (appendix) |

## Round 8: how the posterior is computed (`configs/hawkes_r8.yaml`)

| ID | Prediction | Outcome |
|---|---|---|
| M1 | For every misspecified model, KL(true) − KL(own) > 0 and KL(own)/KL(true) < 0.5 at the 95% level (bootstrap over sequences) | Pass 8/8 |
| P1 | Decayed counts plus pooled log-traces predict the posterior means within 0.05 R² of the best residual probe | Pass |
| P2 | Pooled traces decodable from the residual stream (R² ≥ 0.8), with a gain ≥ 0.1 after block 0 | Fail 3/5 (two models already pool in block 0) |
| P3 | Exploratory: heads with near-uniform attention over past events (total variation < 0.25) | Found in every model |
| P4 | A linear map of [counts, pooled traces] reconstructs a later block ≥ 0.1 better than [counts] alone | Pass |
| I2 | An 8-dimensional DAS subspace reaches held-out transfer ≥ 0.5 and ≥ 3× a random subspace | Pass 5/5 |
| I3 | That subspace contains ≥ 30% of the probe's posterior subspace | Fail (it aligns with the encoding directions instead) |

## Round 9: controls (`configs/hawkes_r9.yaml`)

| ID | Prediction | Outcome |
|---|---|---|
| A1 | Mean-ablating the near-uniform heads raises late-context loss ≥ 3× the mean of ablating the same number of other heads, and above all 10 draws | Fail (holds in 1 of 5 models, the one without time-aware attention) |
| A2 | After that ablation, pooled traces are ≥ 0.05 R² less decodable than after ablating other heads | Not interpretable: the analysis probed the intact baseline one block earlier than the ablated model (disclosed in the paper's appendix) |
| D1 | DAS trained toward an unrelated posterior reaches transfer ≤ 0.3 where the right target reaches ≥ 0.8 | Fail (0.33–0.43; right target 0.96–0.98) |
| D2 | The 3 DAS coordinates predict all three posterior means with R² ≥ 0.8 | Pass 4/5 |
| D3 | Exploratory: DAS subspaces after blocks 0–2 are aligned | Overlap 0.34 (chance 0.02) |
