# Triagegeist: Multi-Modal Emergency Triage Acuity Prediction

**Ensemble gradient boosting with clinical feature engineering, NLP chief complaint analysis, and comprehensive demographic bias detection**

## Clinical Problem Statement

Every year, approximately 130 million patients visit emergency departments across the United States alone. In Nordic healthcare systems — including Finland, where the Laitinen-Fredriksson Foundation is based — nurse-led triage is the standard of care, with each hospital's ED handling 40,000–80,000 visits per year. Within seconds of arrival, a triage nurse must assign each patient an Emergency Severity Index (ESI) level from 1 (resuscitation) to 5 (non-urgent) — a decision that determines how long they wait, what resources they receive, and ultimately whether they survive.

This decision is made under extreme cognitive load: a busy ED nurse may triage 50+ patients per shift, each requiring a rapid synthesis of vital signs, chief complaint, medical history, and clinical gestalt. The consequences of error are asymmetric and severe. **Undertriage** — assigning a lower acuity than warranted — is the critical patient safety threat: a septic patient triaged as ESI 4 instead of ESI 2 may deteriorate in the waiting room before reassessment. Studies in Finnish EDs have shown undertriage rates of 10–15% for patients who subsequently required ICU admission.

Published literature documents significant inter-rater variability in ESI scoring (kappa 0.60–0.80, Gilboy et al., 2020). Systematic undertriage has been identified in elderly patients, non-English speakers, patients with atypical presentations, and certain racial/ethnic groups (Obermeyer et al., 2019). The ESI algorithm itself, while structured, relies heavily on subjective assessments like "Does this patient look sick?" — a question that invites both expertise and bias.

**This project builds an AI-powered clinical decision support system** that predicts ESI triage acuity from structured intake data, free-text chief complaints, and patient comorbidity history, while simultaneously surfacing systematic bias patterns that could harm vulnerable populations.

## Data

All data is provided by the Laitinen-Fredriksson Foundation via the Triagegeist Kaggle Competition (Olaf Yunus Laitinen Imanov, 2026). The dataset contains synthetic emergency department records designed to mirror real-world triage workflows. We integrate three tables by `patient_id`:

- **Structured intake** (80,000 train / 20,000 test): 37 features including vital signs (HR, BP, RR, temperature, SpO2, GCS), demographics (age, sex, language, insurance), arrival context (mode, time, shift), ED utilization history, and the target variable `triage_acuity` (1–5).
- **Chief complaints** (100,000 records): free-text clinical presentation descriptions and body system category labels (14 systems from cardiovascular to psychiatric).
- **Patient history** (100,000 records): 25 binary comorbidity flags spanning cardiovascular, respiratory, metabolic, neurological, and psychiatric conditions.

Missing data is clinically realistic: blood pressure measurements are absent for ~5.2% of patients (consistent with real-world rates for walk-in patients not immediately assessed), respiratory rate for ~3.8%, and temperature for ~0.7%.

## Methodology

### Clinical Feature Engineering

We engineer 50+ clinically-motivated features grounded in emergency medicine literature, organized into six categories:

**Vital sign abnormality flags (18 flags).** Binary indicators for clinically significant thresholds: hypotension (SBP < 90 mmHg), severe hypotension (SBP < 70), tachycardia (HR > 100), severe tachycardia (HR > 130), tachypnea (RR > 20), hypoxia (SpO2 < 92%), altered mental status (GCS < 15), and more. These thresholds map directly to ESI decision branch points in the published algorithm (Gilboy et al., 2020). The aggregate count `num_abnormal_vitals` serves as a composite acuity burden score.

**Sepsis screening scores.** We compute the quick Sequential Organ Failure Assessment (qSOFA) score (range 0–3: SBP ≤ 100, RR ≥ 22, GCS < 15) per Sepsis-3 consensus definitions (Singer et al., 2016). Additionally, we approximate SIRS criteria (temperature, heart rate, respiratory rate abnormalities). These scores are well-validated predictors of clinical deterioration.

**Comorbidity risk composites.** Patient history features are aggregated into clinically meaningful risk groups: cardiovascular risk (6 conditions including hypertension, heart failure, CAD, stroke history), respiratory risk (asthma + COPD), metabolic risk (diabetes types 1 and 2, obesity), and psychiatric risk (depression, anxiety, substance use disorder). These composites reduce dimensionality while preserving clinical interpretability.

**Age-vital interactions.** Features such as `age × shock_index`, `age × NEWS2`, and `age × qSOFA` capture the clinical principle that elderly patients decompensate faster for equivalent vital sign derangements — a 35-year-old with a shock index of 1.0 has a different prognosis than an 85-year-old with the same value.

**NLP pipeline.** Two complementary approaches extract signal from chief complaint text: (1) TF-IDF encoding (150 features, unigrams + bigrams, sublinear term frequency) captures the full statistical text signal, and (2) 16 clinically-curated keyword regex patterns detect high-risk presentations that should trigger higher acuity, including chest pain, seizure, stroke, suicidal ideation, respiratory distress, and GI bleeding. Patients matching "mild" keywords (advice, prescription refill, follow-up) are flagged separately.

**Target-encoded identifiers.** Out-of-fold target encoding for the 50 triage nurses and 5 site IDs captures systematic inter-rater variability without introducing data leakage — a feature that directly models the clinical problem this tool aims to address.

### Ablation Study: Each Component's Contribution

To justify every component of our pipeline, we trained LightGBM models on cumulative feature subsets using 3-fold CV:

| Feature Group | Features | QWK | Marginal Gain |
|:---|:---:|:---:|:---:|
| Vitals only | 22 | 0.9261 | — |
| + Demographics | 34 | 0.9303 | +0.0042 |
| + Patient history | 59 | 0.9305 | +0.0002 |
| + Clinical flags (qSOFA, SIRS) | 126 | 0.9539 | +0.0235 |
| + NLP (TF-IDF + keywords) | 276 | 0.9980 | +0.0441 |

Each layer provides measurable improvement. Notably, clinical flags and NLP together contribute the marginal gains that separate a good model from a near-perfect one, validating our multi-modal approach.

### Model Architecture

We train two complementary gradient boosting models with 5-fold stratified cross-validation:

- **LightGBM**: 3,000 estimators, learning rate 0.03, 127 leaves, early stopping at 150 rounds. Leaf-wise growth excels on sparse TF-IDF features.
- **XGBoost**: 2,000 estimators, learning rate 0.03, max depth 8, early stopping at 150 rounds. Level-wise growth provides complementary regularization.

The models are ensembled via weighted averaging, with optimal weights (75% LightGBM, 25% XGBoost) determined by grid search on out-of-fold predictions. Ensembling reduces prediction variance — a critical property for clinical decision support where overconfident wrong predictions are dangerous.

## Results

The ensemble achieves strong out-of-fold performance across all 80,000 training patients:

| Metric | LightGBM | XGBoost | Ensemble |
|:-------|:--------:|:-------:|:--------:|
| Accuracy | 99.49% | 99.36% | 99.47% |
| Weighted F1 | 99.49% | 99.36% | 99.47% |
| Quadratic Weighted Kappa | 0.9975 | 0.9969 | 0.9974 |

**Calibration analysis** shows the predicted probabilities are well-calibrated (Expected Calibration Error < 0.05), meaning a 70% confidence prediction corresponds to approximately 70% true positive rate. This is clinically essential: a nurse receiving a model prediction needs to trust the associated confidence level. We include per-class reliability diagrams showing calibration across all five ESI levels, with ESI 1 and ESI 5 (the extremes) showing the tightest calibration — exactly where clinical certainty matters most.

**Top predictive features**: NEWS2 score (Pearson r = −0.81 with acuity), GCS total, SpO2, shock index, respiratory rate, pain score, and heart rate. The dominance of NEWS2 validates the National Early Warning Score as an effective aggregate acuity signal. Critically, triage nurse target encoding also ranks highly, confirming measurable inter-rater variability across 50 nurses — the very problem clinical decision support aims to mitigate.

**Chief complaint NLP** contributes meaningful signal: keyword flags for cardiac symptoms, neurological presentations, and trauma rank among the top features, while TF-IDF features capture subtler textual patterns.

**Undertriage safety**: overall undertriage rate is 0.39%, and undertriage of critical ESI 1–2 patients is 0.37% — well within clinically acceptable thresholds for a decision support tool. The model's confidence on incorrect predictions is measurably lower than on correct ones, enabling a "flag for senior review" mechanism on low-confidence cases.

## Bias Analysis

We perform comprehensive demographic bias analysis on OOF predictions across five dimensions:

**Bias delta** (mean predicted − mean actual acuity) quantifies systematic over/under-triage per demographic group. Across sex, age bands, language, insurance type, and arrival mode, no subgroup shows a bias delta exceeding ±0.02 acuity levels — indicating the model does not systematically disadvantage any demographic group.

**Chi-squared significance testing** evaluates whether accuracy differences across demographic groups are statistically significant. We apply Bonferroni correction for multiple comparisons to reduce false discovery risk.

**Intersectional analysis** identifies the highest-risk subgroups at the intersection of sex × age group × language — surfacing compound disadvantage invisible in single-dimension analysis. For example, elderly non-English-speaking females represent a clinically vulnerable intersection where triage errors have the highest morbidity impact.

**Undertriage monitoring** specifically tracks the rate at which truly high-acuity (ESI 1–2) patients are predicted as lower acuity. This is the most dangerous clinical error mode: a missed ESI 1 patient may arrest without intervention. Our model maintains ESI 1–2 undertriage at 0.37%, and overall undertriage at 0.39%.

**Overtriage analysis** quantifies the opposite error — assigning higher acuity than warranted. While overtriage wastes resources, it is clinically safer than undertriage. Our model's overtriage rate remains below 1%, suggesting it would not significantly increase ED resource burden.

In the Finnish healthcare context, where universal coverage eliminates insurance-driven access disparities, equity analysis centers on age, sex, and language — particularly relevant for immigrant populations navigating triage in a non-native language. The Foundation's focus on Nordic healthcare systems makes this bias-aware approach essential for responsible AI deployment in public health infrastructure.

## Toward Clinical Deployment

Before any real-world use, the following validation pathway is necessary: (1) retrospective validation against MIMIC-IV-ED or equivalent real-world data, (2) prospective silent validation in a partner ED where the model runs alongside but does not influence triage decisions, (3) fairness audit with pre-specified equity metrics matching Finnish population demographics, (4) probability calibration verification across clinical subgroups, and (5) integration testing within existing ED information systems (Epic, Cerner, or THL/Kanta systems used in Nordic healthcare).

The Laitinen-Fredriksson Foundation's partnerships with Northern European hospital networks position this work for such translation. A key advantage of gradient boosting over deep learning for this use case is latency: predictions are generated in <10ms, fast enough for real-time triage workflow integration without disrupting the nurse's decision cadence.

## Limitations

1. **Synthetic data** — Performance may not transfer to real clinical environments.
2. **NEWS2 leakage** — NEWS2 is itself a clinical scoring system and may partially encode existing triage decisions.
3. **NLP depth** — TF-IDF captures keyword signal but misses semantic nuance; ClinicalBERT would improve understanding.
4. **No temporal validation** — Time-based splits would better simulate prospective deployment.
5. **No external validation** — Generalization to other institutions is unknown.
6. **Single-snapshot triage** — Real triage involves reassessment; our model predicts initial acuity only.

## Reproducibility

- **Kaggle Notebook**: runs end-to-end without errors
- **GitHub**: [github.com/ladyFaye1998/triagegeist](https://github.com/ladyFaye1998/triagegeist) — full source, modular architecture, setup instructions
- **Interactive Demo**: [ladyfaye1998.github.io/triagegeist](https://ladyfaye1998.github.io/triagegeist/) — browser-based triage predictor
- Random seed 42 fixed for all stochastic operations

## References

- Gilboy, N. et al. (2020). *Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4*. AHRQ Publication No. 20-0045.
- Singer, M. et al. (2016). *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)*. JAMA, 315(8), 801–810.
- Royal College of Physicians (2017). *National Early Warning Score (NEWS) 2*.
- Fernandes, M. et al. (2020). *Clinical Decision Support Systems for Triage in the Emergency Department*. Artif. Intell. Med., 102, 101762.
- Obermeyer, Z. et al. (2019). *Dissecting racial bias in an algorithm used to manage the health of populations*. Science, 366(6464), 447–453.
- Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist. Kaggle.
