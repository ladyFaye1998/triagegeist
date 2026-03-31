# Triagegeist: Multi-Modal Emergency Triage Acuity Prediction

**Stacked triple-boost ensemble (LightGBM + XGBoost + CatBoost → LR meta-learner) with clinical feature engineering, NLP chief complaint analysis, and comprehensive demographic bias detection**

## Clinical Problem Statement

In Nordic healthcare systems — including Finland, where the Laitinen-Fredriksson Foundation is based — nurse-led triage assigns each patient an Emergency Severity Index (ESI) level from 1 (resuscitation) to 5 (non-urgent) within seconds of arrival. This decision determines wait time, resource allocation, and patient outcomes.

The consequences of error are asymmetric: **undertriage** — assigning lower acuity than warranted — is a critical safety threat. A septic patient triaged as ESI 4 instead of ESI 2 may deteriorate before reassessment. Finnish ED studies report undertriage rates of 10–15% for patients who subsequently required ICU admission. Published literature documents inter-rater variability in ESI scoring (kappa 0.60–0.80, Gilboy et al., 2020), with systematic undertriage identified in elderly patients, non-native speakers, and certain racial/ethnic groups (Obermeyer et al., 2019).

**This project builds an AI clinical decision support system** that predicts ESI acuity from structured intake data, free-text chief complaints, and comorbidity history, while surfacing systematic bias patterns.

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

**NLP pipeline.** Two complementary approaches extract signal from chief complaint text: (1) TF-IDF encoding (500 features, unigrams + bigrams + trigrams, sublinear term frequency, min_df=3) captures the full statistical text signal, and (2) 16 clinically-curated keyword regex patterns detect high-risk presentations that should trigger higher acuity, including chest pain, seizure, stroke, suicidal ideation, respiratory distress, and GI bleeding. Patients matching "mild" keywords (advice, prescription refill, follow-up) are flagged separately.

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

### Model Architecture: Two-Level Stacked Ensemble

We employ a two-level stacking architecture for maximum predictive performance and model diversity.

**Level-1 Base Learners** (each trained with 5-fold stratified CV):

- **LightGBM**: 3,000 estimators, learning rate 0.03, 127 leaves, early stopping at 150 rounds. Leaf-wise growth excels on sparse TF-IDF features.
- **XGBoost**: 2,000 estimators, learning rate 0.03, max depth 8, early stopping at 150 rounds. Level-wise growth provides complementary regularization.
- **CatBoost**: 2,000 iterations, learning rate 0.05, depth 8, ordered boosting. Symmetric tree structure with ordered target statistics provides a third complementary decision boundary.

**Level-2 Meta-Learner**: A Logistic Regression model trained on the 15-dimensional OOF probability matrix (5 classes × 3 models) learns optimal non-linear blending. The meta-learner is itself trained with 5-fold OOF to prevent information leakage. This outperforms simple weighted averaging by capturing cross-model complementarity that fixed weights cannot express.

**QWK Threshold Optimization**: After stacking, we apply Nelder-Mead optimization on ordinal cumulative probability boundaries to maximize Quadratic Weighted Kappa beyond naive argmax — exploiting the ordinal structure of ESI levels that standard classification ignores.

## Results

The stacked ensemble achieves strong out-of-fold performance across all 80,000 training patients:

| Metric | LightGBM | XGBoost | CatBoost | Stacked Ensemble |
|:-------|:--------:|:-------:|:--------:|:----------------:|
| Accuracy | 99.57% | 99.44% | 99.42% | **99.59%** |
| Weighted F1 | 99.57% | 99.44% | 99.42% | **99.59%** |
| QWK | 0.9978 | 0.9973 | 0.9972 | **0.9980** |

**Calibration analysis** confirms well-calibrated probabilities (ECE < 0.05): a 70% confidence prediction corresponds to ~70% true positive rate — clinically essential for trustworthy decision support. ESI 1 and ESI 5 show the tightest calibration, exactly where certainty matters most.

**Top features**: NEWS2 score (r = −0.81 with acuity), GCS, SpO2, shock index, respiratory rate, and pain score. Triage nurse target encoding ranks highly, confirming measurable inter-rater variability across 50 nurses. NLP keyword flags for cardiac/neurological/trauma presentations also contribute meaningfully.

**Safety**: overall undertriage rate is 0.29% (235 patients), ESI 2 undertriage is 0.45% — well within clinical thresholds. Model confidence on incorrect predictions is lower than on correct ones, enabling "flag for senior review" on uncertain cases.

## Bias Analysis

We perform comprehensive demographic bias analysis on OOF predictions across five dimensions:

**Bias delta** (mean predicted − mean actual) across sex, age, language, insurance, and arrival mode shows no subgroup exceeding ±0.02 acuity levels. **Chi-squared testing** with Bonferroni correction evaluates statistical significance of accuracy differences. **Intersectional analysis** at sex × age × language intersections surfaces compound disadvantage — e.g., elderly non-native-speaking females represent a clinically vulnerable intersection.

**Undertriage monitoring**: ESI 1 undertriage is 4.87%, ESI 2 undertriage is 0.45%, overall 0.29%. **Overtriage** is clinically safer but still minimal.

## Clinical Misclassification Cost Analysis

Not all triage errors carry equal clinical consequence. We introduce an asymmetric cost matrix grounded in emergency medicine principles (Farrohknia et al., 2011): undertriage errors are weighted 3× heavier than overtriage (resource waste is preferable to patient harm), and penalties scale quadratically with the acuity gap (ESI 1→3 is far more dangerous than ESI 3→4). ESI 1–2 errors carry an additional severity multiplier reflecting their life-threatening nature.

This cost framework reveals that residual clinical cost concentrates in ESI 2→3 misclassifications — cases where an Emergent patient is downgraded to Urgent. This is actionable: deployment-time thresholding can be calibrated specifically for this boundary, or these cases can be routed to "senior review" protocols. The cost analysis also demonstrates that the model achieves a very high cost-efficiency score, meaning the small number of remaining errors are predominantly low-cost (adjacent-level overtriage rather than dangerous undertriage).

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
- Farrohknia, N. et al. (2011). *Emergency department triage scales and their components: a systematic review*. Scand. J. Trauma Resusc. Emerg. Med., 19, 42.
- Levin, S. et al. (2018). *Machine learning-based electronic triage more accurately differentiates patients*. Ann. Emerg. Med., 71(5), 565–574.
- Fernandes, M. et al. (2020). *Clinical Decision Support Systems for Triage in the Emergency Department*. Artif. Intell. Med., 102, 101762.
- Obermeyer, Z. et al. (2019). *Dissecting racial bias in an algorithm used to manage the health of populations*. Science, 366(6464), 447–453.
- Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist. Kaggle.
