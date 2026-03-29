# Triagegeist: Multi-Modal Emergency Triage Acuity Prediction with Clinical Feature Engineering and Demographic Bias Detection

## Clinical Problem Statement

Every minute counts in the emergency department. Triage nurses assign patients to Emergency Severity Index (ESI) levels 1–5 that determine treatment priority — and errors in this assignment directly threaten patient safety. Inter-rater variability for ESI scoring is well-documented (kappa 0.60–0.80), with systematic undertriage of elderly patients and atypical presentations.

**Undertriage** — assigning a lower acuity than clinically warranted — is the critical safety concern. A patient incorrectly triaged as ESI 4 instead of ESI 2 may wait hours for care that should begin immediately. This project builds an AI-powered decision support model that predicts ESI triage acuity and identifies systematic bias patterns across demographic groups.

## Methodology

### Data Fusion
We integrate three competition data sources on `patient_id`:
- **Structured intake** (80,000 train / 20,000 test): vitals, demographics, arrival context, ED utilization
- **Chief complaints** (100,000 records): free-text clinical presentations and body system categories
- **Patient history** (100,000 records): 25 binary comorbidity flags (hypertension, diabetes, COPD, etc.)

### Clinical Feature Engineering (50+ features)
Features are grounded in emergency medicine literature:

- **Vital sign abnormality flags** (18 flags): Hypotension (SBP<90), severe tachycardia (HR>130), hypoxia (SpO2<92), severe GCS (≤8), etc. These map directly to ESI decision branch points (Gilboy et al., 2020).
- **qSOFA score** (0–3): SBP≤100, RR≥22, GCS<15 — sepsis screening per Sepsis-3 guidelines (Singer et al., 2016).
- **SIRS approximation**: Temperature/HR/RR abnormality count for systemic inflammatory response detection.
- **Composite risk scores**: Cardiovascular burden (6 conditions), respiratory risk (asthma+COPD), metabolic risk, psychiatric risk.
- **Age-vital interactions**: `age × shock_index`, `age × NEWS2`, `age × qSOFA` — capturing the clinical principle that elderly patients decompensate faster for equivalent vital sign derangements.
- **Critical NLP keyword flags** (16 patterns): Regex-based detection of high-risk presentations (chest pain, seizure, stroke, suicidal ideation, respiratory distress, GI bleed, etc.) with clinical urgency mapping.
- **TF-IDF** (150 features): Sublinear TF-IDF with unigram+bigram encoding of chief complaint free text.
- **Target-encoded IDs**: Out-of-fold target encoding for nurse ID and site ID, capturing systematic inter-rater variability without leakage.
- **Temporal features**: Cyclical encoding of arrival hour/month, weekend and night-shift flags.

### Model Architecture
- **LightGBM**: 3,000 estimators, learning rate 0.03, 127 leaves, early stopping at 150 rounds
- **XGBoost**: 2,000 estimators, learning rate 0.03, max depth 8, early stopping
- **Ensemble**: Weighted average with weights optimized via grid search on OOF predictions
- **Validation**: 5-fold stratified cross-validation, evaluated on accuracy, weighted F1, and quadratic weighted kappa (QWK)

## Results

The ensemble model achieves strong out-of-fold performance across all 80,000 training patients:

| Metric | Score |
|:-------|------:|
| Accuracy | ~97% |
| Weighted F1 | ~97% |
| Quadratic Weighted Kappa | ~0.98 |

**Top predictive features**: NEWS2 score (r = −0.81 with acuity), GCS total, shock index, pain score, heart rate, triage nurse ID target encoding. NEWS2 being the strongest predictor validates the National Early Warning Score as an effective triage tool.

**Chief complaint NLP** adds meaningful signal: terms related to cardiac symptoms, neurological presentations, and trauma rank highly in feature importance. The 16 critical keyword flags capture domain knowledge that pure statistical features miss.

**Triage nurse variability**: Mean assigned acuity ranges substantially across the 50 nurses in the dataset, confirming the documented inter-rater variability that motivates clinical decision support.

## Bias Analysis

We perform comprehensive demographic bias analysis with statistical significance testing:

- **Bias delta** (mean predicted − mean actual acuity) quantifies systematic over/under-triage per group
- **Chi-squared tests** for accuracy differences across sex, age group, language, and insurance type
- **Intersectional analysis**: Identifying highest-risk subgroups at the intersection of sex × age × language
- **Undertriage monitoring by actual acuity**: Tracking the rate at which truly urgent (ESI 1–2) patients are predicted as less urgent — the most dangerous error mode

## Limitations

1. **Synthetic data**: Performance on competition data may not transfer to real clinical environments. Validation against MIMIC-IV-ED is essential before clinical use.
2. **NEWS2 as feature**: NEWS2 is itself a clinical scoring system and may partially encode existing triage decisions, inflating apparent model performance.
3. **NLP depth**: TF-IDF captures keyword signal but misses semantic nuance; ClinicalBERT would improve chief complaint understanding.
4. **No temporal validation**: Random splits rather than time-based splits; prospective deployment simulation would require chronological holdout.
5. **No probability calibration**: Platt scaling or isotonic regression needed for clinical decision thresholds.
6. **No external validation**: Unknown generalization to other institutions or healthcare systems.

## Reproducibility

- **Kaggle Notebook**: Public, runs end-to-end in ~5 minutes
- **GitHub Repository**: [github.com/ladyFaye1998/triagegeist](https://github.com/ladyFaye1998/triagegeist) — full source code, modular architecture, README with setup instructions
- **Interactive Demo**: [GitHub Pages](https://ladyfaye1998.github.io/triagegeist/) — browser-based triage prediction simulator
- **Random seed 42** fixed for all stochastic operations
- **All datasets cited**: Competition-provided data from the Laitinen-Fredriksson Foundation

## References

- Gilboy, N. et al. (2020). *Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4*. AHRQ Publication No. 20-0045.
- Singer, M. et al. (2016). *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)*. JAMA, 315(8), 801–810.
- Royal College of Physicians (2017). *National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS*.
- Fernandes, M. et al. (2020). *Clinical Decision Support Systems for Triage in the Emergency Department using Intelligent Systems: A Review*. Artificial Intelligence in Medicine, 102, 101762.
- Obermeyer, Z. et al. (2019). *Dissecting racial bias in an algorithm used to manage the health of populations*. Science, 366(6464), 447–453.
