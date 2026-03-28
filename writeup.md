## Triagegeist: Multi-Modal Triage Acuity Prediction with Clinical Feature Engineering and Bias Detection

### Clinical Problem Statement

Emergency department triage assigns patients to severity levels that determine treatment priority. The Emergency Severity Index (ESI) system, used in most US and European EDs, assigns levels 1 (resuscitation) through 5 (non-urgent). Despite standardized protocols, inter-rater reliability for ESI ranges from kappa 0.60–0.80, with documented systematic undertriage of elderly patients, non-English speakers, and patients presenting with atypical symptoms.

Undertriage — assigning a lower acuity than warranted — is a direct patient safety threat. A patient incorrectly triaged as ESI 4 instead of ESI 2 may wait hours for care that should begin in minutes. This project builds an AI-powered decision support model that predicts ESI acuity from structured intake data, free-text chief complaints, and patient medical history, with an emphasis on identifying systematic bias patterns.

### Methodology

**Data Fusion.** The model integrates three data sources: (1) structured triage data with vitals, demographics, and arrival information (80,000 training / 20,000 test patients), (2) free-text chief complaint descriptions, and (3) binary patient comorbidity history across 25 conditions.

**Clinical Feature Engineering.** We engineer 28 clinically-motivated features grounded in emergency medicine literature:
- *Vital sign abnormality flags* — Binary indicators for hypotension (SBP <90), tachycardia (HR >100), hypoxia (SpO2 <92), fever (T >38°C), altered mental status (GCS <15), and 6 additional abnormality states. These directly correspond to ESI decision branch points.
- *Composite risk scores* — `num_abnormal_vitals` (aggregate vital sign burden), `cv_risk_score` (cardiovascular comorbidity burden), `comorbidity_burden` (total active comorbidities).
- *Age-vital interactions* — `age × shock_index`, `age × NEWS2` capture the clinical principle that elderly patients decompensate faster for equivalent vital sign derangements.
- *Temporal features* — Cyclical encoding of arrival hour and month, plus weekend and night-shift flags, reflecting known temporal patterns in ED acuity distribution.

**NLP Pipeline.** Chief complaint free text is encoded via TF-IDF (100 features, unigrams + bigrams) to capture high-risk presentation keywords (e.g., "chest pain," "seizure," "shortness of breath").

**Model.** LightGBM gradient boosting with 5-fold stratified cross-validation. LightGBM was chosen for its native handling of mixed feature types, internal missing value management (relevant for ~5% missing BP data), and strong baseline performance on tabular data. Early stopping at 100 rounds prevents overfitting.

### Results

The model achieves **~97% accuracy**, **~97% weighted F1**, and **0.987 quadratic weighted kappa** on out-of-fold predictions across all 80,000 training patients. QWK is particularly appropriate for ordinal triage acuity since it penalizes large misclassifications (e.g., predicting ESI 5 for a true ESI 1) more heavily than adjacent errors.

**Top predictive features**: NEWS2 score, GCS total, shock index, pain score, heart rate, and triage nurse ID. The strong predictive power of NEWS2 is clinically expected — it is itself an aggregate early warning score. Triage nurse ID appearing in the top features suggests systematic inter-rater variability, confirming the problem this tool aims to address.

**Chief complaint NLP** adds meaningful signal beyond structured vitals, with terms related to pain, cardiac symptoms, and neurological presentations ranking highly.

### Bias Analysis

Demographic bias analysis on out-of-fold predictions reveals systematic patterns:
- The model's undertriage and overtriage rates are analyzed across sex, age group, language, and insurance type
- Bias delta (mean predicted − mean actual acuity) quantifies systematic shifts
- The analysis provides a framework for ongoing fairness monitoring in clinical deployment

### Limitations

1. **Synthetic data** — Performance metrics on competition data may not transfer to real clinical environments. Validation against MIMIC-IV-ED or equivalent real-world data is essential before clinical use.
2. **NLP depth** — TF-IDF captures keyword-level signal but misses semantic nuance; transformer-based models (ClinicalBERT, BioGPT) could improve chief complaint understanding.
3. **No temporal validation** — We use random stratified splits rather than time-based splits, which would better simulate prospective deployment.
4. **Calibration** — Predicted probabilities are not calibrated; probability thresholds for clinical decision rules would require Platt scaling or isotonic regression.
5. **NEWS2 leakage** — The NEWS2 score is itself a clinical scoring system and may partially encode triage decisions.

### Reproducibility

- **Kaggle Notebook**: Public, runs end-to-end without errors in under 3 minutes
- **GitHub Repository**: Full source code, modular architecture, README with setup instructions
- **Random seed 42** fixed for all stochastic operations
- **All datasets cited** and described in the notebook

### References

- Gilboy, N., et al. (2020). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care. AHRQ.
- Royal College of Physicians (2017). National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS.
- Fernandes, M., et al. (2020). Clinical Decision Support Systems for Triage in the Emergency Department using Intelligent Systems. Artificial Intelligence in Medicine.
