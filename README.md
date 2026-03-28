# Triagegeist: AI-Powered Emergency Triage Acuity Prediction

**Multi-modal LightGBM with NLP Chief Complaint Analysis, Clinical Feature Engineering, and Demographic Bias Detection**

Submission for the [Triagegeist Kaggle Competition](https://www.kaggle.com/competitions/triagegeist/) by the Laitinen-Fredriksson Foundation.

---

## Overview

This project builds a clinical decision support system that predicts Emergency Severity Index (ESI) triage acuity levels (1–5) from structured patient intake data, free-text chief complaints, and patient medical history. The system achieves **~97% accuracy** and **0.987 quadratic weighted kappa** on 5-fold cross-validation.

### Key Features

- **Multi-table data fusion** — Combines vitals, demographics, NLP text, and 25 comorbidity flags
- **Clinical feature engineering** — Vital sign abnormality flags, composite risk scores, cardiovascular burden index
- **NLP pipeline** — TF-IDF on chief complaint free text captures high-risk presentation keywords
- **LightGBM with 5-fold stratified CV** — Gradient boosting handles mixed types and missing data natively
- **SHAP interpretability** — Feature-level explanations a clinician can audit
- **Demographic bias analysis** — Systematic over/under-triage detection across sex, age, language, insurance

## Project Structure

```
triagegeist/
├── data/                          # Competition data (not committed)
│   ├── train.csv
│   ├── test.csv
│   ├── chief_complaints.csv
│   ├── patient_history.csv
│   └── sample_submission.csv
├── notebooks/
│   └── triagegeist-triage-acuity-prediction.ipynb   # Main Kaggle notebook
├── src/
│   ├── config.py                  # Central configuration
│   ├── data_loader.py             # Data loading & merging pipeline
│   ├── feature_engineering.py     # Clinical feature engineering + NLP
│   ├── model.py                   # LightGBM training & prediction
│   └── analysis.py                # Bias analysis & interpretability
├── outputs/                       # Generated predictions & figures
├── assets/                        # Cover image and media
├── writeup.md                     # Competition writeup
├── requirements.txt               # Python dependencies
└── README.md
```

## Setup

```bash
git clone https://github.com/ladyFaye1998/triagegeist.git
cd triagegeist
pip install -r requirements.txt
```

### Run the Notebook

```bash
cd notebooks
jupyter notebook triagegeist-triage-acuity-prediction.ipynb
```

The notebook auto-detects Kaggle vs local paths. Place competition data in `data/` for local execution.

## Results

| Metric | Score |
|:-------|------:|
| Accuracy | ~97% |
| Weighted F1 | ~97% |
| Quadratic Weighted Kappa | ~0.987 |

### Top Predictive Features

1. NEWS2 Score — Aggregate early warning score
2. GCS Total — Glasgow Coma Scale
3. Shock Index — Heart rate / systolic BP
4. Pain Score — Patient-reported pain intensity
5. Heart Rate — Tachycardia/bradycardia signal

## Clinical Relevance

The model addresses real gaps in emergency triage:

- **Undertriage detection** — Flags patients assigned lower acuity who may need urgent care
- **Bias monitoring** — Surfaces systematic differences in triage across demographics
- **Decision support** — Provides a quantitative second opinion alongside clinical judgment

## Limitations

1. Trained on synthetic/competition data; requires validation on real clinical data (e.g., MIMIC-IV-ED)
2. TF-IDF NLP is basic; ClinicalBERT would capture richer semantics
3. No temporal validation (time-based splits would better simulate deployment)
4. Predicted probabilities are not calibrated for clinical decision thresholds

## License

MIT

## Author

[ladyFaye1998](https://www.kaggle.com/ladyfaye) — Kaggle profile
