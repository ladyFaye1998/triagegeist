"""Feature engineering for Triagegeist triage acuity prediction.

Clinically-motivated features derived from emergency medicine literature:
- Vital sign abnormality flags (based on normal ranges)
- Composite risk scores
- Chief complaint NLP features (TF-IDF)
- Interaction terms between vitals and demographics
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, HISTORY_FEATURES,
    TARGET_COL, ID_COL, LEAKAGE_COLS,
)


def add_vital_sign_flags(df):
    """Binary flags for clinically abnormal vital signs."""
    df = df.copy()
    df["flag_hypotension"] = (df["systolic_bp"] < 90).astype(int)
    df["flag_hypertension_crisis"] = (df["systolic_bp"] > 180).astype(int)
    df["flag_tachycardia"] = (df["heart_rate"] > 100).astype(int)
    df["flag_bradycardia"] = (df["heart_rate"] < 60).astype(int)
    df["flag_tachypnea"] = (df["respiratory_rate"] > 20).astype(int)
    df["flag_hypothermia"] = (df["temperature_c"] < 36.0).astype(int)
    df["flag_fever"] = (df["temperature_c"] > 38.0).astype(int)
    df["flag_hypoxia"] = (df["spo2"] < 92).astype(int)
    df["flag_altered_mental"] = (df["gcs_total"] < 15).astype(int)
    df["flag_severe_pain"] = (df["pain_score"] >= 8).astype(int)
    df["flag_high_shock_index"] = (df["shock_index"] > 1.0).astype(int)

    # Count of abnormal vitals — a composite acuity signal
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["num_abnormal_vitals"] = df[flag_cols].sum(axis=1)
    return df


def add_interaction_features(df):
    """Clinically meaningful feature interactions."""
    df = df.copy()

    # Age-vital interactions (elderly + abnormal vitals = higher acuity)
    df["age_x_shock_index"] = df["age"] * df["shock_index"]
    df["age_x_news2"] = df["age"] * df["news2_score"]
    df["age_x_gcs"] = df["age"] * df["gcs_total"]
    df["age_x_num_comorbidities"] = df["age"] * df["num_comorbidities"]

    # Vital sign ratios
    df["hr_rr_ratio"] = df["heart_rate"] / df["respiratory_rate"].replace(0, np.nan)
    df["sbp_hr_product"] = df["systolic_bp"] * df["heart_rate"]

    # ED utilization intensity
    df["ed_utilization_score"] = (
        df["num_prior_ed_visits_12m"] + 2 * df["num_prior_admissions_12m"]
    )

    # Comorbidity burden (history features sum)
    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    if hx_cols:
        df["comorbidity_burden"] = df[hx_cols].sum(axis=1)

    # Cardiovascular risk composite
    cv_cols = [
        "hx_hypertension", "hx_heart_failure", "hx_atrial_fibrillation",
        "hx_coronary_artery_disease", "hx_stroke_prior",
        "hx_peripheral_vascular_disease",
    ]
    cv_present = [c for c in cv_cols if c in df.columns]
    if cv_present:
        df["cv_risk_score"] = df[cv_present].sum(axis=1)

    return df


def add_time_features(df):
    """Time-based features from arrival information."""
    df = df.copy()
    df["is_weekend"] = df["arrival_day"].isin(["Saturday", "Sunday"]).astype(int)
    df["is_night"] = df["shift"].eq("night").astype(int)
    df["is_peak_hours"] = df["arrival_hour"].between(10, 18).astype(int)

    hour_rad = 2 * np.pi * df["arrival_hour"] / 24
    df["arrival_hour_sin"] = np.sin(hour_rad)
    df["arrival_hour_cos"] = np.cos(hour_rad)

    month_rad = 2 * np.pi * df["arrival_month"] / 12
    df["arrival_month_sin"] = np.sin(month_rad)
    df["arrival_month_cos"] = np.cos(month_rad)
    return df


class ChiefComplaintEncoder:
    """TF-IDF encoding for chief complaint raw text."""

    def __init__(self, max_features=100, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
        )

    def fit(self, texts):
        self.vectorizer.fit(texts.fillna("unknown"))
        return self

    def transform(self, texts):
        tfidf_matrix = self.vectorizer.transform(texts.fillna("unknown"))
        feature_names = [f"tfidf_{n}" for n in self.vectorizer.get_feature_names_out()]
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=texts.index,
        )

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


def encode_categoricals(df, target_encode_cols=None, target_col=TARGET_COL):
    """Label-encode categoricals. Optionally add target-encoded means."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def build_feature_matrix(train_df, test_df):
    """
    Full feature engineering pipeline.
    Returns X_train, y_train, X_test, feature_names, tfidf_encoder.
    """
    # Apply transforms to both
    for transform_fn in [add_vital_sign_flags, add_interaction_features, add_time_features]:
        train_df = transform_fn(train_df)
        test_df = transform_fn(test_df)

    # TF-IDF on chief complaint
    tfidf_encoder = ChiefComplaintEncoder(max_features=100)
    train_tfidf = tfidf_encoder.fit_transform(train_df["chief_complaint_raw"])
    test_tfidf = tfidf_encoder.transform(test_df["chief_complaint_raw"])

    # Encode categoricals
    train_df = encode_categoricals(train_df)
    test_df = encode_categoricals(test_df)

    # Assemble feature columns
    engineered_numeric = [c for c in train_df.columns if c.startswith("flag_") or c.startswith("is_")]
    engineered_numeric += [
        "num_abnormal_vitals", "age_x_shock_index", "age_x_news2",
        "age_x_gcs", "age_x_num_comorbidities", "hr_rr_ratio",
        "sbp_hr_product", "ed_utilization_score", "comorbidity_burden",
        "cv_risk_score", "is_weekend", "is_night", "is_peak_hours",
        "arrival_hour_sin", "arrival_hour_cos",
        "arrival_month_sin", "arrival_month_cos",
    ]
    # Deduplicate
    engineered_numeric = list(dict.fromkeys(engineered_numeric))

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + HISTORY_FEATURES + engineered_numeric
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = pd.concat([train_df[feature_cols].reset_index(drop=True), train_tfidf.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[feature_cols].reset_index(drop=True), test_tfidf.reset_index(drop=True)], axis=1)

    y_train = train_df[TARGET_COL].values

    return X_train, y_train, X_test, list(X_train.columns), tfidf_encoder
