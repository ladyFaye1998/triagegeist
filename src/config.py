"""Central configuration for the Triagegeist pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
CHIEF_COMPLAINTS_FILE = DATA_DIR / "chief_complaints.csv"
PATIENT_HISTORY_FILE = DATA_DIR / "patient_history.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

TARGET_COL = "triage_acuity"
ID_COL = "patient_id"

# ESI acuity labels (1 = most urgent, 5 = least urgent)
ACUITY_LABELS = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}
NUM_CLASSES = 5

# Columns present in train but not test (leakage targets)
LEAKAGE_COLS = ["disposition", "ed_los_hours", "triage_acuity"]

# Categorical features for encoding
CATEGORICAL_FEATURES = [
    "site_id", "triage_nurse_id", "arrival_mode", "arrival_day",
    "arrival_season", "shift", "age_group", "sex", "language",
    "insurance_type", "transport_origin", "pain_location",
    "mental_status_triage", "chief_complaint_system",
]

# Numeric features from main table
NUMERIC_FEATURES = [
    "arrival_hour", "arrival_month", "age",
    "num_prior_ed_visits_12m", "num_prior_admissions_12m",
    "num_active_medications", "num_comorbidities",
    "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
    "pulse_pressure", "heart_rate", "respiratory_rate",
    "temperature_c", "spo2", "gcs_total", "pain_score",
    "weight_kg", "height_cm", "bmi", "shock_index", "news2_score",
]

# Patient history binary columns (joined from patient_history.csv)
HISTORY_FEATURES = [
    "hx_hypertension", "hx_diabetes_type2", "hx_diabetes_type1",
    "hx_asthma", "hx_copd", "hx_heart_failure", "hx_atrial_fibrillation",
    "hx_ckd", "hx_liver_disease", "hx_malignancy", "hx_obesity",
    "hx_depression", "hx_anxiety", "hx_dementia", "hx_epilepsy",
    "hx_hypothyroidism", "hx_hyperthyroidism", "hx_hiv",
    "hx_coagulopathy", "hx_immunosuppressed", "hx_pregnant",
    "hx_substance_use_disorder", "hx_coronary_artery_disease",
    "hx_stroke_prior", "hx_peripheral_vascular_disease",
]

RANDOM_SEED = 42
N_FOLDS = 5
