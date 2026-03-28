"""Data loading and merging pipeline for Triagegeist."""
import pandas as pd
from .config import (
    TRAIN_FILE, TEST_FILE, CHIEF_COMPLAINTS_FILE,
    PATIENT_HISTORY_FILE, SAMPLE_SUBMISSION_FILE, ID_COL,
)


def load_raw_data(data_dir=None):
    """Load all raw CSV files and return a dict of DataFrames."""
    if data_dir is not None:
        from pathlib import Path
        d = Path(data_dir)
        train = pd.read_csv(d / "train.csv")
        test = pd.read_csv(d / "test.csv")
        complaints = pd.read_csv(d / "chief_complaints.csv")
        history = pd.read_csv(d / "patient_history.csv")
        sample_sub = pd.read_csv(d / "sample_submission.csv")
    else:
        train = pd.read_csv(TRAIN_FILE)
        test = pd.read_csv(TEST_FILE)
        complaints = pd.read_csv(CHIEF_COMPLAINTS_FILE)
        history = pd.read_csv(PATIENT_HISTORY_FILE)
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_FILE)

    return {
        "train": train,
        "test": test,
        "complaints": complaints,
        "history": history,
        "sample_submission": sample_sub,
    }


def merge_auxiliary_tables(df, complaints, history):
    """Left-join chief complaints raw text and patient history onto main table."""
    complaints_for_join = complaints[[ID_COL, "chief_complaint_raw"]].drop_duplicates(
        subset=ID_COL, keep="first"
    )
    df = df.merge(complaints_for_join, on=ID_COL, how="left")
    df = df.merge(history, on=ID_COL, how="left")
    return df


def build_datasets(data_dir=None):
    """Full pipeline: load raw → merge → return train_df, test_df, sample_sub."""
    raw = load_raw_data(data_dir)
    train_df = merge_auxiliary_tables(raw["train"], raw["complaints"], raw["history"])
    test_df = merge_auxiliary_tables(raw["test"], raw["complaints"], raw["history"])
    return train_df, test_df, raw["sample_submission"]
