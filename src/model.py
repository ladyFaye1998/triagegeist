"""Model training, cross-validation, and prediction for Triagegeist.

Uses LightGBM with stratified k-fold CV. Supports both multiclass
classification and ordinal regression framing.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix,
)
from .config import RANDOM_SEED, N_FOLDS, NUM_CLASSES


def get_lgbm_params(trial=None):
    """Default LightGBM hyperparameters tuned for 5-class triage acuity."""
    return {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }


def train_cv(X, y, n_folds=N_FOLDS, params=None):
    """
    Stratified k-fold cross-validation with LightGBM.
    Returns: oof_preds (n_samples, n_classes), models list, metrics dict.
    """
    if params is None:
        params = get_lgbm_params()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros((len(X), NUM_CLASSES))
    models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_probs = model.predict_proba(X_val)
        oof_preds[val_idx] = val_probs

        val_pred_labels = val_probs.argmax(axis=1) + 1  # 1-indexed
        acc = accuracy_score(y_val, val_pred_labels)
        f1 = f1_score(y_val, val_pred_labels, average="weighted")
        kappa = cohen_kappa_score(y_val, val_pred_labels, weights="quadratic")

        fold_metrics.append({"fold": fold_idx + 1, "accuracy": acc, "f1_weighted": f1, "qwk": kappa})
        models.append(model)
        print(f"  Fold {fold_idx+1}/{n_folds}: Acc={acc:.4f}, F1={f1:.4f}, QWK={kappa:.4f}")

    oof_labels = oof_preds.argmax(axis=1) + 1
    overall_metrics = {
        "accuracy": accuracy_score(y, oof_labels),
        "f1_weighted": f1_score(y, oof_labels, average="weighted"),
        "qwk": cohen_kappa_score(y, oof_labels, weights="quadratic"),
        "fold_metrics": fold_metrics,
    }
    print(f"\n  Overall OOF: Acc={overall_metrics['accuracy']:.4f}, "
          f"F1={overall_metrics['f1_weighted']:.4f}, QWK={overall_metrics['qwk']:.4f}")

    return oof_preds, models, overall_metrics


def predict_test(models, X_test):
    """Average predictions across fold models."""
    preds = np.zeros((len(X_test), NUM_CLASSES))
    for model in models:
        preds += model.predict_proba(X_test)
    preds /= len(models)
    return preds


def make_submission(test_ids, preds_proba, filepath="submission.csv"):
    """Create submission CSV from probability predictions."""
    pred_labels = preds_proba.argmax(axis=1) + 1
    sub = pd.DataFrame({
        "patient_id": test_ids,
        "triage_acuity": pred_labels,
    })
    sub.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath} — shape {sub.shape}")
    print(f"Prediction distribution:\n{pd.Series(pred_labels).value_counts().sort_index()}")
    return sub


def get_feature_importance(models, feature_names, top_n=30):
    """Average feature importance across fold models."""
    importances = np.zeros(len(feature_names))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_n)
    return fi
