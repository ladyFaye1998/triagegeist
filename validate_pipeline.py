"""Full end-to-end pipeline validation for Triagegeist."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
SEED, N_FOLDS, N_CLASSES = 42, 5, 5
DATA = 'data'

# Load & merge
train_raw = pd.read_csv(f'{DATA}/train.csv')
test_raw  = pd.read_csv(f'{DATA}/test.csv')
complaints = pd.read_csv(f'{DATA}/chief_complaints.csv')
history    = pd.read_csv(f'{DATA}/patient_history.csv')
cc_dedup = complaints[['patient_id','chief_complaint_raw']].drop_duplicates(subset='patient_id', keep='first')
train = train_raw.merge(cc_dedup, on='patient_id', how='left').merge(history, on='patient_id', how='left')
test  = test_raw.merge(cc_dedup, on='patient_id', how='left').merge(history, on='patient_id', how='left')
print(f'Merged: train={train.shape}, test={test.shape}')

def eng(df):
    df = df.copy()
    df['flag_hypotension'] = (df['systolic_bp'] < 90).astype(np.int8)
    df['flag_hypertension_crisis'] = (df['systolic_bp'] > 180).astype(np.int8)
    df['flag_severe_hypotension'] = (df['systolic_bp'] < 70).astype(np.int8)
    df['flag_tachycardia'] = (df['heart_rate'] > 100).astype(np.int8)
    df['flag_severe_tachycardia'] = (df['heart_rate'] > 130).astype(np.int8)
    df['flag_bradycardia'] = (df['heart_rate'] < 60).astype(np.int8)
    df['flag_tachypnea'] = (df['respiratory_rate'] > 20).astype(np.int8)
    df['flag_severe_tachypnea'] = (df['respiratory_rate'] > 30).astype(np.int8)
    df['flag_hypothermia'] = (df['temperature_c'] < 36.0).astype(np.int8)
    df['flag_fever'] = (df['temperature_c'] > 38.0).astype(np.int8)
    df['flag_high_fever'] = (df['temperature_c'] > 39.0).astype(np.int8)
    df['flag_hypoxia'] = (df['spo2'] < 92).astype(np.int8)
    df['flag_severe_hypoxia'] = (df['spo2'] < 88).astype(np.int8)
    df['flag_altered_mental'] = (df['gcs_total'] < 15).astype(np.int8)
    df['flag_severe_gcs'] = (df['gcs_total'] <= 8).astype(np.int8)
    df['flag_severe_pain'] = (df['pain_score'] >= 8).astype(np.int8)
    df['flag_high_shock_idx'] = (df['shock_index'] > 1.0).astype(np.int8)
    df['flag_critical_shock_idx'] = (df['shock_index'] > 1.3).astype(np.int8)
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    df['num_abnormal_vitals'] = df[flag_cols].sum(axis=1)
    df['qsofa_sbp'] = (df['systolic_bp'] <= 100).astype(np.int8)
    df['qsofa_rr'] = (df['respiratory_rate'] >= 22).astype(np.int8)
    df['qsofa_gcs'] = (df['gcs_total'] < 15).astype(np.int8)
    df['qsofa_score'] = df['qsofa_sbp'] + df['qsofa_rr'] + df['qsofa_gcs']
    df['sirs_temp'] = ((df['temperature_c'] > 38.0) | (df['temperature_c'] < 36.0)).astype(np.int8)
    df['sirs_hr'] = (df['heart_rate'] > 90).astype(np.int8)
    df['sirs_rr'] = (df['respiratory_rate'] > 20).astype(np.int8)
    df['sirs_count'] = df['sirs_temp'] + df['sirs_hr'] + df['sirs_rr']
    hx = [c for c in df.columns if c.startswith('hx_')]
    if hx:
        df['comorbidity_burden'] = df[hx].sum(axis=1)
    cv = ['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation',
          'hx_coronary_artery_disease','hx_stroke_prior','hx_peripheral_vascular_disease']
    df['cv_risk_score'] = df[[c for c in cv if c in df.columns]].sum(axis=1)
    df['respiratory_risk'] = df[[c for c in ['hx_asthma','hx_copd'] if c in df.columns]].sum(axis=1)
    df['metabolic_risk'] = df[[c for c in ['hx_diabetes_type2','hx_diabetes_type1','hx_obesity'] if c in df.columns]].sum(axis=1)
    df['psych_risk'] = df[[c for c in ['hx_depression','hx_anxiety','hx_substance_use_disorder'] if c in df.columns]].sum(axis=1)
    df['age_x_shock_index'] = df['age'] * df['shock_index']
    df['age_x_news2'] = df['age'] * df['news2_score']
    df['age_x_gcs'] = df['age'] * df['gcs_total']
    df['age_x_comorbidities'] = df['age'] * df['num_comorbidities']
    df['age_x_qsofa'] = df['age'] * df['qsofa_score']
    df['hr_rr_ratio'] = df['heart_rate'] / df['respiratory_rate'].replace(0, np.nan)
    df['sbp_hr_product'] = df['systolic_bp'] * df['heart_rate']
    df['map_hr_ratio'] = df['mean_arterial_pressure'] / df['heart_rate'].replace(0, np.nan)
    df['pp_sbp_ratio'] = df['pulse_pressure'] / df['systolic_bp'].replace(0, np.nan)
    df['spo2_rr_product'] = df['spo2'] * df['respiratory_rate']
    df['ed_utilization'] = df['num_prior_ed_visits_12m'] + 2 * df['num_prior_admissions_12m']
    df['high_ed_user'] = (df['num_prior_ed_visits_12m'] >= 4).astype(np.int8)
    df['polypharmacy'] = (df['num_active_medications'] >= 5).astype(np.int8)
    df['is_weekend'] = df['arrival_day'].isin(['Saturday','Sunday']).astype(np.int8)
    df['is_night'] = df['shift'].eq('night').astype(np.int8)
    df['is_peak_hours'] = df['arrival_hour'].between(10,18).astype(np.int8)
    hr_rad = 2 * np.pi * df['arrival_hour'] / 24
    df['hour_sin'] = np.sin(hr_rad)
    df['hour_cos'] = np.cos(hr_rad)
    mr_rad = 2 * np.pi * df['arrival_month'] / 12
    df['month_sin'] = np.sin(mr_rad)
    df['month_cos'] = np.cos(mr_rad)
    return df

train = eng(train)
test = eng(test)

KW = {
    'kw_chest_pain': r'chest\s*pain|angina',
    'kw_sob': r'shortness.*breath|dyspn',
    'kw_stroke': r'stroke|hemipar|aphasia',
    'kw_seizure': r'seizure|convuls',
    'kw_cardiac_arrest': r'cardiac\s*arrest|asystole|cpr',
    'kw_trauma_major': r'major\s*trauma|polytrauma|mva',
    'kw_sepsis': r'sepsis|septic',
    'kw_anaphylaxis': r'anaphyla',
    'kw_suicidal': r'suicid|self.?harm|overdose',
    'kw_altered_mental': r'altered\s*mental|confusion|unresponsive|syncope',
    'kw_gi_bleed': r'haematemes|melena|gi\s*bleed',
    'kw_fracture': r'fracture|broken',
    'kw_abdominal': r'abdominal\s*pain|appendic',
    'kw_headache': r'headache|migraine|thunderclap',
    'kw_fever': r'fever|febrile|pyrexia|rigors',
    'kw_mild': r'advice|follow.?up|prescription|refill|minor|mild',
}
for k, v in KW.items():
    train[k] = train['chief_complaint_raw'].fillna('').str.contains(v, case=False, regex=True).astype(np.int8)
    test[k] = test['chief_complaint_raw'].fillna('').str.contains(v, case=False, regex=True).astype(np.int8)

kw_cols = [c for c in train.columns if c.startswith('kw_')]
train['num_critical_keywords'] = train[kw_cols].sum(axis=1)
test['num_critical_keywords'] = test[kw_cols].sum(axis=1)
train['cc_length'] = train['chief_complaint_raw'].fillna('').str.len()
train['cc_word_count'] = train['chief_complaint_raw'].fillna('').str.split().str.len()
test['cc_length'] = test['chief_complaint_raw'].fillna('').str.len()
test['cc_word_count'] = test['chief_complaint_raw'].fillna('').str.split().str.len()

# TF-IDF
tfidf = TfidfVectorizer(max_features=150, ngram_range=(1,2), stop_words='english',
                         min_df=10, max_df=0.95, sublinear_tf=True)
tr_tf = pd.DataFrame(
    tfidf.fit_transform(train['chief_complaint_raw'].fillna('unknown')).toarray(),
    columns=[f'tfidf_{n}' for n in tfidf.get_feature_names_out()])
te_tf = pd.DataFrame(
    tfidf.transform(test['chief_complaint_raw'].fillna('unknown')).toarray(),
    columns=[f'tfidf_{n}' for n in tfidf.get_feature_names_out()])

# Target encoding OOF
def te_oof(tr_df, te_df, col, target, nf=5, seed=42):
    gm = tr_df[target].mean()
    tr_enc = pd.Series(np.full(len(tr_df), gm), index=tr_df.index)
    skf2 = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
    for ti, vi in skf2.split(tr_df, tr_df[target]):
        means = tr_df.iloc[ti].groupby(col)[target].mean()
        tr_enc.iloc[vi] = tr_df.iloc[vi][col].map(means).fillna(gm).values
    fm = tr_df.groupby(col)[target].mean()
    te_enc = te_df[col].map(fm).fillna(gm)
    return tr_enc, te_enc

for col in ['triage_nurse_id', 'site_id']:
    tre, tee = te_oof(train, test, col, 'triage_acuity')
    train[f'{col}_te'] = tre.values
    test[f'{col}_te'] = tee.values

CAT = ['arrival_mode','arrival_day','arrival_season','shift','age_group','sex','language',
       'insurance_type','transport_origin','pain_location','mental_status_triage','chief_complaint_system']
for col in CAT:
    combined = pd.concat([train[col], test[col]]).astype('category')
    codes = combined.cat.codes
    train[col] = codes.iloc[:len(train)].values
    test[col] = codes.iloc[len(train):].values

NUM = ['arrival_hour','arrival_month','age','num_prior_ed_visits_12m','num_prior_admissions_12m',
       'num_active_medications','num_comorbidities','systolic_bp','diastolic_bp','mean_arterial_pressure',
       'pulse_pressure','heart_rate','respiratory_rate','temperature_c','spo2','gcs_total','pain_score',
       'weight_kg','height_cm','bmi','shock_index','news2_score']
HX = [c for c in train.columns if c.startswith('hx_')]
ENG = ([c for c in train.columns if c.startswith('flag_')]
       + ['num_abnormal_vitals','qsofa_score','sirs_count','comorbidity_burden','cv_risk_score',
          'respiratory_risk','metabolic_risk','psych_risk','age_x_shock_index','age_x_news2',
          'age_x_gcs','age_x_comorbidities','age_x_qsofa','hr_rr_ratio','sbp_hr_product',
          'map_hr_ratio','pp_sbp_ratio','spo2_rr_product','ed_utilization','high_ed_user',
          'polypharmacy','is_weekend','is_night','is_peak_hours','hour_sin','hour_cos',
          'month_sin','month_cos','triage_nurse_id_te','site_id_te',
          'num_critical_keywords','cc_length','cc_word_count'] + kw_cols)
ENG = list(dict.fromkeys(ENG))
fcols = [c for c in CAT + NUM + HX + ENG if c in train.columns]

X_train = pd.concat([train[fcols].reset_index(drop=True), tr_tf.reset_index(drop=True)], axis=1)
X_test = pd.concat([test[fcols].reset_index(drop=True), te_tf.reset_index(drop=True)], axis=1)
y_train = train['triage_acuity'].values
print(f'Features: {X_train.shape[1]}')

# LightGBM
lgbm_params = {
    'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss',
    'boosting_type': 'gbdt', 'n_estimators': 3000, 'learning_rate': 0.03,
    'num_leaves': 127, 'max_depth': -1, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_alpha': 0.05,
    'reg_lambda': 1.0, 'random_state': 42, 'verbose': -1, 'n_jobs': -1,
}
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgbm = np.zeros((len(X_train), N_CLASSES))
test_lgbm = np.zeros((len(X_test), N_CLASSES))

print('\nLightGBM 5-fold CV:')
for fold, (ti, vi) in enumerate(skf.split(X_train, y_train)):
    m = lgb.LGBMClassifier(**lgbm_params)
    m.fit(X_train.iloc[ti], y_train[ti],
          eval_set=[(X_train.iloc[vi], y_train[vi])],
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    oof_lgbm[vi] = m.predict_proba(X_train.iloc[vi])
    test_lgbm += m.predict_proba(X_test) / N_FOLDS
    vp = oof_lgbm[vi].argmax(axis=1) + 1
    acc = accuracy_score(y_train[vi], vp)
    qwk = cohen_kappa_score(y_train[vi], vp, weights='quadratic')
    print(f'  Fold {fold+1}: Acc={acc:.4f}  QWK={qwk:.4f}  iter={m.best_iteration_}')

lgbm_labels = oof_lgbm.argmax(axis=1) + 1
lgbm_qwk = cohen_kappa_score(y_train, lgbm_labels, weights='quadratic')
lgbm_acc = accuracy_score(y_train, lgbm_labels)
print(f'LightGBM OOF: Acc={lgbm_acc:.4f}  QWK={lgbm_qwk:.4f}')

# XGBoost
xgb_params = {
    'objective': 'multi:softprob', 'num_class': 5, 'eval_metric': 'mlogloss',
    'tree_method': 'hist', 'n_estimators': 2000, 'learning_rate': 0.03,
    'max_depth': 8, 'min_child_weight': 5, 'subsample': 0.8,
    'colsample_bytree': 0.6, 'reg_alpha': 0.05, 'reg_lambda': 1.0,
    'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
}
oof_xgb = np.zeros((len(X_train), N_CLASSES))
test_xgb = np.zeros((len(X_test), N_CLASSES))

print('\nXGBoost 5-fold CV:')
for fold, (ti, vi) in enumerate(skf.split(X_train, y_train)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X_train.iloc[ti], y_train[ti] - 1,
          eval_set=[(X_train.iloc[vi], y_train[vi] - 1)], verbose=False)
    oof_xgb[vi] = m.predict_proba(X_train.iloc[vi])
    test_xgb += m.predict_proba(X_test) / N_FOLDS
    vp = oof_xgb[vi].argmax(axis=1) + 1
    acc = accuracy_score(y_train[vi], vp)
    qwk = cohen_kappa_score(y_train[vi], vp, weights='quadratic')
    print(f'  Fold {fold+1}: Acc={acc:.4f}  QWK={qwk:.4f}')

xgb_labels = oof_xgb.argmax(axis=1) + 1
xgb_qwk = cohen_kappa_score(y_train, xgb_labels, weights='quadratic')
xgb_acc = accuracy_score(y_train, xgb_labels)
print(f'XGBoost OOF: Acc={xgb_acc:.4f}  QWK={xgb_qwk:.4f}')

# Ensemble
best_qwk, best_w = -1, 0.5
for w in np.arange(0.3, 0.8, 0.05):
    blend = w * oof_lgbm + (1 - w) * oof_xgb
    bl = blend.argmax(axis=1) + 1
    q = cohen_kappa_score(y_train, bl, weights='quadratic')
    if q > best_qwk:
        best_qwk, best_w = q, w

oof_ens = best_w * oof_lgbm + (1 - best_w) * oof_xgb
ens_labels = oof_ens.argmax(axis=1) + 1
ens_acc = accuracy_score(y_train, ens_labels)
ens_f1 = f1_score(y_train, ens_labels, average='weighted')
print(f'\nENSEMBLE (w_lgbm={best_w:.2f}):')
print(f'  Accuracy:  {ens_acc:.4f}')
print(f'  F1:        {ens_f1:.4f}')
print(f'  QWK:       {best_qwk:.4f}')

test_ens = best_w * test_lgbm + (1 - best_w) * test_xgb
test_labels = test_ens.argmax(axis=1) + 1
sub = pd.DataFrame({'patient_id': test_raw['patient_id'], 'triage_acuity': test_labels})
sub.to_csv('outputs/submission.csv', index=False)
print(f'\nSubmission saved: {sub.shape}')
print(sub['triage_acuity'].value_counts().sort_index())
print('\n=== PIPELINE VALIDATED SUCCESSFULLY ===')
