"""
Model Evaluation Script
========================
Evaluates the severity classifier and duration regressor with:
- Train/Test split metrics
- 5-Fold Stratified Cross-Validation
- Confusion Matrix analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, 
    roc_auc_score, accuracy_score, classification_report,
    mean_absolute_error, median_absolute_error, r2_score,
)

from backend.forecasting import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TEXT_FEATURE, ALL_FEATURES,
    train_severity_classifier, load_full_data, load_and_prepare_data,
)

# Load processed data
df = load_full_data()  # Full dataset for severity (8K+ rows)

print("\n" + "=" * 70)
print("  COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)

# ─── Severity Classifier ─────────────────────────────────────────────────────

severity_features = [f for f in ALL_FEATURES if f != "corridor"]
X = df[severity_features].copy()
y = df["severity_tier"].copy()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

model_dict = train_severity_classifier(df.copy())
clf = model_dict["pipeline"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
if len(le.classes_) == 2:
    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
else:
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print("\n" + "=" * 70)
print("  SEVERITY CLASSIFIER — FINAL RESULTS")
print("=" * 70)
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {'':>12} Pred High  Pred Low")
for i, cls in enumerate(le.classes_):
    print(f"  True {cls:>5}  {cm[i][0]:>8}  {cm[i][1]:>8}")
print(f"  Classes: {list(le.classes_)}")

# ─── Duration Regressor ───────────────────────────────────────────────────────

df_dur = load_and_prepare_data()  # Only rows with valid duration

X_dur = df_dur[ALL_FEATURES].copy()
y_dur = df_dur["duration_to_close_min"].copy()
y_dur_log = np.log1p(y_dur)

X_dur_train, X_dur_test, y_dur_train, y_dur_test = train_test_split(
    X_dur, y_dur_log, test_size=0.2, random_state=42
)

from backend.forecasting import train_duration_regressor
dur_dict = train_duration_regressor(df_dur.copy())
reg = dur_dict["pipeline"]

y_dur_pred = np.clip(np.expm1(reg.predict(X_dur_test)), 1, None)
y_dur_test_orig = np.expm1(y_dur_test)

dur_mae = mean_absolute_error(y_dur_test_orig, y_dur_pred)
dur_medae = median_absolute_error(y_dur_test_orig, y_dur_pred)
dur_r2 = r2_score(y_dur_test_orig, y_dur_pred)

print("\n" + "=" * 70)
print("  DURATION REGRESSOR — FINAL RESULTS")
print("=" * 70)
print(f"  MAE:        {dur_mae:.1f} min")
print(f"  Median AE:  {dur_medae:.1f} min")
print(f"  R² Score:   {dur_r2:.4f}")

# ─── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  BEFORE vs AFTER COMPARISON")
print("=" * 70)
print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
print(f"  {'-'*55}")
print(f"  {'Severity Accuracy':<25} {'0.8022':>10} {accuracy_score(y_test, y_pred):>10.4f} {'✅' if accuracy_score(y_test, y_pred) > 0.8022 else '❌':>10}")
print(f"  {'Severity F1':<25} {'0.7956':>10} {f1:>10.4f} {'✅' if f1 > 0.7956 else '❌':>10}")
print(f"  {'Duration MAE (min)':<25} {'480.8':>10} {dur_mae:>10.1f} {'✅' if dur_mae < 480.8 else '❌':>10}")
print(f"  {'Duration R²':<25} {'0.0774':>10} {dur_r2:>10.4f} {'✅' if dur_r2 > 0.0774 else '❌':>10}")
print(f"  {'CV Accuracy Mean':<25} {'N/A':>10} {model_dict['metrics'].get('cv_accuracy_mean', 'N/A'):>10}")
print(f"  {'CV Accuracy Std':<25} {'N/A':>10} {model_dict['metrics'].get('cv_accuracy_std', 'N/A'):>10}")
print("=" * 70)
