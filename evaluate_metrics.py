import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, accuracy_score
import numpy as np

from backend.forecasting import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TEXT_FEATURE
from backend.forecasting import train_severity_classifier

df = pd.read_csv("data/processed/events_clean.csv", low_memory=False)

df = df.dropna(subset=["hour_of_day", "day_of_week", "severity_tier", "duration_to_close_min"]).copy()
df["requires_road_closure_int"] = df["requires_road_closure"].apply(lambda x: 1 if str(x).lower() == "true" or x == 1 else 0)
df["description"] = df["description"].fillna("")

X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TEXT_FEATURE]]
y = df["severity_tier"]

model_dict = train_severity_classifier(df.copy())
clf = model_dict["pipeline"]
le = model_dict["label_encoder"]

y_encoded = le.transform(y)
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

print("=== METRICS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classes:", le.classes_)
