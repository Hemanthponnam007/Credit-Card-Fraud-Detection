import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if "Class" not in df.columns:
    raise ValueError("CSV must contain a 'Class' column.")

df = df.drop_duplicates().dropna()
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# Keep numeric features for this tabular ML project.
X = X.select_dtypes(include=np.number)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
    })

# Isolation Forest is evaluated separately as an anomaly detector.
iso = IsolationForest(
    n_estimators=150, contamination=float(y.mean()),
    random_state=42
)
iso.fit(X_train_scaled)
iso_pred = iso.predict(X_test_scaled)
iso_pred = np.where(iso_pred == -1, 1, 0)

results.append({
    "Model": "Isolation Forest",
    "Accuracy": accuracy_score(y_test, iso_pred),
    "Precision": precision_score(y_test, iso_pred, zero_division=0),
    "Recall": recall_score(y_test, iso_pred, zero_division=0),
    "F1 Score": f1_score(y_test, iso_pred, zero_division=0),
})

results_df = pd.DataFrame(results)
results_df.to_csv("models/model_results.csv", index=False)

# Select the best supervised model by F1 score.
best_name = results_df[results_df["Model"].isin(models.keys())].sort_values(
    "F1 Score", ascending=False
).iloc[0]["Model"]

best_model = models[best_name]
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/features.pkl")

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nBest supervised model: {best_name}")

pred = best_model.predict(X_test_scaled)
print("\nClassification report for best model:")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
