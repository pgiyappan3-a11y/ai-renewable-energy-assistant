"""
Capstone 11: AI-Based Renewable Energy Source Selection Assistant
Dataset is SIMULATED for educational demonstration, not measured field data.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "simulated_renewable_energy.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = [
    "solar_irradiance_wm2", "wind_speed_ms", "temperature_c",
    "humidity_pct", "cloud_cover_pct", "hour", "month"
]
target = "recommended_source"
X, y = df[features], df[target]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocess = ColumnTransformer([("numeric", numeric_pipe, features)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, class_weight="balanced")
}
results = []
best_name, best_model, best_f1 = None, None, -1
for name, classifier in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", classifier)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    score = f1_score(y_test, pred, average="weighted")
    results.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "weighted_f1": round(score, 4)
    })
    if score > best_f1:
        best_name, best_model, best_f1 = name, pipe, score

# Baseline comparator helps contextualize model performance
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
results.append({
    "model": "Most-frequent baseline",
    "accuracy": round(accuracy_score(y_test, baseline_pred), 4),
    "weighted_f1": round(f1_score(y_test, baseline_pred, average="weighted"), 4)
})

joblib.dump(best_model, OUT / "best_model.joblib")
pd.DataFrame(results).to_csv(OUT / "model_comparison.csv", index=False)
pred = best_model.predict(X_test)
with open(OUT / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Selected model: {best_name}\\n")
    f.write(classification_report(y_test, pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=30)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=160)
plt.close()

print("Training complete.")
print(pd.DataFrame(results).to_string(index=False))
print("Selected:", best_name)
print("Outputs saved in:", OUT)
