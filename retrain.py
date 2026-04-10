import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import os
from datetime import datetime

print("\n" + "="*50)
print("⚙️ INITIATING AUTOMATED RETRAINING PIPELINE ⚙️")
print("="*50)

# 1. Simulate pulling new data (In reality, this would query a SQL database)
# We will generate a new dataset that reflects the "drifted" reality
print("[1/4] Fetching new data telemetry...")
np.random.seed(int(datetime.now().timestamp())) # Randomize so it's fresh
n_samples = 5000

# Notice the distributions here are slightly higher to represent the "new normal"
data = {
    'age': np.random.normal(70, 15, n_samples).astype(int), 
    'bmi': np.random.normal(30, 6, n_samples),
    'blood_pressure': np.random.normal(140, 20, n_samples),
    'previous_admissions': np.random.poisson(2.5, n_samples),
    'cholesterol': np.random.normal(220, 50, n_samples)
}
df = pd.DataFrame(data)

risk_score = (df['age'] * 0.02) + (df['bmi'] * 0.05) + (df['previous_admissions'] * 0.5) + np.random.normal(0, 1, n_samples)
df['readmitted'] = (risk_score > risk_score.median()).astype(int)

# 2. Train the New Model
print("[2/4] Training new Random Forest model...")
X = df.drop('readmitted', axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"      -> New Model Accuracy: {accuracy:.4f}")

# 3. Update Artifacts (Overwrite old model and baseline)
print("[3/4] Overwriting old artifacts with V2 Model...")
joblib.dump(model, 'artifacts/model.joblib')

baseline_stats = {}
for col in X_train.columns:
    baseline_stats[col] = {
        'mean': float(X_train[col].mean()),
        'std': float(X_train[col].std()),
        'min': float(X_train[col].min()),
        'max': float(X_train[col].max())
    }

with open('artifacts/baseline_stats.json', 'w') as f:
    json.dump(baseline_stats, f, indent=4)

# 4. Clean up old logs so the monitor starts fresh
print("[4/4] Archiving old inference logs...")
if os.path.exists("artifacts/inference_logs.csv"):
    os.rename("artifacts/inference_logs.csv", f"artifacts/logs_archive_{int(datetime.now().timestamp())}.csv")

print("="*50)
print("✅ RETRAINING COMPLETE. SYSTEM RESTORED TO HEALTH.")
print("="*50 + "\n")
