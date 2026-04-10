import pandas as pd
import numpy as np
import json
import os

print("--- Generating Realistic Drift Telemetry ---")

# 1. Load the baseline stats
with open("artifacts/baseline_stats.json", 'r') as f:
    baseline = json.load(f)

# 2. Target: ~0.5 KL Divergence (shifting means by ~0.7 to 1.0 standard deviations)
num_samples = 50 
live_data = pd.DataFrame()

# Apply the realistic shift math
live_data['age'] = np.random.normal(
    loc=baseline['age']['mean'] + (baseline['age']['std'] * 0.8), 
    scale=baseline['age']['std'], 
    size=num_samples
)

live_data['bmi'] = np.random.normal(
    loc=baseline['bmi']['mean'] + (baseline['bmi']['std'] * 0.9), 
    scale=baseline['bmi']['std'], 
    size=num_samples
)

live_data['blood_pressure'] = np.random.normal(
    loc=baseline['blood_pressure']['mean'] + (baseline['blood_pressure']['std'] * 1.0), 
    scale=baseline['blood_pressure']['std'], 
    size=num_samples
)

live_data['previous_admissions'] = np.random.normal(
    loc=baseline['previous_admissions']['mean'] + (baseline['previous_admissions']['std'] * 0.7), 
    scale=baseline['previous_admissions']['std'], 
    size=num_samples
)

live_data['cholesterol'] = np.random.normal(
    loc=baseline['cholesterol']['mean'] + (baseline['cholesterol']['std'] * 0.85), 
    scale=baseline['cholesterol']['std'], 
    size=num_samples
)

# 3. Overwrite the inference logs
os.makedirs("artifacts", exist_ok=True)
live_data.to_csv("artifacts/inference_logs.csv", index=False)

print("✅ Successfully injected realistic drift data into artifacts/inference_logs.csv!")
print("Now run: python monitor.py")
