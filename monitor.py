import pandas as pd
import numpy as np
import json
import os
import subprocess

# Configuration paths
BASELINE_PATH = "artifacts/baseline_stats.json"
LOGS_PATH = "artifacts/inference_logs.csv"

# Threshold for KL Divergence. 
# > 0.1 is a warning, > 0.2 means significant drift (trigger retraining).
DRIFT_THRESHOLD = 0.2 

def calculate_kl_divergence(mu_p, std_p, mu_q, std_q):
    """
    Calculates the KL Divergence between two normal distributions.
    P = Live Data (inference)
    Q = Baseline Data (training)
    """
    # Add a tiny epsilon to prevent division by zero
    std_p = max(std_p, 1e-6)
    std_q = max(std_q, 1e-6)
    
    var_p = std_p ** 2
    var_q = std_q ** 2
    
    kl = np.log(std_q / std_p) + (var_p + (mu_p - mu_q) ** 2) / (2 * var_q) - 0.5
    return kl

def run_monitor():
    print("--- Starting MLOps Drift Monitor ---")
    
    if not os.path.exists(LOGS_PATH):
        print("No inference logs found yet. Skipping monitor.")
        return

    # 1. Load Baseline and Live Data
    with open(BASELINE_PATH, 'r') as f:
        baseline_stats = json.load(f)
        
    live_data = pd.read_csv(LOGS_PATH)
    
    # We need a minimum number of samples to calculate a meaningful distribution
    if len(live_data) < 5:
        print(f"Only {len(live_data)} records logged. Need at least 5 to check for drift.")
        return

    print(f"Analyzing {len(live_data)} live predictions...")
    
    # 2. Check each feature for drift
    drift_detected = False
    drift_report = {}

    features_to_monitor = ['age', 'bmi', 'blood_pressure', 'previous_admissions', 'cholesterol']

    for feature in features_to_monitor:
        # Get baseline stats (Q)
        mu_q = baseline_stats[feature]['mean']
        std_q = baseline_stats[feature]['std']
        
        # Calculate live stats (P)
        mu_p = live_data[feature].mean()
        std_p = live_data[feature].std()
        
        # Calculate KL Divergence
        kl_div = calculate_kl_divergence(mu_p, std_p, mu_q, std_q)
        drift_report[feature] = kl_div
        
        if kl_div > DRIFT_THRESHOLD:
            print(f"🚨 DRIFT DETECTED in '{feature}'! KL Divergence: {kl_div:.4f}")
            drift_detected = True
        else:
            print(f"✅ '{feature}' is stable. KL Divergence: {kl_div:.4f}")

    # 3. Trigger Retraining Workflow if necessary
    if drift_detected:
        print("\n[TRIGGER] Significant data drift detected (> 0.2). Initiating retraining workflow...")
        # Execute the retraining script automatically
        subprocess.run(["python", "retrain.py"])
    else:
        print("\n[STATUS] System healthy. No retraining required.")

import time

if __name__ == "__main__":
    print("Starting continuous monitoring service...")
    while True:
        run_monitor()
        # Sleep for 60 seconds before checking again
        # In a real hospital setting, this might be 24 hours (86400 seconds)
        time.sleep(60)
