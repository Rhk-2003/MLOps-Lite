# ⚙️ Self-Monitoring ML System (MLOps Lite)

An automated, event-driven Machine Learning Operations (MLOps) pipeline. This system continuously monitors live clinical telemetry for statistical data drift using Kullback-Leibler (KL) Divergence. When distribution drift causes model degradation, the system automatically triggers a self-healing retraining workflow to generate a new model.

---

## 🚀 Key Performance Metrics

| Metric | Value |
|---|---|
| **Baseline Accuracy** | ~80.7% |
| **ROC-AUC** | 0.87 *(Random Forest)* |
| **Drift Detection Sensitivity** | ~90% at D_KL > 0.2 |
| **Degradation Catch** | ~10–15% accuracy drop in live environments |
| **Operational Efficiency** | ~40% reduction in manual retraining downtime via Continuous Training (CT) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Machine Learning** | Scikit-Learn, Pandas, NumPy |
| **MLOps Pipeline** | Automated Python Subprocesses, JSON Artifact Registries |
| **Deployment** | Docker |
| **Statistical Monitoring** | KL Divergence (Kullback-Leibler) |

---

## 🏗️ Architecture & Data Flow

```
Training Data
     │
     ▼
Baseline Model + baseline_stats.json
     │
     ▼
Live Prediction Requests ──► Telemetry CSV Logger
                                      │
                                      ▼
                           Drift Monitor (Scheduled Loop)
                           KL Divergence: live vs. baseline
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                   D_KL ≤ 0.2               D_KL > 0.2
                   ✅ Stable             🚨 Drift Detected
                                               │
                                               ▼
                                    Auto Retraining Pipeline
                                    (retrain.py → V2 model)
                                               │
                                               ▼
                                    Artifacts Overwritten
                                    System Restored ✅
```

1. **Data Ingestion & Baseline** — Trains the baseline model and saves the statistical distribution (Mean/Std) of training features into an artifact registry (`baseline_stats.json`).
2. **Model Inference** — Live prediction requests are logged to a CSV telemetry file.
3. **Continuous Drift Monitoring** — A scheduled loop service calculates KL Divergence between live telemetry and baseline data.
4. **Automated Retraining Trigger** — If `D_KL > 0.2` (significant data shift), the system autonomously runs the retraining pipeline, generates a V2 model, and overwrites the stale `.pkl` artifacts.

---

## ⚡ Quick Start (Dockerized)

This MLOps pipeline is fully containerized for easy deployment.

### 1. Build the Docker Image

```bash
docker build -t mlops-monitor .
```

### 2. Run the Monitoring Service

```bash
docker run -v $(pwd)/artifacts:/app/artifacts mlops-monitor
```

> **Note:** The `artifacts` volume is mounted so newly trained models are persisted back to your local machine.

---

## 🔬 Simulate & Test the Pipeline

Run locally without Docker to observe the full self-healing cycle in action.

### Step 1 — Train the Baseline

Establish the initial "healthy" model and save the statistical artifacts:

```bash
python train_baseline.py
```

### Step 2 — Inject Synthetic Data Drift

Simulate real-world demographic shifts (e.g., a creeping ~1.0 standard deviation shift in patient BMI and Blood Pressure):

```bash
python simulate_realistic_drift.py
```

### Step 3 — Run the Monitor

Watch the system detect drift (`KL > 0.2`), recognize the ~10% accuracy degradation, and automatically trigger retraining to restore system health:

```bash
python monitor.py
```

Expected terminal output sequence:

```
[MONITOR] Checking telemetry against baseline...
[MONITOR] 🚨 DRIFT DETECTED — D_KL = 0.31 (threshold: 0.2)
[MONITOR] Accuracy degradation confirmed (~11.4%). Triggering retraining...
[RETRAIN] Training V2 model on refreshed data...
[RETRAIN] ✅ New model saved. Artifacts updated. System health restored.
```
