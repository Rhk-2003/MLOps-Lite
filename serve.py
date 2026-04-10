from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from datetime import datetime

# Initialize the FastAPI app
app = FastAPI(title="Readmission Prediction API (MLOps Lite)")

# Load the trained model on startup
MODEL_PATH = "artifacts/model.joblib"
LOG_PATH = "artifacts/inference_logs.csv"
model = joblib.load(MODEL_PATH)

# Define the expected input schema using Pydantic
class PatientData(BaseModel):
    age: int
    bmi: float
    blood_pressure: float
    previous_admissions: int
    cholesterol: float

# Function to log predictions asynchronously so it doesn't slow down the API
def log_prediction(patient_dict: dict, prediction: int, probability: float):
    # Add timestamp and prediction results to the data
    log_entry = patient_dict.copy()
    log_entry['timestamp'] = datetime.now().isoformat()
    log_entry['predicted_readmission'] = prediction
    log_entry['probability'] = probability
    
    df = pd.DataFrame([log_entry])
    
    # If the file doesn't exist, write headers; otherwise, append
    write_header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode='a', header=write_header, index=False)

@app.post("/predict")
async def predict(patient: PatientData, background_tasks: BackgroundTasks):
    # Convert input to dictionary, then to DataFrame for the model
    patient_dict = patient.dict()
    input_df = pd.DataFrame([patient_dict])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Log the data in the background
    background_tasks.add_task(log_prediction, patient_dict, int(prediction), float(probability))
    
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "status": "success"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
