import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer

# ==============================================================================
# 1. Data Ingestion & JSON Validation
# ==============================================================================

# Simulate incoming raw data in JSON format from fitness devices
def generate_raw_data_json(n_samples=1000, n_anomalies=10):
    np.random.seed(42)
    data = []
    
    # Simulate a user's normal heart rate and activity
    base_hr = np.random.normal(70, 5, n_samples)
    base_steps = np.random.normal(5000, 1000, n_samples)
    
    # Inject some anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    for i in range(n_samples):
        record = {
            "timestamp": f"2025-09-17T00:00:00Z",
            "device_id": "device_XYZ123",
            "metrics": {
                "heart_rate_bpm": float(base_hr[i]),
                "steps_count": int(base_steps[i]),
                "battery_level": 95
            }
        }
        
        # Introduce a high heart rate anomaly and a low step count anomaly
        if i in anomaly_indices[:5]:
            record["metrics"]["heart_rate_bpm"] += np.random.uniform(50, 80)
        elif i in anomaly_indices[5:]:
            record["metrics"]["steps_count"] = np.random.randint(0, 100)
            
        data.append(record)
        
    return json.dumps(data, indent=2)

# Data Validation Function
# Checks for correct data types and presence of key fields
def validate_data(json_data):
    try:
        data = json.loads(json_data)
        validated_data = []
        for record in data:
            # Check for required fields and data types
            if (isinstance(record.get('timestamp'), str) and
                isinstance(record.get('metrics', {}).get('heart_rate_bpm'), (float, int)) and
                isinstance(record.get('metrics', {}).get('steps_count'), int)):
                validated_data.append(record)
            else:
                print(f"Skipping invalid record: {record}")
        return validated_data
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return None

# ==============================================================================
# 2. Data Preprocessing & Feature Engineering
# ==============================================================================

def preprocess_and_load(validated_data):
    df = pd.json_normalize(validated_data)
    df = df.rename(columns={
        'metrics.heart_rate_bpm': 'heart_rate',
        'metrics.steps_count': 'steps_count'
    })
    
    # Simple feature engineering: calculate a heart rate to step ratio
    df['hr_to_steps_ratio'] = df['heart_rate'] / (df['steps_count'] + 1)
    
    return df[['heart_rate', 'steps_count', 'hr_to_steps_ratio']]

# ==============================================================================
# 3. Anomaly Detection Pipeline
# ==============================================================================

# Define preprocessing steps for the model
# StandardScaler is vital for algorithms sensitive to feature scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), ['heart_rate', 'steps_count', 'hr_to_steps_ratio'])
    ])

# Define the anomaly detection model pipeline
# It chains the preprocessing and the final IsolationForest model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('detector', IsolationForest(contamination=0.01, random_state=42)) 
])

# ==============================================================================
# 4. Execution and Output
# ==============================================================================

if __name__ == "__main__":
    # --- Step 1: Ingest and Validate Data ---
    print("Step 1: Ingesting and validating data...")
    raw_json_data = generate_raw_data_json()
    validated_records = validate_data(raw_json_data)
    
    if validated_records is None or not validated_records:
        print("Data validation failed. Exiting.")
    else:
        print(f"Successfully validated {len(validated_records)} records.\n")
        
        # --- Step 2: Preprocess Data ---
        print("Step 2: Preprocessing data and creating features...")
        df_processed = preprocess_and_load(validated_records)
        print("Processed Data Snapshot:")
        print(df_processed.head())
        print("-" * 50)
        
        # --- Step 3: Fit the Anomaly Detection Pipeline ---
        print("Step 3: Fitting the anomaly detection pipeline...")
        model_pipeline.fit(df_processed)
        print("Pipeline fitted successfully.\n")
        
        # --- Step 4: Predict and Get Scores ---
        print("Step 4: Predicting anomalies and generating scores...")
        df_processed['is_anomaly'] = model_pipeline.predict(df_processed)
        df_processed['anomaly_score'] = model_pipeline.decision_function(df_processed)
        
        # Get the actual anomalies and normal points
        anomalies = df_processed[df_processed['is_anomaly'] == -1]
        normal_data = df_processed[df_processed['is_anomaly'] == 1]
        
        # --- Step 5: Output and Visualization ---
        print("Step 5: Displaying results and visualization.")
        print("=" * 50)
        print("Detected Anomalies:")
        print(anomalies)
        print("-" * 50)
        
        print(f"\nTotal Anomalies Detected: {len(anomalies)} out of {len(df_processed)} records.")
        
        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.scatter(normal_data.index, normal_data['heart_rate'], c='b', label='Normal Data', s=20)
        plt.scatter(anomalies.index, anomalies['heart_rate'], c='r', label='Anomalies', s=50, edgecolors='k')
        plt.title('FitPulse Anomaly Detection: Heart Rate over Time')
        plt.xlabel('Record Index (Time)')
        plt.ylabel('Heart Rate (BPM)')
        plt.legend()
        plt.grid(True)
        plt.show()
