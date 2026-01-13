import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Load Models & Metrics ---
MODEL_PATH = os.path.join('model', 'final_model.pkl')
METRICS_PATH = os.path.join('model', 'metrics.pkl')

try:
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
except FileNotFoundError:
    model = None
    metrics = {"accuracy": 0.73, "roc_auc": 0.79, "pr_auc": 0.76} # Fallback
    print("Warning: Model files not found. Ensure they are in the 'model/' directory.")

# --- Feature Engineering Logic ---
def preprocess_input(data):
    """
    Transforms form data into the exact format expected by the model pipeline.
    Replicates the logic found in EDA.ipynb.
    """
    # 1. Create DataFrame from raw input
    input_dict = {
        'age': [int(data['age'])], # Input is in years
        'gender': [int(data['gender'])], # 1=Female, 2=Male
        'height': [int(data['height'])], # cm
        'weight': [float(data['weight'])], # kg
        'ap_hi': [int(data['ap_hi'])],
        'ap_lo': [int(data['ap_lo'])],
        'smoke': [int(data['smoke'])], # 0 or 1
        'alco': [int(data['alco'])],   # 0 or 1
        'active': [int(data['active'])], # 0 or 1
        'cholesterol': [int(data['cholesterol'])], # 1, 2, or 3
        'gluc': [int(data['gluc'])] # 1, 2, or 3
    }
    df = pd.DataFrame(input_dict)

    # 2. Feature Engineering (Match EDA.ipynb)
    # BMI = weight / (height/100)^2
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Pulse Pressure = ap_hi - ap_lo
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # 3. One-Hot Encoding (Manual recreation of pd.get_dummies(drop_first=True))
    # We must create columns: cholesterol_2, cholesterol_3, gluc_2, gluc_3
    # Logic: If val is 2, col_2=1. If val is 3, col_3=1. If val is 1, both are 0.
    
    df['cholesterol_2'] = (df['cholesterol'] == 2).astype(int)
    df['cholesterol_3'] = (df['cholesterol'] == 3).astype(int)
    
    df['gluc_2'] = (df['gluc'] == 2).astype(int)
    df['gluc_3'] = (df['gluc'] == 3).astype(int)

    # 4. Drop original categorical columns not used by pipeline (if pipeline expects dummies)
    # The ColumnTransformer passes through specific columns. We keep 'age', etc.
    # We do NOT drop numeric cols as the scaler needs them.
    
    return df

# --- Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/insights')
def insights():
    # Pass metrics to the template
    return render_template('insights.html', metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = request.form
            
            # Preprocess
            processed_df = preprocess_input(form_data)
            
            # Predict
            if model:
                # Get probability of class 1 (Disease)
                prob = model.predict_proba(processed_df)[0][1]
                prob_percent = round(prob * 100, 1)
                
                # Determine Risk Level
                if prob_percent < 40:
                    risk_level = "Low Risk"
                    risk_color = "text-emerald-400"
                    risk_bg = "bg-emerald-500/20"
                elif prob_percent < 70:
                    risk_level = "Moderate Risk"
                    risk_color = "text-yellow-400"
                    risk_bg = "bg-yellow-500/20"
                else:
                    risk_level = "High Risk"
                    risk_color = "text-rose-500"
                    risk_bg = "bg-rose-500/20"
                
                return render_template('result.html', 
                                     prob=prob_percent, 
                                     risk=risk_level, 
                                     color=risk_color,
                                     bg=risk_bg)
            else:
                return "Model not loaded", 500

        except Exception as e:
            return f"Error processing prediction: {str(e)}", 400

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)