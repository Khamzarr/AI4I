from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Загружаем модель
model = joblib.load('ai4i_catboost_fe_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Feature Engineering
    temp_diff = data['process_temp'] - data['air_temp']
    power_kw = data['torque'] * data['rpm'] / 9.5488
    wear_per_power = data['tool_wear'] / (power_kw + 1)
    stress_index = data['torque'] / (data['tool_wear'] + 1)
    
    type_h = 1 if data['equipment_type'] == "H" else 0
    type_l = 1 if data['equipment_type'] == "L" else 0
    type_m = 1 if data['equipment_type'] == "M" else 0
    
    features = np.array([[
        data['air_temp'], data['process_temp'], data['rpm'], data['torque'],
        data['tool_wear'], temp_diff, power_kw, wear_per_power, stress_index,
        type_h, type_l, type_m
    ]])
    
    proba = model.predict_proba(features)[0][1]
    
    # Два порога для разных сценариев
    threshold_strict = 0.903   # для минимизации ложных тревог
    threshold_work = 0.75    # для максимизации найденных отказов
    
    prediction_strict = "FAILURE" if proba >= threshold_strict else "NORMAL"
    prediction_work = "FAILURE" if proba >= threshold_work else "NORMAL"
    
    risk_level = "LOW" if proba < 0.5 else "MEDIUM" if proba < 0.75 else "HIGH" if proba < 0.903 else "CRITICAL"
    
    return jsonify({
        "failure_probability": round(float(proba), 4),
        "risk_level": risk_level,
        "prediction_strict": prediction_strict,
        "prediction_work": prediction_work,
        "thresholds": {"strict": threshold_strict, "work": threshold_work}
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "CatBoost with FE", "roc_auc": 0.9832})

if __name__ == '__main__':
    app.run(debug=True, port=8000)