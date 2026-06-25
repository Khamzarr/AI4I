import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import joblib
    import numpy as np

    app = FastAPI(title="AI4I Predictive Maintenance API")

    # Загружаем модель
    model = joblib.load('ai4i_catboost_fe_pipeline.pkl')

    # Структура входных данных
    class EquipmentData(BaseModel):
        air_temp: float          # Air temperature [K]
        process_temp: float      # Process temperature [K]
        rpm: int                 # Rotational speed [rpm]
        torque: float            # Torque [Nm]
        tool_wear: int           # Tool wear [min]
        equipment_type: str      # "L", "M" или "H"

    @app.post("/predict")
    def predict(data: EquipmentData):
        # Создаём фичи (включая Feature Engineering)
        temp_diff = data.process_temp - data.air_temp
        power_kw = data.torque * data.rpm / 9.5488
        wear_per_power = data.tool_wear / (power_kw + 1)
        stress_index = data.torque / (data.tool_wear + 1)
    
        # One-hot encoding для типа
        type_h = 1 if data.equipment_type == "H" else 0
        type_l = 1 if data.equipment_type == "L" else 0
        type_m = 1 if data.equipment_type == "M" else 0
    
        # Формируем вектор признаков (в том же порядке, что при обучении)
        features = np.array([[
            data.air_temp,
            data.process_temp,
            data.rpm,
            data.torque,
            data.tool_wear,
            temp_diff,
            power_kw,
            wear_per_power,
            stress_index,
            type_h,
            type_l,
            type_m
        ]])
    
        # Предсказание
        proba = model.predict_proba(features)[0][1]
        prediction = int(proba >= 0.903)  # наш оптимальный порог
    
        risk_level = "LOW" if proba < 0.5 else "MEDIUM" if proba < 0.8 else "HIGH"
    
        return {
            "failure_probability": round(float(proba), 4),
            "prediction": "FAILURE" if prediction == 1 else "NORMAL",
            "risk_level": risk_level,
            "threshold_used": 0.903
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "CatBoost with Feature Engineering", "roc_auc": 0.9832}

    return


if __name__ == "__main__":
    app.run()
