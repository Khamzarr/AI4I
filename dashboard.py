import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests

st.set_page_config(page_title="AI4I Predictive Maintenance", layout="wide")

# Заголовок
st.title("🔧 AI4I Predictive Maintenance Dashboard")
st.markdown("Industrial Analytics Platform — CatBoost Model (ROC-AUC: 0.983)")

# Боковая панель — ввод данных
st.sidebar.header("📊 Equipment Parameters")
equipment_type = st.sidebar.selectbox("Equipment Type", ["L", "M", "H"])
air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 298.1, 0.1)
process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 314.0, 308.6, 0.1)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, 1551, 10)
torque = st.sidebar.slider("Torque [Nm]", 3.0, 77.0, 42.8, 0.1)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 260, 0, 1)

# Кнопка предсказания
if st.sidebar.button("🔍 Predict Failure"):
    # Запрос к API
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json={
            "air_temp": air_temp,
            "process_temp": process_temp,
            "rpm": rpm,
            "torque": torque,
            "tool_wear": tool_wear,
            "equipment_type": equipment_type
        })
        result = response.json()
        
        # Отображение результатов
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Failure Probability", f"{result['failure_probability']*100:.1f}%")
        with col2:
            st.metric("Risk Level", result['risk_level'])
        with col3:
            st.metric("Prediction (Work)", result['prediction_work'])
        
        # График вероятности
        fig, ax = plt.subplots(figsize=(8, 2))
        prob = result['failure_probability']
        color = 'green' if prob < 0.5 else 'orange' if prob < 0.75 else 'red'
        ax.barh(['Probability'], [prob], color=color, height=0.3)
        ax.axvline(x=0.75, color='orange', linestyle='--', label='Work threshold')
        ax.axvline(x=0.903, color='red', linestyle='--', label='Strict threshold')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"API Error: {e}. Make sure Flask API is running on port 8000.")

# Загрузка датасета для анализа
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    return df

df = load_data()

# Вкладки
tab1, tab2, tab3 = st.tabs(["📈 Dataset Overview", "🔥 Correlations", "⚠️ Failure Analysis"])

with tab1:
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Failures", df['Machine failure'].sum())
    col3.metric("Failure Rate", f"{df['Machine failure'].mean()*100:.2f}%")
    
    st.dataframe(df.head(20))

with tab2:
    st.header("Feature Correlations")
    numeric_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
                    "Torque [Nm]", "Tool wear [min]", "Machine failure"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("Failure Analysis")
    failure_df = df[df['Machine failure'] == 1]
    st.write(f"Total failures: {len(failure_df)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Torque distribution
    df[df['Machine failure']==0]['Torque [Nm]'].hist(bins=30, alpha=0.6, label='Normal', ax=axes[0,0], color='green')
    df[df['Machine failure']==1]['Torque [Nm]'].hist(bins=30, alpha=0.6, label='Failure', ax=axes[0,0], color='red')
    axes[0,0].set_title('Torque Distribution')
    axes[0,0].legend()
    
    # Tool wear distribution
    df[df['Machine failure']==0]['Tool wear [min]'].hist(bins=30, alpha=0.6, label='Normal', ax=axes[0,1], color='green')
    df[df['Machine failure']==1]['Tool wear [min]'].hist(bins=30, alpha=0.6, label='Failure', ax=axes[0,1], color='red')
    axes[0,1].set_title('Tool Wear Distribution')
    axes[0,1].legend()
    
    # RPM distribution
    df[df['Machine failure']==0]['Rotational speed [rpm]'].hist(bins=30, alpha=0.6, label='Normal', ax=axes[1,0], color='green')
    df[df['Machine failure']==1]['Rotational speed [rpm]'].hist(bins=30, alpha=0.6, label='Failure', ax=axes[1,0], color='red')
    axes[1,0].set_title('RPM Distribution')
    axes[1,0].legend()
    
    # Failure by type
    failure_by_type = df.groupby('Type')['Machine failure'].agg(['sum', 'count'])
    failure_by_type['rate'] = failure_by_type['sum'] / failure_by_type['count'] * 100
    failure_by_type['rate'].plot(kind='bar', ax=axes[1,1], color=['red', 'orange', 'green'])
    axes[1,1].set_title('Failure Rate by Type (%)')
    axes[1,1].set_ylabel('Rate %')
    
    plt.tight_layout()
    st.pyplot(fig)