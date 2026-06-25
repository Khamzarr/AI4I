import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    df=pd.read_csv("ai4i2020.csv")
    df.head()
    return df, np, pd


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    # 5. Базовая статистика
    df.describe()
    return


@app.cell
def _(df):
    df["Type"].value_counts()
    return


@app.cell
def _(df):
    df["Machine failure"].value_counts()
    return


@app.cell
def _(df):
    df.groupby("Type")["Machine failure"].sum()
    return


@app.cell
def _(df):
    # Выберем только числовые колонки для корреляции
    numeric_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
                    "Torque [Nm]", "Tool wear [min]", "Machine failure"]
    df[numeric_cols].corr()
    return (numeric_cols,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


@app.cell
def _(df, plt):
    failure_by_type = df.groupby("Type")["Machine failure"].agg(["sum", "count"])
    failure_by_type["failure_rate"] = failure_by_type["sum"] / failure_by_type["count"] * 100

    plt.figure(figsize=(8, 5))
    failure_by_type["failure_rate"].plot(kind="bar", color=["#e74c3c", "#f39c12", "#2ecc71"])
    plt.title("Failure Rate by Equipment Type (%)", fontsize=14)
    plt.ylabel("Failure Rate %", fontsize=12)
    plt.xlabel("Equipment Type", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt):
    plt.figure(figsize=(12, 6))
    df[df["Machine failure"] == 0]["Torque [Nm]"].hist(bins=50, alpha=0.6, label="Normal", color="#2ecc71", density=True)
    df[df["Machine failure"] == 1]["Torque [Nm]"].hist(bins=50, alpha=0.6, label="Failure", color="#e74c3c", density=True)
    plt.xlabel("Torque [Nm]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Torque Distribution: Normal vs Failure", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, numeric_cols, plt, sns):
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt):
    plt.figure(figsize=(12, 6))
    df[df["Machine failure"] == 0]["Tool wear [min]"].hist(bins=50, alpha=0.6, label="Normal", color="#2ecc71", density=True)
    df[df["Machine failure"] == 1]["Tool wear [min]"].hist(bins=50, alpha=0.6, label="Failure", color="#e74c3c", density=True)
    plt.xlabel("Tool wear [min]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Tool Wear Distribution: Normal vs Failure", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # Кодируем категориальный признак Type
    df_ml = df.copy()
    le = LabelEncoder()
    df_ml["Type_encoded"] = le.fit_transform(df_ml["Type"])

    # Выбираем признаки (X) и целевую переменную (y)
    feature_cols = ["Type_encoded", "Air temperature [K]", "Process temperature [K]", 
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    X = df_ml[feature_cols]
    y = df_ml["Machine failure"]

    # Разделение: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train failures: {y_train.sum()} / {len(y_train)} ({y_train.mean()*100:.2f}%)")
    print(f"Test failures: {y_test.sum()} / {len(y_test)} ({y_test.mean()*100:.2f}%)")
    return (
        LogisticRegression,
        StandardScaler,
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        classification_report,
        confusion_matrix,
        feature_cols,
        roc_auc_score,
        train_test_split,
        y_test,
        y_train,
    )


@app.cell
def _(LogisticRegression, X_test_scaled, X_train_scaled, y_train):
    # Обучаем Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Предсказания
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("Model trained!")
    return model, y_pred, y_pred_proba


@app.cell
def _(
    classification_report,
    confusion_matrix,
    feature_cols,
    model,
    roc_auc_score,
    y_pred,
    y_pred_proba,
    y_test,
):
    # Метрики
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n=== ROC-AUC ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Важность признаков
    print("\n=== Feature Importance (Coefficients) ===")
    for name, coef in zip(feature_cols, model.coef_[0]):
        print(f"{name:30s}: {coef:+.4f}")
    return


@app.cell
def _(
    LogisticRegression,
    X_test_scaled,
    X_train_scaled,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    y_test,
    y_train,
):
    from sklearn.utils.class_weight import compute_class_weight

    # Вариант 1: class_weight='balanced'
    model_balanced = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model_balanced.fit(X_train_scaled, y_train)

    y_pred_balanced = model_balanced.predict(X_test_scaled)
    y_pred_proba_balanced = model_balanced.predict_proba(X_test_scaled)[:, 1]

    print("=== Balanced Logistic Regression ===")
    print(classification_report(y_test, y_pred_balanced, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_balanced):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_balanced)}")
    return


@app.cell
def _(
    X_test_scaled,
    X_train_scaled,
    classification_report,
    confusion_matrix,
    feature_cols,
    roc_auc_score,
    y_test,
    y_train,
):
    from sklearn.ensemble import RandomForestClassifier

    # Baseline Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

    print("=== Random Forest (balanced) ===")
    print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

    # Feature importance — переименовали name в fname
    print("\n=== Feature Importance ===")
    for fname, importance in zip(feature_cols, rf.feature_importances_):
        print(f"{fname:30s}: {importance:.4f}")
    return (y_pred_proba_rf,)


@app.cell
def _(classification_report, confusion_matrix, np, y_pred_proba_rf, y_test):
    from sklearn.metrics import precision_recall_curve

    precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_test, y_pred_proba_rf)

    best_idx_rf = np.where(recalls_rf >= 0.6)[0]
    if len(best_idx_rf) > 0:
        best_idx_rf = best_idx_rf[np.argmax(precisions_rf[best_idx_rf])]
        best_threshold_rf = thresholds_rf[best_idx_rf]
    
        y_pred_rf_threshold = (y_pred_proba_rf >= best_threshold_rf).astype(int)
        print(f"Optimal threshold: {best_threshold_rf:.3f}")
        print(classification_report(y_test, y_pred_rf_threshold, target_names=["Normal", "Failure"]))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf_threshold)}")
    precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_test, y_pred_proba_rf)

    best_idx_rf = np.where(recalls_rf >= 0.6)[0]
    if len(best_idx_rf) > 0:
        best_idx_rf = best_idx_rf[np.argmax(precisions_rf[best_idx_rf])]
        best_threshold_rf = thresholds_rf[best_idx_rf]
    
        y_pred_rf_threshold = (y_pred_proba_rf >= best_threshold_rf).astype(int)
        print(f"Optimal threshold: {best_threshold_rf:.3f}")
        print(classification_report(y_test, y_pred_rf_threshold, target_names=["Normal", "Failure"]))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf_threshold)}")
    return


@app.cell
def _(
    X_test_scaled,
    X_train_scaled,
    classification_report,
    confusion_matrix,
    feature_cols,
    roc_auc_score,
    y_test,
    y_train,
):
    from xgboost import XGBClassifier

    # XGBoost с балансировкой через scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    xgb.fit(X_train_scaled, y_train)

    y_pred_xgb = xgb.predict(X_test_scaled)
    y_pred_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

    print("=== XGBoost ===")
    print(classification_report(y_test, y_pred_xgb, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_xgb)}")

    # Feature importance — переименовали переменные
    print("\n=== Feature Importance (XGBoost) ===")
    for feature_name, feat_importance in zip(feature_cols, xgb.feature_importances_):
        print(f"{feature_name:30s}: {feat_importance:.4f}")
    return (scale_pos_weight,)


@app.cell
def _(
    X_test_scaled,
    X_train_scaled,
    classification_report,
    confusion_matrix,
    feature_cols,
    roc_auc_score,
    scale_pos_weight,
    y_test,
    y_train,
):
    from lightgbm import LGBMClassifier

    lgb = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )

    lgb.fit(X_train_scaled, y_train)

    y_pred_lgb = lgb.predict(X_test_scaled)
    y_pred_proba_lgb = lgb.predict_proba(X_test_scaled)[:, 1]

    print("=== LightGBM ===")
    print(classification_report(y_test, y_pred_lgb, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lgb):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lgb)}")

    print("\n=== Feature Importance (LightGBM) ===")
    for f_name, f_imp in zip(feature_cols, lgb.feature_importances_):
        print(f"{f_name:30s}: {f_imp:.4f}")
    return


@app.cell
def _(
    X_test_scaled,
    X_train_scaled,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    y_test,
    y_train,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Конвертируем данные в тензоры
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

    # Создаём DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Определяем MLP
    class MLP(nn.Module):
        def __init__(self, input_size):
            super(MLP, self).__init__()
            self.layer1 = nn.Linear(input_size, 64)
            self.layer2 = nn.Linear(64, 32)
            self.layer3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.sigmoid = nn.Sigmoid()
    
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.sigmoid(self.layer3(x))
            return x

    # Инициализация
    model_nn = MLP(X_train_scaled.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)

    # Взвешивание классов (аналог scale_pos_weight)
    pos_weight = torch.FloatTensor([(y_train == 0).sum() / (y_train == 1).sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Обучение
    epochs = 1000
    model_nn.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_nn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Предсказания
    model_nn.eval()
    with torch.no_grad():
        y_pred_proba_nn = model_nn(X_test_tensor).numpy().flatten()
        y_pred_nn = (y_pred_proba_nn >= 0.5).astype(int)

    print("\n=== Neural Network (MLP) ===")
    print(classification_report(y_test, y_pred_nn, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_nn):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_nn)}")
    return


@app.cell
def _(
    X_test,
    X_train,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    y_test,
    y_train,
):
    from catboost import CatBoostClassifier

    cat = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.1,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=False
    )

    cat.fit(X_train, y_train)  # CatBoost не нужен scaler!

    y_pred_cat = cat.predict(X_test)
    y_pred_proba_cat = cat.predict_proba(X_test)[:, 1]

    print("=== CatBoost ===")
    print(classification_report(y_test, y_pred_cat, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_cat):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_cat)}")
    return (y_pred_proba_cat,)


@app.cell
def _(
    classification_report,
    confusion_matrix,
    np,
    roc_auc_score,
    y_pred_proba_cat,
    y_test,
):
    from sklearn.metrics import precision_recall_curve as prc

    precisions_cat, recalls_cat, thresholds_cat = prc(y_test, y_pred_proba_cat)

    # Ищем порог с recall >= 0.75 и максимальной precision
    best_idx_cat = np.where(recalls_cat >= 0.75)[0]
    if len(best_idx_cat) > 0:
        best_idx_cat = best_idx_cat[np.argmax(precisions_cat[best_idx_cat])]
        best_threshold_cat = thresholds_cat[best_idx_cat]
    
        y_pred_cat_threshold = (y_pred_proba_cat >= best_threshold_cat).astype(int)
        print(f"Optimal threshold: {best_threshold_cat:.3f}")
        print(classification_report(y_test, y_pred_cat_threshold, target_names=["Normal", "Failure"]))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_cat_threshold)}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_cat):.4f}")
    return (prc,)


@app.cell
def _(StandardScaler, df, pd, train_test_split):
    df_fe = df.copy()

    # Температурный градиент
    df_fe["Temp_diff"] = df_fe["Process temperature [K]"] - df_fe["Air temperature [K]"]

    # Мощность = Torque * RPM / 9.5488 (перевод в кВт)
    df_fe["Power_kW"] = df_fe["Torque [Nm]"] * df_fe["Rotational speed [rpm]"] / 9.5488

    # Износ на единицу мощности
    df_fe["Wear_per_power"] = df_fe["Tool wear [min]"] / (df_fe["Power_kW"] + 1)

    # Удельная нагрузка (Torque / Tool wear)
    df_fe["Stress_index"] = df_fe["Torque [Nm]"] / (df_fe["Tool wear [min]"] + 1)

    # One-hot encoding для Type
    df_fe = pd.get_dummies(df_fe, columns=["Type"], prefix="Type")

    # Обновляем признаки
    feature_cols_fe = [c for c in df_fe.columns if c not in ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]]
    print(f"New features: {feature_cols_fe}")

    X_fe = df_fe[feature_cols_fe]
    y_fe = df_fe["Machine failure"]

    X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42, stratify=y_fe)

    scaler_fe = StandardScaler()
    X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
    X_test_fe_scaled = scaler_fe.transform(X_test_fe)

    print(f"Train: {X_train_fe.shape}, Test: {X_test_fe.shape}")
    return (
        X_test_fe,
        X_train_fe,
        feature_cols_fe,
        scaler_fe,
        y_test_fe,
        y_train_fe,
    )


@app.cell
def _(
    X_test_fe,
    X_train_fe,
    classification_report,
    confusion_matrix,
    feature_cols_fe,
    roc_auc_score,
    y_test_fe,
    y_train_fe,
):
    from catboost import CatBoostClassifier as CBC

    cat_fe = CBC(
        iterations=100,
        depth=5,
        learning_rate=0.1,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=False
    )

    cat_fe.fit(X_train_fe, y_train_fe)

    y_pred_cat_fe = cat_fe.predict(X_test_fe)
    y_pred_proba_cat_fe = cat_fe.predict_proba(X_test_fe)[:, 1]

    print("=== CatBoost with Feature Engineering ===")
    print(classification_report(y_test_fe, y_pred_cat_fe, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test_fe, y_pred_proba_cat_fe):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_fe, y_pred_cat_fe)}")

    print("\n=== Feature Importance (with FE) ===")
    for feat_name, feat_imp in zip(feature_cols_fe, cat_fe.feature_importances_):
        print(f"{feat_name:35s}: {feat_imp:.4f}")
    return cat_fe, y_pred_proba_cat_fe


@app.cell
def _(
    classification_report,
    confusion_matrix,
    np,
    prc,
    roc_auc_score,
    y_pred_proba_cat_fe,
    y_test_fe,
):
    precisions_cat_fe, recalls_cat_fe, thresholds_cat_fe = prc(y_test_fe, y_pred_proba_cat_fe)

    best_idx_cat_fe = np.where(recalls_cat_fe >= 0.75)[0]
    if len(best_idx_cat_fe) > 0:
        best_idx_cat_fe = best_idx_cat_fe[np.argmax(precisions_cat_fe[best_idx_cat_fe])]
        best_threshold_cat_fe = thresholds_cat_fe[best_idx_cat_fe]
    
        y_pred_cat_fe_threshold = (y_pred_proba_cat_fe >= best_threshold_cat_fe).astype(int)
        print(f"Optimal threshold: {best_threshold_cat_fe:.3f}")
        print(classification_report(y_test_fe, y_pred_cat_fe_threshold, target_names=["Normal", "Failure"]))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test_fe, y_pred_cat_fe_threshold)}")
        print(f"ROC-AUC: {roc_auc_score(y_test_fe, y_pred_proba_cat_fe):.4f}")
    return


@app.cell
def _(X_test_fe, cat_fe, scaler_fe):
    import joblib
    from sklearn.pipeline import Pipeline

    # Сохраняем scaler + модель
    pipeline_final = Pipeline([
        ('scaler', scaler_fe),
        ('catboost', cat_fe)
    ])

    joblib.dump(pipeline_final, 'ai4i_catboost_fe_pipeline.pkl')
    print("✓ Best model saved: ai4i_catboost_fe_pipeline.pkl")

    # Тест загрузки
    loaded = joblib.load('ai4i_catboost_fe_pipeline.pkl')
    test_sample = X_test_fe[:3]
    test_pred = loaded.predict(test_sample)
    test_proba = loaded.predict_proba(test_sample)[:, 1]
    print(f"\nTest predictions: {test_pred}")
    print(f"Test probabilities: {test_proba}")
    return


if __name__ == "__main__":
    app.run()
