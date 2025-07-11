# anomaly_detection_unsw.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Anomaly Detection | UNSW-NB15", layout="wide")
st.title("🔐 Anomaly Detection in Network Logs (UNSW-NB15)")
st.markdown("Upload a preprocessed UNSW-NB15 dataset to identify anomalous network behavior using Isolation Forest, One-Class SVM, or LOF.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])
contamination = st.sidebar.slider("Contamination Rate (expected anomalies)", 0.01, 0.30, 0.05, step=0.01)

file = st.sidebar.file_uploader("Upload CSV (UNSW-NB15 or subset)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    if 'label' in df.columns:
        y_true = df['label']  # 0: normal, 1: anomaly
        df = df.drop(columns=['label'])
    else:
        y_true = None

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Model selection
    if model_choice == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        fit_predict = model.fit_predict(X_scaled)
    elif model_choice == "One-Class SVM":
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        fit_predict = model.fit_predict(X_scaled)
    else:
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        fit_predict = model.fit_predict(X_scaled)

    y_pred = np.where(fit_predict == -1, 1, 0)  # 1=anomaly, 0=normal

    df_result = pd.DataFrame(X_scaled, columns=df.columns)
    df_result['Predicted Anomaly'] = y_pred
    if y_true is not None:
        df_result['True Label'] = y_true.values

    st.subheader("📌 Anomaly Detection Results")
    st.dataframe(df_result.head(20))

    # Plot anomaly count
    st.subheader("📊 Anomaly Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Predicted Anomaly', data=df_result, palette='Set2', ax=ax)
    ax.set_xticklabels(['Normal (0)', 'Anomaly (1)'])
    st.pyplot(fig)

    # Classification report if true labels exist
    if y_true is not None:
        st.subheader("📈 Evaluation Report")
        st.code(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    # SHAP explainability (only for Isolation Forest)
    if model_choice == "Isolation Forest":
        st.subheader("🔎 Feature Importance (SHAP)")
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)
        shap.summary_plot(shap_values, X_scaled, feature_names=df.columns, show=False)
        st.pyplot(bbox_inches='tight')

    # Export model (Isolation Forest or SVM only)
    if model_choice in ["Isolation Forest", "One-Class SVM"]:
        if st.button("📤 Export Trained Model"):
            joblib.dump(model, "anomaly_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(df.columns.tolist(), "features.pkl")
            st.success("Model, scaler, and feature list exported as .pkl files!")

    # Upload new logs for inference
    st.subheader("📉 Upload New Log Data for Inference")
    new_file = st.file_uploader("Upload New Logs CSV (same structure)", type=["csv"], key="new")
    if new_file:
        try:
            new_data = pd.read_csv(new_file)
            new_data = new_data[joblib.load("features.pkl")]
            new_data_scaled = joblib.load("scaler.pkl").transform(new_data)
            loaded_model = joblib.load("anomaly_model.pkl")
            new_preds = loaded_model.predict(new_data_scaled)
            final_preds = np.where(new_preds == -1, 1, 0)
            new_data['Predicted Anomaly'] = final_preds
            st.write("🔍 New Log Predictions")
            st.dataframe(new_data.head(20))
        except Exception as e:
            st.error(f"Prediction Failed: {e}")
else:
    st.warning("Please upload a CSV file to start analysis.")
