# 🔐 Anomaly Detection in Network Logs (UNSW-NB15)

This project is a powerful and interactive Streamlit web app designed to detect anomalies in network traffic logs using unsupervised machine learning models. Built on the real-world **UNSW-NB15 dataset**, it helps identify suspicious patterns that could indicate cyber attacks, intrusions, or abnormal behavior.

---

## 🚀 Features

- 📁 Upload any preprocessed UNSW-NB15 CSV or your own network logs
- 🧠 Model Options:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)
- ⚖️ Adjust anomaly contamination threshold
- 📊 Live visualization of:
  - Anomaly predictions
  - Anomaly distribution chart
  - Evaluation metrics (Precision, Recall, F1-Score, ROC AUC)
- 🔎 **SHAP Explainability** for feature importance
- 📤 Export trained models for later use
- 📉 Upload new logs for real-time prediction

---

## 🧪 Tech Stack

- **Python** 🐍
- **Streamlit** – for interactive UI
- **Scikit-learn** – Isolation Forest, One-Class SVM, LOF
- **SHAP** – model explainability
- **Pandas, NumPy, Matplotlib, Seaborn** – data handling and visualization
- **Joblib** – model saving and loading

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/anomaly-detection-unsw.git
cd anomaly-detection-unsw

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run anomaly_detection_unsw.py
```
---
## 📂 Sample Dataset
- You can test the app using this pre-created sample subset of UNSW-NB15:
  - 🔗 Download sample CSV
  - (https://github.com/yourusername/anomaly-detection-unsw/blob/main/unsw_nb15_sample_subset.csv)
---
## 🎯 Use-Cases
- Intrusion Detection Systems (IDS)
- Cybersecurity log analysis
- Outlier detection in any classification dataset
- ML explainability with SHAP
---
## 📈 Screenshot

### 🔍 Main App Interface
![Dashboard](https://github.com/FutureDevGIT/Anomaly_Detection_UNFW/raw/main/Screenshot_1.png)
---
### 📊 Anomaly Distribution Chart
![Anomalies](https://github.com/FutureDevGIT/Anomaly_Detection_UNFW/raw/main/Screenshot_2.png)
---
## 📝 License
- MIT License - free to use, modify, and contribute.
---
