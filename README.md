# 🫀 Heart Risk AI Dashboard

### An end-to-end Machine Learning web application that predicts the risk of heart disease using clinical health parameters. Built with Streamlit + Scikit-learn, this project provides real-time prediction, interactive dashboards, and downloadable medical reports.

# 🚀 Live Demo

## 📸 UI Preview

(Add screenshot here)

assets/ui_dashboard.png

## 📊 Risk Analysis Charts

(Add screenshot here)

assets/ui_charts.png

## 📄 PDF Report Output

(Add screenshot here)

assets/ui_pdf.png

# 🎯 Project Objective

To build a clinical decision-support system that:

- Predicts heart disease risk
- Visualizes health risk probability
- Provides explainable AI insights
- Generates downloadable medical reports

# 🧠 Features

### 🔍 Machine Learning

- Logistic Regression / trained classification model
- Data preprocessing with scaling & encoding
- Real-time inference pipeline

### 📊 Interactive Dashboard
- Risk probability visualization
- Feature impact analysis chart
- Medical-style UI using Streamlit

### 📄 PDF Medical Report
- Auto-generated patient report

- Includes:
1. Patient details
2. Risk prediction
3. Probability scores
4. Timestamp

### 🧾 Explainability
- Feature importance visualization
- Transparent model outputs

# 🛠️ Tech Stack

- Python 🐍
- Streamlit ⚡
- Scikit-learn 🤖
- Pandas 📊
- NumPy 🔢
- ReportLab 📄
- Joblib 💾

# 🏗️ Project Architecture

```
User Input (Streamlit UI)
        ↓
Data Preprocessing (Encoding + Scaling)
        ↓
ML Model (Logistic Regression / Classifier)
        ↓
Prediction Output (Risk Probability)
        ↓
Visualization + PDF Report Generation
```

📂 Project Structure

```
heart-risk-ai-dashboard/
│
├── app.py                  # Streamlit web app
├── heart.pkl               # Trained ML model
├── scaler.pkl              # Feature scaler
├── columns.pkl             # Feature columns
│
├── assets/                 # UI screenshots
│   ├── dashboard.png
│   ├── charts.png
│   └── pdf_report.png
│
├── requirements.txt
└── README.md

```

# ⚙️ How to Run Locally

1️⃣ Clone the repository

```
git clone https://github.com/your-username/heart-risk-ai-dashboard.git
cd heart-risk-ai-dashboard
```

2️⃣ Install dependencies

```
pip install -r requirements.txt
```

3️⃣ Run Streamlit app

```
streamlit run app.py
```

# 📦 Requirements

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- reportlab

# 📊 Model Input Features

1. Age
2. Gender
3. Chest Pain Type
4. Resting Blood Pressure
5. Cholesterol
6. Fasting Blood Sugar
7. ECG Results
8. Max Heart Rate
9. Exercise Induced Angina
10. ST Depression (Oldpeak)
11. ST Slope

# 📈 Output

- 🟢 Low Risk %
- 🔴 High Risk %
- 📊 Risk Probability Chart
- 🧠 Feature Importance
- 📄 Downloadable PDF Report

# 👨‍💻 Author

Made with ❤️ by Shubham Sinha | AI Engineer 

### ⭐ If you like this project

- Give it a ⭐ on GitHub — it helps a lot!