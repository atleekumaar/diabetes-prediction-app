
# 🩺 Diabetes Health Risk Predictor

An end-to-end Machine Learning web application built using **Support Vector Classifier (SVM)** and **Streamlit** to assess diabetes risk based on medical parameters. 
          web app link ==> https://diabetes-prediction-app-1256.streamlit.app/
          ||   [Diabetes Health Risk Predictor](https://github.com/atleekumaar/diabetes-prediction-app) | [Live Web App](https://diabetes-prediction-app-1256.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)

---

## 📌 Project Overview

This repository contains a full machine learning pipeline — from exploratory data analysis and model training to web deployment. The predictive engine uses a pre-trained **Support Vector Machine (SVM)** model, standardized via **StandardScaler**, to predict whether a patient is likely diabetic based on 8 clinical indicators (Glucose, BMI, Age, Insulin, etc.).

---

## 🚀 Key Features

- **Interactive Dark Dashboard** — Modern, responsive UI custom-styled with Streamlit and CSS.
- **Instant Risk Assessment** — Real-time predictions based on key health metrics.
- **Diagnostic Summary Cards** — Visual breakdown of critical indicators (Glucose, BMI, Blood Pressure, Age).
- **Downloadable Medical Reports** — Single-click export of patient diagnosis reports in `.txt` format.
- **Cached Inference Engine** — Fast model execution using `@st.cache_resource`.

---

## 🛠️ Repository Structure

```
diabetes-prediction-app/
│
├── Project 3 -Diabetes_Prediction (1).ipynb  # ML notebook (EDA, preprocessing, model training)
├── diabetes_model.pkl                      # Saved Support Vector Machine (SVM) model
├── scaler.pkl                              # Trained StandardScaler object
├── app.py                                  # Interactive Streamlit web application
└── requirements.txt                        # Python dependencies for deployment
```

---

## 📊 Model & Dataset Details

| Detail | Description |
|---|---|
| **Dataset Features** | Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age |
| **Algorithm** | Support Vector Machine — `svm.SVC(kernel='linear')` |
| **Preprocessing** | `StandardScaler` feature normalization |
| **Evaluation Accuracy** | ~77.3% test accuracy |
| **Tech Stack** | Python, Scikit-Learn, NumPy, Joblib, Streamlit |

---

## 💻 How to Run Locally

### Option 1: Standard `pip` Setup

**1. Clone the repository**
```bash
git clone https://github.com/atleekumaar/diabetes-prediction-app.git
cd diabetes-prediction-app
```

**2. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**3. Launch the Streamlit app**
```bash
python -m streamlit run app.py
```

### Option 2: Fast Setup via `uv`

**1. Create and activate a virtual environment**
```bash
uv venv

# Windows (CMD):
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**2. Install requirements & run**
```bash
uv pip install -r requirements.txt
uv run streamlit run app.py

---

