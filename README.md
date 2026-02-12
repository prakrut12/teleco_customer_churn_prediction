# 📊 Telco Customer Churn Prediction System

A machine learning project that predicts whether a telecom customer is likely to churn using business-relevant features.

## 🚀 Project Overview

This project uses a real-world Telco Customer Churn dataset to build a classification model that identifies high-risk customers. The goal is to help businesses take preventive action before customers leave.

## 🧠 ML Concepts Demonstrated

- Data preprocessing & feature engineering
- Handling class imbalance
- One-hot encoding for categorical features
- Train-test split
- Random Forest Classifier
- Precision, Recall, F1-score evaluation
- ROC-AUC score
- Confusion Matrix analysis
- Feature Importance interpretation
- Model deployment using Streamlit

## 📊 Model Performance

- Accuracy: ~75%
- Churn Recall: ~71%
- ROC-AUC: ~0.80+
- Class weighting used to improve churn detection

## 🖥️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

  customer_churn_prediction/
│
├── data/
├── models/
├── train.py
├── app.py
└── README.md


## ▶️ How to Run

1. Install dependencies:
pip install -r requirements.txt


2. Train the model:

python train.py


3. Run the app:

python -m streamlit run app.py

## 📁 Project Structure

