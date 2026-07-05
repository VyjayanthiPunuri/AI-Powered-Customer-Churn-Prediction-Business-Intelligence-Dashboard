# AI-Powered-Customer-Churn-Prediction-Business-Intelligence-Dashboard
An end-to-end AIpowered Customer Churn Prediction and Business Intelligence Dashboard built using Python, Scikit-Learn, Streamlit, and Plotly.The project includes data preprocessing,exploratory data analysis,feature engineering machine learning model training,live churn prediction,interactive dashboards,and business insights for customer retention.
# 📊 ChurnShield – AI-Powered Customer Churn Prediction Dashboard

## Overview

ChurnShield is an end-to-end Data Science project developed to predict customer churn using Machine Learning. The project analyzes customer demographics, service usage, contract information, and billing data to identify customers who are likely to leave a telecom company.

The project also includes an interactive Streamlit dashboard that allows business users to explore customer insights, visualize churn trends, and perform live churn predictions.

---

# Objectives

- Perform data cleaning and preprocessing
- Explore customer behavior through interactive visualizations
- Engineer meaningful features
- Train and compare Machine Learning models
- Predict customer churn
- Build an interactive business dashboard
- Provide business recommendations for customer retention

---

# Dataset

**Dataset:** IBM Telco Customer Churn Dataset

- Total Customers: **7,043**
- Features: **21**
- Target Variable: **Churn**

The dataset contains customer demographic information, account details, subscribed services, billing information, and churn status.

---

# Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Models
      │
      ▼
Model Evaluation
      │
      ▼
Streamlit Dashboard
```

---

# Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- Random Forest
- Logistic Regression

### Dashboard

- Streamlit

### Model Persistence

- Joblib

---

# Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best-performing model was saved as:

```
best_model.pkl
```

---

# Dashboard Features

## Executive Dashboard

- KPI Cards
- Customer Distribution
- Contract Analysis
- Revenue Overview
- Monthly Charges
- Customer Tenure

---

## Exploratory Data Analysis

- Customer Filters
- Interactive Plotly Charts
- Correlation Heatmap
- Scatter Plots
- Histograms
- Box Plots
- Sunburst Chart

---

## Live Prediction

Users can enter customer information and receive:

- Churn Prediction
- Churn Probability
- Risk Level
- Business Recommendation

---

## Model Performance

Displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Feature Importance
- Model Comparison

---

## Business Report

Provides:

- Executive Summary
- Business Insights
- Customer Retention Strategy
- Risk Analysis
- Downloadable Report

---

# Project Structure

```
ChurnShield
│
├── app.py
├── cleaned_telco.csv
├── best_model.pkl
├── preprocessing_pipeline.pkl
├── model_results.csv
├── requirements.txt
├── README.md
│
├── notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_explainability.ipynb
│
└── reports
```

---

# Installation

Clone the repository

```bash
git clone <repository_url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# Results

The project successfully predicts customer churn using Machine Learning and presents business insights through an interactive Streamlit dashboard.

Key capabilities include:

- Customer churn prediction
- Business intelligence dashboards
- Interactive visualizations
- Customer risk analysis
- Retention recommendations

---

# Future Improvements

- SHAP Explainability
- XGBoost Integration
- Customer Login
- Prediction History
- Cloud Deployment
- Automated Report Generation

---

# Author

**Vyjayanthi Punuri**

BS-MS Computer Science & Business Management

Vishwa Vishwani Institute of Systems and Management

2026

---

# License

This project is developed for educational and internship purposes.
