# Multiple-Disease-Prediction

## Problem Statement

Early and accurate disease detection is crucial for effective treatment.

Traditional methods rely on manual diagnosis — time-consuming and prone to human error.

Need for an AI-based system that can predict multiple diseases from patient data efficiently.

## Objectives

✅ Build a unified ML system to:

Predict the presence of Parkinson’s, Kidney, and Liver diseases.

Use data-driven insights for early detection.

Provide doctors/patients with quick predictions via a Streamlit web interface.

Improve prediction accuracy using optimized models.

## Data Preprocessing

🧹 Steps followed:

Handling missing values

Encoding categorical variables

Outlier detection and capping

Feature scaling using StandardScaler

Train-Test split (80-20)

SMOTE for class imbalance (if required)

## Exploratory Data Analysis (EDA)

📊 EDA insights:

Distribution of numerical & categorical features

Correlation heatmaps

Boxplots to identify outliers

Disease distribution (Healthy vs Diseased)

Feature importance analysis

## Machine Learning Models

### 🏗️ Models Tested:

Logistic Regression

Random Forest

Support Vector Machine

XGBoost (Best Performing)

### ✅ Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Curve

## Model Development Workflow

Machine Learning Pipeline Steps:

Data Loading

Preprocessing

Feature Selection

Model Training

Hyperparameter Tuning (GridSearchCV)

Model Evaluation

Model Saving using Joblib

Streamlit Deployment


## Streamlit Web App

🖥️ Features:

User-friendly UI

Takes input from users (symptoms or metrics)

Select disease type from dropdown

Displays prediction result and probability

Model loaded via joblib

🧩 Backend: Trained ML models

🎨 Frontend: Streamlit Interface

## System Architecture

Flowchart:
User Input → Preprocessing → Model Prediction → Result Display

(You can show this as a simple block diagram.)

## Advantages

✅ Accurate and fast predictions
✅ Supports multiple diseases in one platform
✅ Easy to use for doctors and patients
✅ Scalable and extendable to more diseases

## Future Enhancements

Add more diseases (Heart, Diabetes, Cancer)

Integrate Deep Learning models (CNN/LSTM)

Deploy as a cloud API

Connect to hospital databases for real-time input

## Conclusion

✨ The Multi-Disease Prediction System efficiently predicts multiple diseases using machine learning.
✅ Helps in early detection and treatment.
✅ Demonstrates the power of AI in healthcare.

## Tools and Technologies 

Programming Language: Python 

Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn 

Frontend: Streamlit 
