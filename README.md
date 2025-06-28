# Heart Disease Prediction Machine Learning Model

---

## Problem Definition

Financial institutions receive thousands of loan applications daily. Manually evaluating these applications is time-consuming, prone to human error, and may lead to inconsistent decisions. To address this, a machine learning system is needed to automatically predict whether a loan should be approved based on the applicant's profile. To develop a machine learning model that accurately predicts the presence of heart disease in a patient based on medical attributes and clinical data.

## Problem Statement:

Cardiovascular diseases are among the leading causes of death globally. Early detection of heart disease can significantly improve treatment outcomes and reduce mortality rates. However, manual diagnosis is time-consuming and prone to human error. To address this, we aim to build a machine learning-based system that can assist healthcare professionals by predicting the likelihood of heart disease using patient data.

## Input:

A structured dataset containing various medical attributes such as:

Age

Gender

Chest pain type

Resting blood pressure

Cholesterol levels

Fasting blood sugar

Resting ECG results

Max heart rate achieved

Exercise-induced angina

ST depression

Number of major vessels

Thalassemia

## Output:

A binary classification:

1: Presence of heart disease

0: No presence of heart disease

---

## Project Architecture
ml_project/
├── src/
│   └── Model/
│       ├── data_loader.py       # Load and split data
│       ├── preprocessing.py     # Feature engineering and transformation
│       ├── model.py             # Model creation and persistence
│       ├── train.py             # Training pipeline
│       ├── evaluate.py          # Evaluation metrics
│       ├── predict.py           # Inference pipeline
│       └── __init__.py
├── config/
│   └── config.yaml              # Configurations for model/data/paths
├── data/
│   └── raw_data.csv             # Raw dataset
├── notebooks/
│   └── eda.ipynb                # EDA and visualizations
├── tests/
│   └── test_*.py                # Unit tests using pytest
├── main.py                      # Entry point for the ML pipeline
├── requirements.txt             # List of dependencies
├── setup.py                     # Package definition
├── README.md                    # Project documentation
└── .gitignore                   # Files to ignore in version control


