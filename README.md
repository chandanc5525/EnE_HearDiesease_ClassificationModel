# Bank Loan Approval Prediction System

---

## Problem Definition

Financial institutions receive thousands of loan applications daily. Manually evaluating these applications is time-consuming, prone to human error, and may lead to inconsistent decisions. To address this, a machine learning system is needed to automatically predict whether a loan should be approved based on the applicant's profile.

---

## Objectives

- Build a machine learning model to classify loan approvals (Yes/No).

- Ensure modular, reusable, and production-ready code.

- Allow configuration-driven experimentation using YAML.

- Provide clean interfaces for training, evaluation, and inference.

- Create visualizations for Exploratory Data Analysis (EDA).

- Write unit tests to ensure code reliability.

- Support command-line and Python-based execution.

---

## Project Architecture
ml_project/
├── src/
│   └── bank_loan_approval/
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


