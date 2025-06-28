import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and train model on startup
df = pd.read_csv("data/raw/heart-disease.csv")
X = df.drop("target", axis=1)
y = df["target"]
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])
    prediction = model.predict(input_data)[0]
    return "Heart Disease Detected ✅" if prediction == 1 else "No Heart Disease ❌"

# Gradio Interface
inputs = [
    gr.Number(label="Age"),
    gr.Radio([0, 1], label="Sex (0 = female, 1 = male)"),
    gr.Dropdown([0, 1, 2, 3], label="Chest Pain Type (cp)"),
    gr.Number(label="Resting BP (trestbps)"),
    gr.Number(label="Cholesterol (chol)"),
    gr.Radio([0, 1], label="Fasting Blood Sugar > 120 (fbs)"),
    gr.Dropdown([0, 1, 2], label="Rest ECG (restecg)"),
    gr.Number(label="Max Heart Rate (thalach)"),
    gr.Radio([0, 1], label="Exercise Induced Angina (exang)"),
    gr.Number(label="ST Depression (oldpeak)"),
    gr.Dropdown([0, 1, 2], label="Slope of ST (slope)"),
    gr.Dropdown([0, 1, 2, 3], label="Major Vessels Colored (ca)"),
    gr.Dropdown([0, 1, 2, 3], label="Thalassemia (thal)")
]

gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs="text",
    title="Heart Disease Predictor",
    description="Predict the risk of heart disease using clinical inputs."
).launch()