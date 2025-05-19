import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Stroke Prediction using ML Models")

@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df.drop(['id'], axis=1, inplace=True)
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    df['age_bmi'] = df['age'] * df['bmi']
    df['ht_hd'] = df['hypertension'] * df['heart_disease']
    label_encoder = LabelEncoder()
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        df[col] = label_encoder.fit_transform(df[col])
    return df

df = load_data()

# Prepare training data
X = df.drop('stroke', axis=1)
y = df['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Select and train model
model_name = st.selectbox(
    "Choose a model to train and evaluate",
    ("Logistic Regression", "Decision Tree", "Random Forest", "KNN", "XGBoost")
)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

# --- PREDICTION SECTION ---
st.subheader("🧪 Try Your Own Prediction")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.slider("Average Glucose Level", 40.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict Stroke Risk")

    if submitted:
        encoder = LabelEncoder()
        input_data = {
            "gender": encoder.fit(["Male", "Female", "Other"]).transform([gender])[0],
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": encoder.fit(["No", "Yes"]).transform([ever_married])[0],
            "work_type": encoder.fit(["children", "Govt_job", "Never_worked", "Private", "Self-employed"]).transform([work_type])[0],
            "Residence_type": encoder.fit(["Urban", "Rural"]).transform([residence_type])[0],
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": encoder.fit(["formerly smoked", "never smoked", "smokes", "Unknown"]).transform([smoking_status])[0]
        }

        input_df = pd.DataFrame([input_data])
        input_df['age_bmi'] = input_df['age'] * input_df['bmi']
        input_df['ht_hd'] = input_df['hypertension'] * input_df['heart_disease']

        # Ensure same feature order
        final_input = input_df[X.columns]
        scaled_input = scaler.transform(final_input)
        prediction = model.predict(scaled_input)[0]

        st.markdown(f"### 🩺 Prediction: {'🧠 Stroke Risk' if prediction == 1 else '✅ No Stroke Risk'}")
