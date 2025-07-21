import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv("adult 3 (1).csv")
df = df.replace("?", np.nan).dropna()

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target
X = df.drop("income", axis=1)
y = df["income"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Class Prediction")

st.markdown("Enter employee details to predict salary class (<=50K or >50K):")

# User input form
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
education = st.selectbox("Education", label_encoders['education'].classes_)
edu_num = st.slider("Education Number", 1, 16, 10)
marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
race = st.selectbox("Race", label_encoders['race'].classes_)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours = st.slider("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)
fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)

# Encoding inputs
input_data = pd.DataFrame({
    "age": [age],
    "workclass": [label_encoders['workclass'].transform([workclass])[0]],
    "fnlwgt": [fnlwgt],
    "education": [label_encoders['education'].transform([education])[0]],
    "educational-num": [edu_num],
    "marital-status": [label_encoders['marital-status'].transform([marital_status])[0]],
    "occupation": [label_encoders['occupation'].transform([occupation])[0]],
    "relationship": [label_encoders['relationship'].transform([relationship])[0]],
    "race": [label_encoders['race'].transform([race])[0]],
    "gender": [label_encoders['gender'].transform([gender])[0]],
    "capital-gain": [capital_gain],
    "capital-loss": [capital_loss],
    "hours-per-week": [hours],
    "native-country": [label_encoders['native-country'].transform([native_country])[0]]
})

if st.button("Predict Salary Class"):
    prediction = model.predict(input_data)[0]
    salary_class = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"Predicted Salary Class: {salary_class}")

    # Optional: Show feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))
