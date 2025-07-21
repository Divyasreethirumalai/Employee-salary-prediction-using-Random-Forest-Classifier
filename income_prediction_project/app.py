import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Smart Income Prediction App", page_icon="💼", layout="centered")

# Load model and encoders
model = joblib.load("income_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
df = pd.read_csv("adult_dataset.csv")
df = df.replace("?", pd.NA).dropna()

# Sidebar - About section
with st.sidebar:
    st.title("🔧 Settings & Info")
    st.markdown("""
    **Internship Project**
    
    AICTE IBM Internship  
    Developed by: DIVYASREE T  
    """)
    if st.button("📃 Show Sample Dataset"):
        st.write(df.head())

# Title
st.title("💼 Smart Income Prediction App")
st.markdown("🚀 Predict whether a person earns >50K or <=50K based on profile")

# Input Form
with st.form("prediction_form"):
    st.subheader("📝 Enter Details")
    
    age = st.slider("Age", 18, 70, 30)
    education = st.selectbox("Education", encoders['education'].classes_)
    occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    hours = st.slider("Work Hours per Week", 1, 100, 40)
    
    submitted = st.form_submit_button("🔍 Predict")

# Prediction
if submitted:
    edu = encoders['education'].transform([education])[0]
    occ = encoders['occupation'].transform([occupation])[0]
    gen = encoders['gender'].transform([gender])[0]

    prediction = model.predict([[age, edu, occ, hours, gen]])
    income = target_encoder.inverse_transform(prediction)[0]
    confidence = model.predict_proba([[age, edu, occ, hours, gen]])[0].max()

    st.success(f"💸 Predicted Income Category: **{income}**")
    st.info(f"📈 Model Confidence: `{confidence:.2f}`")

    # Smart tip
    if income == ">50K":
        st.markdown("🎉 Excellent! You belong to a high-income category!")
    else:
        st.markdown("💡 Tip: Upskilling and education could improve income potential.")

# Visualizations
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔹 Income Distribution")
    st.bar_chart(df['income'].value_counts())

with col2:
    st.markdown("### 🔹 Average Work Hours by Income")
    avg_hours = df.groupby('income')['hours-per-week'].mean()
    st.bar_chart(avg_hours)

# Age vs Hours Scatter Plot
st.markdown("### 🔹 Age vs Work Hours by Income")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="age", y="hours-per-week", hue="income", ax=ax)
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Project: Employee Salary Prediction")
