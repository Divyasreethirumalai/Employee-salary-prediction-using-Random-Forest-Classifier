# Employee-salary-prediction-using-Random-Forest-Classifier
A Streamlit-based ML web app that predicts whether a person earns more than $50K using demographic and work-related features via Random Forest.

# 🔍 Overview
This project was developed as part of the **AICTE-IBM Internship Program**. The app takes user inputs such as age, education, occupation, work hours, and gender to predict income category (`<=50K` or `>50K`) using a trained machine learning model.

# 🚀 Features
- 📊 Interactive Web App with **Streamlit**
- 🧠 **Random Forest Classifier** Model
- 🔧 User-friendly sliders and dropdowns for input
- 📈 Model Confidence Display
- 📊 Visual insights: Income Distribution, Work Hours, and Age Scatter Plot
- 💾 Encoded & trained on the cleaned **Adult Census Dataset**
  
# 📁 Project Structure
├── app.py                # Streamlit app interface
├── model_train.py        # ML model training script
├── adult_dataset.csv     # Cleaned input dataset
├── README.md             # Project documentation

# ⚠️ Important Note
> The trained model file (`income_model.pkl`) and encoder files are **not uploaded** to GitHub due to size limitations.  
> 👉 Please run `model_train.py` locally to generate them before launching the app.

# ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib & Seaborn
- Joblib
  
# 🧠 Model Details
- **Algorithm**: Random Forest Classifier  
- **Target**: `income` (binary: `>50K` or `<=50K`)  
- **Input Features**:  
  - Age  
  - Education  
  - Occupation  
  - Hours-per-week  
  - Gender
    
## 💡 How to Run

### 1. Clone the Repository

git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction

### 2. Install Dependencies

pip install pandas scikit-learn streamlit joblib matplotlib seaborn

### 3. Train the Model

python model_train.py

### 4. Run the Web App

streamlit run app.py



## 👩‍💻 Developed By

**Divyasree T**  
AICTE IBM Internship Participant


