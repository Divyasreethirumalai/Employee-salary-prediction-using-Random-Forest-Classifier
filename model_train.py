import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("adult_dataset.csv")

# Clean data
df = df.replace("?", pd.NA).dropna()

# Select features and target
X = df[['age', 'education', 'occupation', 'hours-per-week', 'gender']]
y = df['income']

# Encode categorical features
encoders = {}
for col in ['education', 'occupation', 'gender']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "income_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ Model training complete!")
