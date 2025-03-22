import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
file_path = "career_pred.csv"
df = pd.read_csv(file_path)

# Remove any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Encode categorical target variable (ROLE)
le_role = LabelEncoder()
df['ROLE'] = le_role.fit_transform(df['ROLE'])  # Encode career roles

# Select relevant features (all except ROLE)
X = df.drop(columns=['ROLE'])
y = df['ROLE']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model and encoders
with open('career_model.pkl', 'wb') as file:
    pickle.dump((model, le_role), file)

print("Model Trained and Saved!")
