import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
file_path = "career_pred.csv"
df = pd.read_csv(file_path)


df.columns = df.columns.str.strip()

le_role = LabelEncoder()
df['ROLE'] = le_role.fit_transform(df['ROLE'])  


X = df.drop(columns=['ROLE'])
y = df['ROLE']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


with open('career_model.pkl', 'wb') as file:
    pickle.dump((model, le_role), file)

print("Model Trained and Saved!")
