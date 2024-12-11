# import modules
import pandas as pd
import os
import matplotlib.pyplot as pt

# Import data from csv file
path = "C:/Users/Eric/Desktop/COMP_309_Data_Warehousing/Toronto_Bike_Thefts"
filename = 'bike_thefts.csv'
fullpath = os.path.join(path, filename)

# Read data
data = pd.read_csv(fullpath)
print(data.head(10))
print(data.columns.values)
data.dtypes
data.describe()
data.info()

dummy_month  = pd.get_dummies((data['OCC_MONTH']))


#Chinbo's Example:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from flask import Flask, request, jsonify

# Define the path and filename
path = "C:\\Users\\Kai\\Desktop\\Data_warehouse"
filename = 'theft.csv'

# Load the dataset
df = pd.read_csv(f"{path}\\{filename}")

# ---------------- Data Exploration ----------------

# Basic information about the dataset
print("Dataset Info:")
print(df.info())

# Summary statistics for numeric columns
print("\nSummary Statistics:")
print(df.describe())

# Unique values and counts for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns and Unique Values:")
for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())
    print()

# Range for numerical columns
numeric_columns = df.select_dtypes(include=['number']).columns
print("\nRange for Numerical Columns:")
for col in numeric_columns:
    print(f"{col}: {df[col].min()} to {df[col].max()}")

# ---------------- Missing Data Evaluation ----------------

# Missing data per column
missing_data = df.isnull().sum()
print("\nMissing Data Per Column:")
print(missing_data)

# ---------------- Handling Missing Data ----------------

# Fill missing values with the mean for numerical columns
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing values with mode for categorical columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify no missing values remain
print("\nRemaining Missing Data:")
print(df.isnull().sum())

# ---------------- Feature Engineering (Label Encoding) ----------------

# Use label encoding for categorical columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ---------------- Train-Test Split ----------------

# Prepare X and y (keep 'STATUS' in y)
X = df.drop(columns=['STATUS'])
y = df['STATUS']

# Sampling to reduce memory usage (if dataset is too large)
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.95, random_state=42)  # Keeping 5% of data for faster processing

# Split the sampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# ---------------- Scale Features ----------------

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Model Building ----------------

# Logistic Regression with liblinear solver
logistic_model = LogisticRegression(solver='liblinear', max_iter=2000)
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)

print("\nLogistic Regression Report:")
print(classification_report(y_test, logistic_predictions))

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

print("\nDecision Tree Report:")
print(classification_report(y_test, tree_predictions))

# ---------------- Model Evaluation ----------------

# Confusion Matrix for Logistic Regression
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, logistic_predictions))

# Confusion Matrix for Decision Tree
print("\nConfusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, tree_predictions))
from sklearn.metrics import roc_auc_score

# ROC AUC for multi-class classification (One-vs-Rest)
logistic_roc_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test_scaled), multi_class='ovr')
print(f"\nLogistic Regression ROC AUC Score (One-vs-Rest): {logistic_roc_auc:.2f}")

# ROC AUC for Decision Tree (One-vs-Rest)
tree_roc_auc = roc_auc_score(y_test, tree_model.predict_proba(X_test), multi_class='ovr')
print(f"Decision Tree ROC AUC Score (One-vs-Rest): {tree_roc_auc:.2f}")


# ---------------- Model Deployment ----------------

# Save the best model (e.g., logistic regression)
with open("model.pkl", "wb") as file:
    pickle.dump(logistic_model, file)

# Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = logistic_model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
