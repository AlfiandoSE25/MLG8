import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load Dataset (Make sure the CSV is in the same folder as this script)
try:
    df = pd.read_csv('dating_app_behavior_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: dating_app_behavior_dataset.csv not found in this directory.")
    exit()

# 2. Data Preprocessing
# We use errors='ignore' so the code doesn't crash if a column name is slightly different
columns_to_drop = ['user_id', 'interest_tags', 'User_ID', 'ID'] 
df_clean = df.drop(columns=columns_to_drop, errors='ignore')

# Create the num_interests feature only if interest_tags exists
if 'interest_tags' in df.columns:
    df_clean['num_interests'] = df['interest_tags'].apply(lambda x: len(str(x).split(', ')))
else:
    print("Warning: 'interest_tags' not found. Skipping num_interests calculation.")

# Keep your target mapping
target_map = {
    "Mutual Match": "Mutual Match",
    "Ghosted": "Ghosted",
    "Catfished": "Catfished",
    "Chat Ignored": "No Response",
    "No Action": "No Response"
}
df_clean = df_clean[df_clean["match_outcome"].isin(target_map.keys())].copy()

# 3. Use all features
# Drop the target 'match_outcome' from X, and keep it for y
X = df_clean.drop(columns=['match_outcome'])
y = df_clean['match_outcome']

# TURN WORDS INTO NUMBERS (One-Hot Encoding)
# This handles gender, location_type, etc.
X = pd.get_dummies(X, drop_first=True)

print(f"Total features being used: {X.shape[1]}")

# 4. Encoding & Splitting
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Scaling & PCA
scaler = StandardScaler()
# This will now scale ALL features, including the new 0/1 columns from encoding
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA is great here because it will compress those many features back into 
# the most important "Principal Components"
pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 6. Model Training - Decision Tree
# max_depth=10 helps prevent overfitting (High Variance)
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train_pca, y_train)

# 7. Predictions & Evaluation
y_pred = dt_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nDecision Tree Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))