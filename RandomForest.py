# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 2: DATA LOADING
# Load raw user behavior dataset from the dating application
df = pd.read_csv("dating_app_behavior_dataset.csv")

# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
print("=== DATASET METADATA INFORMATION ===")
df.info()

print("\n=== DESCRIPTIVE STATISTICAL SUMMARY ===")
print(df.describe())

print("\n=== MISSING VALUES INTEGRITY CHECK ===")
print(df.isnull().sum())

# STEP 4: FEATURE ENGINEERING
# Feature Extraction: Transform 'interest_tags' string into a numerical count
df['num_interests'] = df['interest_tags'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# STEP 5: TARGET PREPROCESSING (PROBLEM REFRAMING)
# Map complex interaction outcomes into a simplified binary format (Success vs Failed)
target_map = {
    "Mutual Match": "Success",
    "Ghosted": "Failure",
    "Catfished": "Failure",
    "Chat Ignored": "Failure",
    "No Action": "Failure"
}
df = df[df["match_outcome"].isin(target_map.keys())].copy()
df["relationship_outcome"] = df["match_outcome"].map(target_map)

print("\n=== REFRAMED TARGET CLASS DISTRIBUTION ===")
print(df["relationship_outcome"].value_counts())

# STEP 6: FEATURE SELECTION
# A. Select continuous and discrete numerical variables
numerical_features = [
    'bio_length', 'likes_received', 'app_usage_time_min', 
    'message_sent_count', 'emoji_usage_rate', 'swipe_right_ratio', 
    'mutual_matches', 'last_active_hour', 'num_interests'
]
X_num = df[numerical_features]

# B. Isolate nominal categorical features for encoder transformation
categorical_features = ['gender', 'location_type', 'education_level']

# STEP 7: ENCODING PIPELINE
# A. One-Hot Encoding: Convert textual categories into structural numeric flags
X_cat = pd.get_dummies(df[categorical_features], drop_first=True)

# B. Feature Matrix Assembly: Merge numerical matrix with encoded nominal matrix
X = pd.concat([X_num, X_cat], axis=1)

# C. Isolate Target Variable (y)
y = df['relationship_outcome']

# D. Label Encoding: Convert target text labels into an array of structural integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# STEP 8: TRAIN-TEST SPLIT
# Partition dataset arrays: 80% assigned for training, 20% reserved for validation testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# STEP 9: MODEL TRAINING (RANDOM FOREST)
# Construct the Random Forest model with constraints to avoid overfitting
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,          
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train the model to learn statistical patterns
rf_model.fit(X_train, y_train)

# STEP 10: PREDICTION & RESULTS VISUALIZATION
# A. Generate target class predictions on the unseen testing partition
y_pred = rf_model.predict(X_test)

# B. Compute and print global validation classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n=========================================")
print(f"🌟 EVALUATION ACCURACY: {accuracy*100:.2f}%")
print("=========================================\n")

# C. Print specific localized metrics (Precision, Recall, F1-Score)
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# D. Generate and plot Confusion Matrix chart for your project report
plt.figure(figsize=(9, 6))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=le.classes_, 
    yticklabels=le.classes_
)

plt.title("Confusion Matrix - Tuned Random Forest Model", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Predicted Labels", fontsize=10)
plt.ylabel("Actual Labels", fontsize=10)
plt.tight_layout()
plt.show()
