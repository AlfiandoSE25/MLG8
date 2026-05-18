import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# --- Precision Configurations ---
# Set global print options to 4 decimal places for professional assignment formatting
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(precision=4, suppress=True)

# 1. Data Collection & Loading 
# Loading the programmatic dating app dataset containing 50,000 records [cite: 12, 13]
df = pd.read_csv('dating_app_behavior_dataset.csv')

# 2.Exploratory Data Analysis (EDA)
print("Dataset Overview (Descriptive Statistics with 4-decimal precision):")
print(df.info())
print(df.describe())  # Numerical outputs will now strictly display 4 decimal places

# Visualizing the target variable distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='Match_Outcome', data=df, palette='Blues')
plt.title('Distribution of Match Outcomes')
plt.show()

# 3. Data Pre-processing
# Handling target column encoding to solve the 'le' definition error [cite: 26, 27]
le = LabelEncoder()
df['Match_Outcome'] = le.fit_transform(df['Match_Outcome'])

# Feature Selection: Drop unique identifiers and apply one-hot encoding [cite: 28]
features = df.drop(['User_ID', 'Match_Outcome'], axis=1)
X = pd.get_dummies(features, drop_first=True)
y = df['Match_Outcome']

# Splitting data into training and validation sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for distance-based SVM boundaries) [cite: 26, 27]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.Feature Extraction (PCA)
# Dimensionality reduction via PCA as per the ML pipeline instructions [cite: 28]
pca = PCA(n_components=0.95)  # Retain 95% of cumulative variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 5.Model Training
# Training SVM with an RBF kernel to handle non-linear interaction complexities [cite: 29, 31]
print("Training SVM model on PCA-reduced features...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_pca, y_train)

# 6.Model Evaluation
# Generating predictions for model performance comparison 
y_pred = svm_model.predict(X_test_pca)

print("\n" + "="*20 + " SVM PERFORMANCE METRICS " + "="*20)
# Format accuracy output to 4 decimal places
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report (Precision/Recall/F1-score locked to 4 decimals):")
# CRITICAL: digits=4 prevents sklearn from truncating your metrics to 2 decimal places
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
print("="*65)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')
plt.show()