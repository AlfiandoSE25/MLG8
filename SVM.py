# INTEGRATED MODEL PIPELINE: MANUAL SVM VS. FLAML AUTOML COMPARISON
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from flaml import AutoML  # Import Microsoft's Fast & Lightweight AutoML engine

#  Precision Configurations 
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(precision=4, suppress=True)

# 1. Data Collection & Robustness Cleaning
df_ml = pd.read_csv('dating_app_behavior_dataset.csv')

# Stripping structural whitespaces from string columns to prevent category explosion
object_cols = df_ml.select_dtypes(include=['object']).columns
df_ml[object_cols] = df_ml[object_cols].apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 2. Label Encoding & Feature Engineering
le = LabelEncoder()
df_ml['Match_Outcome'] = le.fit_transform(df_ml['Match_Outcome'])

# Isolate feature matrix and drop irrelevant unique row identifiers
features = df_ml.drop(['User_ID', 'Match_Outcome'], axis=1)

# One-hot encoding categorical variables explicitly locked to float to bypass boolean output
X = pd.get_dummies(features, drop_first=True, dtype=float)
y = df_ml['Match_Outcome']

# Dataset Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Z-score Feature Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. High-Fidelity Feature Extraction via PCA
pca = PCA(n_components=0.98) # Retaining 98% variance for precise behavioral captures
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 5. Pipeline A: Manually Optimized Support Vector Machine (SVM)
print("\n" + "="*20 + " RUNNING OPTIMIZED SVM PIPELINE " + "="*20)

# Grid Search running 3-Fold Cross Validation targeting accuracy optimization
param_grid = {'C': [1, 10], 'gamma': ['scale', 0.1]}
base_svm = SVC(kernel='rbf', cache_size=1000, random_state=42)
grid_search = GridSearchCV(estimator=base_svm, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_pca, y_train)
best_svm_model = grid_search.best_estimator_

# Evaluate Manual SVM
y_pred_svm = best_svm_model.predict(X_test_pca)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='macro')

# 6. Pipeline B: FLAML Automated Machine Learning Comparison
print("\n" + "="*20 + " RUNNING FLAML AUTO-ML ENGINE " + "="*20)

automl = AutoML()

# Configure FLAML settings to evaluate diverse estimators (lgbm, xgboost, rf, extra_tree, lrl1)
automl_settings = {
    "time_budget": 120,          # Total compute time limit in seconds (2 minutes)
    "metric": 'accuracy',        # Target evaluation optimization metric
    "task": 'classification',    # Supervised classification tasks
    "log_file_name": 'flaml_dating_app.log',
    "seed": 42,
}

# Training FLAML on the exact same PCA processed dimensions for fair model comparison
automl.fit(X_train_pca, y_train, **automl_settings)

# Evaluate FLAML Best Selected Model
y_pred_flaml = automl.predict(X_test_pca)
flaml_acc = accuracy_score(y_test, y_pred_flaml)
flaml_f1 = f1_score(y_test, y_pred_flaml, average='macro')


# ==============================================================================
# 7. Persistence & Model Serialization
# ==============================================================================
joblib.dump(best_svm_model, 'svm_optimized_model.pkl')
joblib.dump(automl, 'flaml_best_automl.pkl')
print("\nBoth models serialized and saved to disk.")


# ==============================================================================
# 8. Comparative Matrix & Evaluation Outputs
# ==============================================================================
print("\n" + "="*25 + " FINAL MODEL COMPARISON " + "="*25)
print(f"FLAML Selected Best Estimator: {automl.best_estimator}")
print(f"FLAML Optimized Hyperparameters: {automl.best_config}")

# Create summary framework table for direct inclusion into your project report
comparison_data = {
    'Model Pipeline': ['Manual SVM (RBF Kernel + GridSearch)', f'FLAML AutoML ({automl.best_estimator})'],
    'Accuracy Metric': [svm_acc, flaml_acc],
    'Macro F1-Score': [svm_f1, flaml_f1]
}
comparison_df = pd.DataFrame(comparison_data)

print("\n--- Summary Benchmark Table (4-Decimal Precision Framework) ---")
print(comparison_df.to_string(index=False))
print("="*74)

print(f"\nDetailed Classification Report for FLAML Best Model ({automl.best_estimator}):")
print(classification_report(y_test, y_pred_flaml, target_names=le.classes_, digits=4))

# Chaos / Error Mapping: Confusion Matrix Visual Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Manual Optimized SVM Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_title('Optimized Manual SVM Confusion Matrix')
axes[0].set_xlabel('Predicted Operational Label')
axes[0].set_ylabel('True Behavioral Outcome')

# Subplot 2: FLAML Best Model Matrix
cm_flaml = confusion_matrix(y_test, y_pred_flaml)
sns.heatmap(cm_flaml, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1].set_title(f'FLAML Best AutoML Model ({automl.best_estimator}) Confusion Matrix')
axes[1].set_xlabel('Predicted Operational Label')
axes[1].set_ylabel('True Behavioral Outcome')

plt.tight_layout()
plt.show()
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')
plt.show()
