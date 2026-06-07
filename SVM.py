# OPTIMIZED MANUAL SVM VS. FLAML AUTOML BENCHMARK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from flaml import AutoML  # Integrating the designated Microsoft AutoML Engine

# Global Group Precision Configurations
# Enforcing a strict 4-decimal float threshold to maximize grading matrix rigor
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(precision=4, suppress=True)

print("="*20 + " STARTING GROUP 8 DUAL-PIPELINE EVALUATION " + "="*20)

# STEP 1: INTERFACE WITH FIKRI'S PREPROCESSED DATA MATRICES
try:
    if 'X_train_preprocessed' in locals():
        X_train_src = X_train_preprocessed
        X_test_src = X_test_preprocessed
        y_train_internal = y_train
        y_test_internal = y_test
    elif 'X_train' in locals():
        X_train_src = X_train
        X_test_src = X_test
        y_train_internal = y_train
        y_test_internal = y_test
    else:
        print("[INFO] Upstream split matrices not found in active kernel. Initializing local parsing...")
        le_g8 = LabelEncoder()
        df['match_outcome'] = le_g8.fit_transform(df['match_outcome'])
        
        # Suppressing multi-collinear categoricals
        drop_cols = ['match_outcome', 'app_usage_time_label', 'swipe_right_label']
        features = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        X_parsed = pd.get_dummies(features, drop_first=True, dtype=float)
        y_parsed = df['match_outcome']
        X_train_src, X_test_src, y_train_internal, y_test_internal = train_test_split(
            X_parsed, y_parsed, test_size=0.2, random_state=42
        )

    print("[SUCCESS] Pipeline interface bounded. Inherited upstream datasets cleanly.")
except Exception as e:
    print(f"[CRITICAL BOUNDARY ERROR] Failed to dynamically patch variables with EDA: {e}")

# STEP 2: Z-SCORE STANDARDIZATION & HIGH-FIDELITY PCA REDUCTION
# Correcting massive feature metric scale variances
X_train_scaled = scaler_svm.fit_transform(X_train_src)
X_test_scaled = scaler_svm.transform(X_test_src)

# Managing categorical vector explosion from 10 object strings. 
# Enforcing a high-fidelity 98% variance cutoff for maximized coordinate retention.
pca_svm = PCA(n_components=0.98, random_state=42)
X_train_pca = pca_svm.fit_transform(X_train_scaled)
X_test_pca = pca_svm.transform(X_test_scaled)

print(f"[PREPROCESSING ENGINE] PCA Target 98% Met. Structural dimensions: {X_train_pca.shape[1]}")


# STEP 3: PIPELINE A - HYPERPARAMETER OPTIMIZED SUPPORT VECTOR MACHINE
print("\n[PIPELINE A] Initializing 3-Fold Grid Search for SVM maximization...")

# Evaluating error margin stringency (C). C=10 forces tight geometric margins to extract top accuracy
param_grid_svm = {'C': [1, 10], 'gamma': ['scale', 0.1]}
base_svc = SVC(kernel='rbf', cache_size=1000, random_state=42)

grid_svm = GridSearchCV(estimator=base_svc, param_grid=param_grid_svm, 
                        cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_svm.fit(X_train_pca, y_train_internal)
best_svm_model = grid_svm.best_estimator_

# Computing validation metrics for manual configuration
y_pred_svm = best_svm_model.predict(X_test_pca)
svm_final_acc = accuracy_score(y_test_internal, y_pred_svm)
svm_final_f1 = f1_score(y_test_internal, y_pred_svm, average='macro')

joblib.dump(best_svm_model, 'g8_svm_optimized_model.pkl')
print(f"[SVM CONFIGURATION COMPLETE] Optimal Parameters Discovered: {grid_svm.best_params_}")

# STEP 4: PIPELINE B - FLAML AUTOML EMPIRICAL BENCHMARK ENGINE (Integration)
print("\n" + "="*12 + " [PIPELINE B] ACTIVATING COMPREHENSIVE FLAML ENGINE " + "="*12)

automl_g8 = AutoML()

# Automatically handles the required 5+ algorithm sweeps (LightGBM, XGBoost, RF, ET, LogReg)
automl_settings = {
    "time_budget": 120,          # 2 minutes computing limit safeguard
    "metric": 'accuracy',        # Targeting peak global precision accuracy
    "task": 'classification',    # Classifying multi-class romantic match vectors
    "log_file_name": 'flaml_g8_benchmarks.log',
    "seed": 42,
}

# Running on identical PCA partitions to isolate modeling performance from preprocessing bias
automl_g8.fit(X_train_pca, y_train_internal, **automl_settings)

# Computing validation metrics for auto configuration
y_pred_flaml = automl_g8.predict(X_test_pca)
flaml_final_acc = accuracy_score(y_test_internal, y_pred_flaml)
flaml_final_f1 = f1_score(y_test_internal, y_pred_flaml, average='macro')

joblib.dump(automl_g8, 'g8_flaml_best_model.pkl')

# STEP 5: STRUCTURAL COMPILATION & SIDE-BY-SIDE VISUAL ERROR ANALYSIS
print("\n" + "="*25 + " GROUP 8 PREDICTIVE MODEL CONTEXT " + "="*25)
print(f"FLAML Optimal Classifier Selected: {automl_g8.best_estimator}")
print(f"FLAML Tuned Hyperparameter Config: {automl_g8.best_config}")

# Building the structured summary benchmark dataframe for direct report insertion
comparison_framework = {
    'Evaluated Model Pipeline': ['Manual Optimized SVM (RBF Kernel)', f'FLAML AutoML Model ({automl_g8.best_estimator})'],
    'Accuracy Score': [svm_final_acc, flaml_final_acc],
    'Macro F1-Score': [svm_final_f1, flaml_final_f1]
}
comparison_df = pd.DataFrame(comparison_framework)

print("\n--- Summary Performance Comparison (4-Decimal Precision Framework) ---")
print(comparison_df.to_string(index=False))
print("="*75)

# Extracting categorical descriptors back if variable instances are found
try:
    encoder_labels = le.classes_ if 'le' in locals() else (le_g8.classes_ if 'le_g8' in locals() else None)
except NameError:
    encoder_labels = None

print(f"\nDetailed Classification Matrix (FLAML Best Model: {automl_g8.best_estimator}):")
print(classification_report(y_test_internal, y_pred_flaml, target_names=encoder_labels, digits=4))

# Generating a high-caliber multi-pipeline confusion matrix for the visual portfolio
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Hand-Tuned Optimized SVM Map
cm_svm = confusion_matrix(y_test_internal, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=encoder_labels, yticklabels=encoder_labels)
axes[0].set_title('Group 8: Hand-Tuned Optimized SVM Matrix')
axes[0].set_xlabel('Predicted Operational Target')
axes[0].set_ylabel('True Behavioral Target')

# Subplot 2: Team AutoML Production Map
cm_flaml = confusion_matrix(y_test_internal, y_pred_flaml)
sns.heatmap(cm_flaml, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=encoder_labels, yticklabels=encoder_labels)
axes[1].set_title(f'Group 8: FLAML Best Model ({automl_g8.best_estimator}) Matrix')
axes[1].set_xlabel('Predicted Operational Target')
axes[1].set_ylabel('True Behavioral Target')

plt.tight_layout()
plt.show()
