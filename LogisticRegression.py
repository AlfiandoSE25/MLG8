import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Save plots as PNG (no display needed in VSCode)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection  import (train_test_split, GridSearchCV,
                                       cross_val_score, StratifiedKFold,
                                       learning_curve)
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, roc_auc_score,
                                       roc_curve, precision_recall_curve,
                                       average_precision_score,
                                       f1_score, precision_score, recall_score)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output folder — all PNG plots will be saved here
OUTPUT_DIR = "lr_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  LOGISTIC REGRESSION — Dating App Behavior Prediction")
print("=" * 60)
print("✅ Libraries loaded.")
print(f"   Plots will be saved to:  ./{OUTPUT_DIR}/\n")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 : DATA LOADING  (smart auto-detect: xlsx / xls / csv)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_FILE = "dating_app_behavior_dataset.xlsx"

def load_dataset(filepath):
    """
    Auto-detect the real file format using magic bytes, then load it.
    Kaggle datasets are sometimes CSV saved with an .xlsx extension —
    this function handles that case automatically.
    """
    with open(filepath, "rb") as fh:
        magic = fh.read(8)

    # PK header (50 4B) → real .xlsx (ZIP-based Office Open XML)
    if magic[:2] == b"PK":
        print("   Detected format : Real XLSX (ZIP/Office Open XML)")
        return pd.read_excel(filepath, engine="openpyxl")

    # OLE2 header → legacy .xls (Excel 97-2003)
    if magic[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        print("   Detected format : Legacy XLS (Excel 97-2003)")
        return pd.read_excel(filepath, engine="xlrd")

    # Anything else → treat as plain-text CSV
    print("   Detected format : CSV / plain-text  "
          "(file has .xlsx extension but is actually CSV — common on Kaggle)")
    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.read_csv(filepath, sep="\t")

print(f"[Step 2] Loading dataset from '{DATASET_FILE}' ...")
df = load_dataset(DATASET_FILE)

print(f"   Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n=== DATASET METADATA ===")
df.info()

print(f"\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe().round(3).to_string())

print(f"\n=== MISSING VALUES CHECK ===")
missing = df.isnull().sum()
if missing.any():
    print(missing[missing > 0])
else:
    print("   No missing values detected ✅")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 : EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Generating EDA plots ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 3A. Raw match_outcome distribution
outcome_counts = df["match_outcome"].value_counts()
colors_bar = sns.color_palette("Set2", len(outcome_counts))
outcome_counts.plot(kind="bar", ax=axes[0], color=colors_bar, edgecolor="black")
axes[0].set_title("Raw Match Outcome Distribution", fontweight="bold")
axes[0].set_xlabel("Match Outcome")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)
for bar, val in zip(axes[0].patches, outcome_counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 150,
                 f"{val:,}", ha="center", va="bottom", fontsize=8)

# 3B. App usage time distribution
axes[1].hist(df["app_usage_time_min"], bins=40,
             color="steelblue", edgecolor="white", alpha=0.85)
axes[1].set_title("App Usage Time Distribution", fontweight="bold")
axes[1].set_xlabel("App Usage Time (min)")
axes[1].set_ylabel("Frequency")

# 3C. Swipe right ratio vs mutual matches (sample 2 000 points)
scatter_sample = df.sample(2000, random_state=RANDOM_STATE)
color_map = {
    "Mutual Match" : "green", "Ghosted"      : "red",
    "Catfished"    : "orange","Chat Ignored"  : "blue",
    "No Action"    : "purple"
}
for outcome, grp in scatter_sample.groupby("match_outcome"):
    axes[2].scatter(grp["swipe_right_ratio"], grp["mutual_matches"],
                    c=color_map.get(outcome, "gray"),
                    label=outcome, alpha=0.4, s=12)
axes[2].set_title("Swipe Right Ratio vs Mutual Matches", fontweight="bold")
axes[2].set_xlabel("Swipe Right Ratio")
axes[2].set_ylabel("Mutual Matches")
axes[2].legend(fontsize=7, loc="upper left")

plt.suptitle("EDA — Dating App Behavior Dataset",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "01_eda.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 : FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Feature engineering ...")
df["num_interests"] = df["interest_tags"].apply(
    lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
)
print(f"   'num_interests' added  →  "
      f"range [{df['num_interests'].min()}, {df['num_interests'].max()}], "
      f"mean {df['num_interests'].mean():.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 : TARGET PREPROCESSING — Binary Reframing
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] Reframing target variable to binary ...")

target_map = {
    "Mutual Match" : "Success",
    "Ghosted"      : "Failure",
    "Catfished"    : "Failure",
    "Chat Ignored" : "Failure",
    "No Action"    : "Failure",
}
df = df[df["match_outcome"].isin(target_map.keys())].copy()
df["relationship_outcome"] = df["match_outcome"].map(target_map)

counts = df["relationship_outcome"].value_counts()
print(f"\n=== REFRAMED TARGET CLASS DISTRIBUTION ===")
print(counts.to_string())
print(f"\n   Imbalance ratio : {counts.max() / counts.min():.2f}x")

# Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(counts.index, counts.values,
              color=["#2ecc71", "#e74c3c"], edgecolor="black", width=0.4)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{val:,}\n({val / counts.sum() * 100:.1f}%)",
            ha="center", fontsize=10)
ax.set_title("Reframed Binary Target Distribution", fontweight="bold")
ax.set_xlabel("Relationship Outcome")
ax.set_ylabel("Sample Count")
plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "02_target_distribution.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 : FEATURE SELECTION & ENCODING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 6] Feature selection & encoding ...")

numerical_features = [
    "bio_length", "likes_received", "app_usage_time_min",
    "message_sent_count", "emoji_usage_rate", "swipe_right_ratio",
    "mutual_matches", "last_active_hour", "num_interests",
]
categorical_features = ["gender", "location_type", "education_level"]

# One-Hot Encode categorical features
X_cat = pd.get_dummies(df[categorical_features], drop_first=True)

# Assemble final feature matrix
X = pd.concat([df[numerical_features], X_cat], axis=1)

# Target encoding
y = df["relationship_outcome"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)           # Failure=0, Success=1

print(f"   Total features  : {X.shape[1]}  "
      f"(numerical: {len(numerical_features)}, encoded categorical: {len(X_cat.columns)})")
print(f"   Feature names   : {list(X.columns)}")
print(f"   Class mapping   : {dict(zip(le.classes_, le.transform(le.classes_)))}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 : STRATIFIED TRAIN-TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 7] Stratified train-test split (80/20) ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded,       # preserves class ratio in both subsets
)
print(f"   Training set : {X_train.shape[0]:,} samples")
print(f"   Testing  set : {X_test.shape[0]:,} samples")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 : FEATURE SCALING — MANDATORY FOR LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 8] Applying StandardScaler (fit on train only — no data leakage) ...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # transform only on test

print(f"   Train means (first 3 features) : "
      f"{X_train_scaled.mean(axis=0)[:3].round(4)}")
print(f"   Train stds  (first 3 features) : "
      f"{X_train_scaled.std(axis=0)[:3].round(4)}")
print("   All means ≈ 0, all stds ≈ 1 ✅")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 : BASELINE MODEL (default parameters)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 9] Training baseline Logistic Regression (default params) ...")

lr_baseline = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs",
    max_iter=1000, random_state=RANDOM_STATE,
)
lr_baseline.fit(X_train_scaled, y_train)
y_pred_base = lr_baseline.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, y_pred_base)

print(f"   Baseline Accuracy : {baseline_acc * 100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 : HYPERPARAMETER TUNING — GridSearchCV
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 10] Hyperparameter tuning with GridSearchCV (5-fold) ...")
print("   This may take 1–3 minutes depending on your machine ...\n")

param_grid = {
    "C"            : [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty"      : ["l1", "l2"],
    "solver"       : ["liblinear", "saga"],
    "class_weight" : [None, "balanced"],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n   Best parameters  : {grid_search.best_params_}")
print(f"   Best CV accuracy : {grid_search.best_score_ * 100:.2f}%")

# Top 5 combinations
results_df = pd.DataFrame(grid_search.cv_results_)
top5 = (results_df
        .sort_values("mean_test_score", ascending=False)
        [["param_C", "param_penalty", "param_solver",
          "param_class_weight", "mean_test_score", "std_test_score"]]
        .head(5))
top5.columns = ["C", "Penalty", "Solver", "Class Weight", "Mean CV Acc", "Std"]
top5["Mean CV Acc"] = top5["Mean CV Acc"].apply(lambda x: f"{x*100:.2f}%")
top5["Std"]        = top5["Std"].apply(lambda x: f"±{x*100:.2f}%")
print("\n   Top 5 parameter combinations:")
print(top5.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 11 : FINAL MODEL EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 11] Evaluating tuned model on test set ...")

lr_best      = grid_search.best_estimator_
y_pred       = lr_best.predict(X_test_scaled)
y_pred_proba = lr_best.predict_proba(X_test_scaled)[:, 1]   # P(Success)

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)
avg_prec  = average_precision_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "=" * 55)
print(f"   🌟 TUNED LR ACCURACY  : {accuracy * 100:.2f}%")
print(f"   🎯 ROC-AUC SCORE      : {roc_auc:.4f}")
print(f"   📌 AVERAGE PRECISION  : {avg_prec:.4f}")
print("=" * 55)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 12 : CROSS-VALIDATION (10-fold)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 12] 10-fold stratified cross-validation ...")

cv_10fold = StratifiedKFold(n_splits=10, shuffle=True,
                             random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    lr_best, X_train_scaled, y_train,
    cv=cv_10fold, scoring="accuracy", n_jobs=-1,
)

print("   Cross-Validation Results:")
for i, score in enumerate(cv_scores, 1):
    bar = "█" * int(score * 40)
    print(f"     Fold {i:>2} : {score * 100:.2f}%  {bar}")
print(f"\n   Mean  : {cv_scores.mean() * 100:.2f}%")
print(f"   Std   : ±{cv_scores.std() * 100:.2f}%")
print(f"   Min   : {cv_scores.min() * 100:.2f}%")
print(f"   Max   : {cv_scores.max() * 100:.2f}%")

if cv_scores.std() < 0.01:
    print("\n   ✅ Low variance — model is stable and generalises well.")
elif cv_scores.std() < 0.02:
    print("\n   ⚠️  Moderate variance — acceptable.")
else:
    print("\n   ❌ High variance — possible overfitting.")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 13 : VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 13] Generating evaluation plots ...")

# ── 13A. Confusion Matrix + ROC Curve ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
annot = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(rv, rp)]
                  for rv, rp in zip(cm, cm_pct)])
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0], linewidths=0.5, linecolor="white")
axes[0].set_title("Confusion Matrix — Logistic Regression (Tuned)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Predicted Label", fontsize=10)
axes[0].set_ylabel("Actual Label", fontsize=10)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color="royalblue", lw=2.5,
             label=f"Logistic Regression  (AUC = {roc_auc:.4f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1.2,
             label="Random Classifier   (AUC = 0.50)")
axes[1].fill_between(fpr, tpr, alpha=0.08, color="royalblue")
axes[1].set_xlabel("False Positive Rate", fontsize=10)
axes[1].set_ylabel("True Positive Rate", fontsize=10)
axes[1].set_title("ROC Curve — Logistic Regression",
                  fontsize=11, fontweight="bold")
axes[1].legend(loc="lower right", fontsize=9)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])
axes[1].grid(alpha=0.3)

plt.suptitle("Model Evaluation", fontsize=13, fontweight="bold")
plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "03_confusion_roc.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {_path}")

# ── 13B. Feature Coefficients ────────────────────────────────────────────────
feature_names = list(X.columns)
coef          = lr_best.coef_[0]
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("AbsCoef", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"],
        color=colors, edgecolor="white", height=0.6)
ax.axvline(x=0, color="black", linewidth=1, linestyle="--", alpha=0.7)
for _, row in coef_df.iterrows():
    w = row["Coefficient"]
    ax.text(w + (0.005 if w >= 0 else -0.005),
            coef_df.index.get_loc(_) if False else
            list(coef_df["Feature"]).index(row["Feature"]),
            f"{w:.3f}", va="center",
            ha="left" if w >= 0 else "right", fontsize=7.5)
red_p   = mpatches.Patch(color="#e74c3c",
                          label="Positive → increases P(Success)")
green_p = mpatches.Patch(color="#2ecc71",
                          label="Negative → increases P(Failure)")
ax.legend(handles=[red_p, green_p], loc="lower right", fontsize=9)
ax.set_title("Feature Coefficients — Logistic Regression\n"
             "(Magnitude = importance  |  Sign = direction of effect)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Coefficient Value", fontsize=10)
plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "04_feature_coefficients.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {_path}")

print("\n   Top 5 features by absolute coefficient:")
top5_feat = coef_df.sort_values("AbsCoef", ascending=False).head(5)
print(top5_feat[["Feature", "Coefficient", "AbsCoef"]].to_string(index=False))

# ── 13C. Learning Curve + Precision-Recall Curve ─────────────────────────────
train_sizes, train_sc, val_sc = learning_curve(
    lr_best, X_train_scaled, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True,
                       random_state=RANDOM_STATE),
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
)
tr_mean, tr_std = train_sc.mean(axis=1), train_sc.std(axis=1)
va_mean, va_std = val_sc.mean(axis=1),   val_sc.std(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_sizes, tr_mean, "o-", color="steelblue",
             lw=2, label="Training Score")
axes[0].fill_between(train_sizes, tr_mean - tr_std,
                     tr_mean + tr_std, alpha=0.15, color="steelblue")
axes[0].plot(train_sizes, va_mean, "o-", color="darkorange",
             lw=2, label="Validation Score")
axes[0].fill_between(train_sizes, va_mean - va_std,
                     va_mean + va_std, alpha=0.15, color="darkorange")
axes[0].set_title("Learning Curve — Logistic Regression",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Training Set Size")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba)
baseline_rate = y_test.sum() / len(y_test)
axes[1].plot(rec_vals, prec_vals, color="purple", lw=2,
             label=f"LR (AP = {avg_prec:.4f})")
axes[1].axhline(y=baseline_rate, color="gray", linestyle="--",
                label=f"No-skill baseline = {baseline_rate:.2f}")
axes[1].set_title("Precision-Recall Curve — Logistic Regression",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "05_learning_pr_curve.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {_path}")

# ── 13D. Regularization Sensitivity (C sweep) ────────────────────────────────
print("\n[Step 13D] Regularization sensitivity sweep ...")
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accs, test_accs = [], []
for C in C_values:
    _lr = LogisticRegression(C=C, solver="liblinear",
                              max_iter=1000, random_state=RANDOM_STATE)
    _lr.fit(X_train_scaled, y_train)
    train_accs.append(accuracy_score(y_train, _lr.predict(X_train_scaled)))
    test_accs.append(accuracy_score(y_test,  _lr.predict(X_test_scaled)))

plt.figure(figsize=(9, 5))
plt.semilogx(C_values, [a * 100 for a in train_accs],
             "o-", color="steelblue", lw=2, label="Training Accuracy")
plt.semilogx(C_values, [a * 100 for a in test_accs],
             "o-", color="darkorange", lw=2, label="Test Accuracy")
plt.axvline(x=grid_search.best_params_["C"], color="red",
            linestyle="--", alpha=0.7,
            label=f"Best C = {grid_search.best_params_['C']}")
plt.xlabel("C  (Regularization Strength — log scale)", fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=10)
plt.title("Effect of Regularization Strength (C) on Model Accuracy",
          fontsize=11, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "06_regularization_sensitivity.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {_path}")
