import pandas as pd
from ftfy import fix_text

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
file_path = "dating_app_behavior_dataset.csv"

try:
    df = pd.read_csv(file_path)
except Exception:
    df = pd.read_excel(file_path)

print("Dataset loaded:", df.shape)

# 2. Clean text data
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(lambda x: fix_text(x) if isinstance(x, str) else x)

# 3. Create target variable
target_map = {
    "Mutual Match": "Mutual Match",
    "Ghosted": "Ghosted",
    "Catfished": "Catfished",
    "Chat Ignored": "No Response",
    "No Action": "No Response"
}

df = df[df["match_outcome"].isin(target_map.keys())].copy()
df["relationship_outcome"] = df["match_outcome"].map(target_map)

print("\nTarget distribution:")
print(df["relationship_outcome"].value_counts())

# 4. Feature engineering
df["num_interests"] = df["interest_tags"].apply(
    lambda x: len([item.strip() for item in str(x).split(",")])
)

# 5. Define features and target
selected_features = [
    "message_sent_count",
    "bio_length",
    "likes_received",
    "app_usage_time_min",
    "swipe_right_ratio",
    "emoji_usage_rate",
    "last_active_hour",
    "mutual_matches",
    "profile_pics_count",
    "num_interests"
]

X = df[selected_features]
y = df["relationship_outcome"]

# 6. Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nTarget classes:", list(label_encoder.classes_))

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 8. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_scaled, y_train)

# 10. Predict and evaluate
predictions = log_reg_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print("\nLogistic Regression Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))
