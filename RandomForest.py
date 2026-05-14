# 1.Import Library
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ftfy import fix_text

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 2.Load Dataset
df = pd.read_csv("dating_app_behavior_dataset.csv")
df.head()

# 3. Exploratory Data Analysis (EDA)
df.info()
df.describe()
df.isnull().sum()

# 4. Data Preprocessing
target_map = {
    "Mutual Match": "Mutual Match",
    "Ghosted": "Ghosted",
    "Catfished": "Catfished",
    "Chat Ignored": "No Response",
    "No Action": "No Response"
}

df = df[df["match_outcome"].isin(target_map.keys())].copy()
df["relationship_outcome"] = df["match_outcome"].map(target_map)
print(df["relationship_outcome"].value_counts())

# 5. Feature Selection

df['num_interests'] = df['interest_tags'].apply(lambda x: len(x.split(', ')))

top_features = [
    'message_sent_count', 'bio_length', 'likes_received',
    'app_usage_time_min', 'emoji_usage_rate', 'last_active_hour',
    'mutual_matches', 'profile_pics_count', 'num_interests'
]
X = df[top_features]
y = df['match_outcome']

#Encoding 
pd.get_dummies(df['gender'])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Split Data(Training 80%, Test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Model Training
#RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

#Accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy with Top 9 Features: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

#Confusion Matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Top 9 Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
